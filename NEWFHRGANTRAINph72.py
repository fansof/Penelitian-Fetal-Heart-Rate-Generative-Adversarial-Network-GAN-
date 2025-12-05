import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure we can import preprocessing/windowing from aFHRGANv2
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined in notebooks; fall back to CWD
    BASE_DIR = os.getcwd()
V2_DIR = os.path.join(BASE_DIR, 'aFHRGANv2')
if V2_DIR not in sys.path:
    sys.path.append(V2_DIR)

from NewFHRGANmodel131018nov import Generator, Discriminator
from LoadDatasetph72 import myDataset
from windowed_dataset import WindowedFromLongDataset


def set_seed(seed: int = 81):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    b = real_samples.size(0)
    alpha = torch.rand(b, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    # Use logits for WGAN-GP (apply GP to critic function before sigmoid)
    d_interpolates, _ = discriminator(interpolates, return_logits=True)
    ones = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(b, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


class FHRGANTrainer1310:
    """
    Trainer implementing CCWGAN-GP with Auxiliary Classifier, following the paper.
    1D model is used to match your preprocessing/windowing.
    """

    def __init__(
        self,
        data_folder,
        ph_file,
        latent_dim=100,
        num_classes=2,
        sequence_length=1000,
        batch_size=32,
        accumulation_steps = 16,
        n_critic=5,
        lambda_gp=10.0,
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        device='cuda',
        save_dir='./checkpoints_ccwgan_gp_1310',
        seed=81,
        train_normal=75,
        train_pathological=75,
    ):
        set_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.save_dir = save_dir
        self.accumulation_steps = accumulation_steps

        os.makedirs(save_dir, exist_ok=True)

        self.generator = Generator(
            latent_dim=latent_dim, num_classes=num_classes, sequence_length=sequence_length
        ).to(self.device)
        self.discriminator = Discriminator(
            num_classes=num_classes, sequence_length=sequence_length
        ).to(self.device)

        self._init_weights()

        self.opt_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-8)
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-8)

        # Cross-entropy on class logits (paper: log-likelihood)
        self.criterion_cls = nn.CrossEntropyLoss()

        base_dataset = myDataset(
            data_folder=data_folder,
            ph_file=ph_file,
            sequence_length=4000,
            mode='train',
            train_normal=train_normal,
            train_pathological=train_pathological,
            random_seed=seed,
        )
        windowed_dataset = WindowedFromLongDataset(base_dataset, win_len=1000)
        self.loader = DataLoader(
            windowed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )

        self.history = {k: [] for k in ['g_loss', 'd_loss', 'gp', 'cls_loss']}

    def _init_weights(self):
        for m in self.generator.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.discriminator.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # def _train_discriminator(self, real, labels):
    #     self.opt_D.zero_grad()
    #     b = real.size(0)

    #     z = torch.randn(b, self.latent_dim, device=self.device)
    #     labels_1h = F.one_hot(labels, self.num_classes).float()

    #     with torch.no_grad():
    #         fake = self.generator(z, labels_1h)

    #     # D returns probabilities here (sigmoid for source, softmax for class)
    #     real_Dp, real_Cp = self.discriminator(real, return_logits=False)   # (B,1), (B,C)
    #     fake_Dp, fake_Cp = self.discriminator(fake, return_logits=False)

    #     # pick the **correct-class** probability C_y via gather
    #     idx = labels.view(-1, 1)                      # (B,1)
    #     real_Cy = real_Cp.gather(1, idx).squeeze(1)   # (B,)
    #     fake_Cy = fake_Cp.gather(1, idx).squeeze(1)   # (B,)

    #     # Eq. (9): (E_fake D - E_real D) + λ·GP + (E_fake C_y - E_real C_y)
    #     source_term = fake_Dp.mean() - real_Dp.mean()
    #     class_term  = fake_Cy.mean() - real_Cy.mean()

    #     # GP (use logits inside GP if that stabilizes your run — up to you)
    #     gp = compute_gradient_penalty(self.discriminator, real.data, fake.data, self.device)

    #     d_loss = source_term + self.lambda_gp * gp + class_term
    #     d_loss.backward()
    #     self.opt_D.step()

    #     return d_loss.item(), gp.item(), class_term.item()

    def save_training_history(self, suffix='final'):
        """Save training history as both numpy and JSON for easy comparison"""
        import json
        
        # Save as numpy arrays
        np.save(os.path.join(self.save_dir, f'g_losses_{suffix}.npy'), 
                np.array(self.history['g_loss'], dtype=np.float32))
        np.save(os.path.join(self.save_dir, f'd_losses_{suffix}.npy'), 
                np.array(self.history['d_loss'], dtype=np.float32))
        np.save(os.path.join(self.save_dir, f'gp_{suffix}.npy'), 
                np.array(self.history['gp'], dtype=np.float32))
        np.save(os.path.join(self.save_dir, f'cls_loss_{suffix}.npy'), 
                np.array(self.history['cls_loss'], dtype=np.float32))
        
        # Save as JSON for readability
        with open(os.path.join(self.save_dir, f'training_history_{suffix}.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {self.save_dir}")


    def _train_discriminator(self, real, labels):
        # if not accumulate:
        #     self.opt_D.zero_grad()
        self.opt_D.zero_grad()
        b = real.size(0)

        z = torch.randn(b, self.latent_dim, device=self.device)
        labels_1h = F.one_hot(labels, self.num_classes).float()

        with torch.no_grad():
            fake = self.generator(z, labels_1h)

        # Use logits for WGAN-GP + CE (per paper)
        real_v, real_c = self.discriminator(real, return_logits=False)

        real_v = real_v.mean()

        fake_v, fake_c = self.discriminator(fake, return_logits=False)

        fake_v = fake_v.mean()

        wasserstein = fake_v - real_v
        
        gp = compute_gradient_penalty(self.discriminator, real.data, fake.data, self.device)


        cls_real = self.criterion_cls(real_c, labels)
        cls_fake = self.criterion_cls(fake_c, labels)
        # cls_loss = 0.5 * (cls_real + cls_fake)
        idx = labels.view(-1,1)
        cls_loss = fake_c.gather(1,idx).squeeze(1).mean()-real_c.gather(1,idx).squeeze(1).mean()
        # cls_loss = (cls_fake - cls_real)
        gptimeslambda = self.lambda_gp * gp
        # print(f'DISCRIMINATOR W = {wasserstein:.2f}\t cls = {cls_loss}\t real = {cls_real}\t fake = {cls_fake}')
        # loss_D = wasserstein + self.lambda_gp * gp + cls_loss
        loss_D = (wasserstein + cls_loss) + gptimeslambda
        loss_D.backward()
        self.opt_D.step()
        # if not accumulate:
        #     self.opt_D.step()

        return loss_D.item(), gp.item(), cls_loss.item()

    def _train_generator(self, b, real, labels):
        # if not accumulate:
        self.opt_G.zero_grad()
        # b = real.size(0)
        z = torch.randn(b, self.latent_dim, device=self.device)
        labels_1h = F.one_hot(labels, self.num_classes).float()

        fake = self.generator(z, labels_1h)
        real_v, real_c = self.discriminator(real, return_logits=False)
        fake_v, fake_c = self.discriminator(fake, return_logits=False)

        adv = fake_v.mean()-real_v.mean()
        idx = labels.view(-1, 1)
        cls = fake_c.gather(1,idx).squeeze(1).mean() - real_c.gather(1,idx).squeeze(1).mean()
        # print(f'"GENERATOR cls = {cls:.3f} ", "adv ={adv:.3f}"')
        # cls2 = fake_c.gather(1, idx). squeeze(1)
        # loss_G = (cls + adv) 
        loss_G = (cls-adv)
        # loss_G= -fake_v.mean() - fake_c.gather(1,idx).squeeze(1).mean()
        loss_G.backward()
        # if not accumulate:
        self.opt_G.step()
        return loss_G.item()

    # def _train_generator(self, b, labels):
    #     self.opt_G.zero_grad()
    #     z = torch.randn(b, self.latent_dim, device=self.device)
    #     labels_1h = F.one_hot(labels, self.num_classes).float()

    #     fake = self.generator(z, labels_1h)
    #     fake_Dp, fake_Cp = self.discriminator(fake, return_logits=False)

    #     idx = labels.view(-1, 1)
    #     fake_Cy = fake_Cp.gather(1, idx).squeeze(1)

    #     # minimize E_fake D  -  E_fake C_y
    #     g_loss = fake_Dp.mean() - fake_Cy.mean()
    #     g_loss.backward()
    #     self.opt_G.step()
    #     return g_loss.item()
    

    # def _train_generator(self, b, labels):
    #     self.opt_G.zero_grad()
    #     z = torch.randn(b, self.latent_dim, device=self.device)
    #     labels_1h = F.one_hot(labels, self.num_classes).float()

    #     fake = self.generator(z, labels_1h)
    #     fake_Dp, fake_Cp = self.discriminator(fake, return_logits=False)

    #     idx = labels.view(-1, 1)
    #     fake_Cy = fake_Cp.gather(1, idx).squeeze(1)

    #     # minimize E_fake D  -  E_fake C_y
    #     g_loss = fake_Dp.mean() - fake_Cy.mean()
    #     g_loss.backward()
    #     self.opt_G.step()
    #     return g_loss.item()


    def train(self, epochs=20):
        print(f"Starting CCWGAN-GP training on {self.device}")
        print(f"Batch size: {self.batch_size}, Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {self.batch_size * self.accumulation_steps}")

        for epoch in range(epochs):
            d_losses, g_losses, gps, clss = [], [], [], []
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (real, labels) in enumerate(pbar):
                real = real.float().to(self.device)
                if real.dim() == 2:
                    real = real.unsqueeze(1)
                labels = labels.long().to(self.device)

                is_last_accum = (batch_idx + 1) % self.accumulation_steps == 0

                d_loss_acc, gp_acc, cls_acc = 0.0, 0.0, 0.0
                for critic_iter in range(self.n_critic):
                    is_last_critic = critic_iter == self.n_critic - 1
                    # accumulate = not (is_last_accum and is_last_critic)
                    
                    dl, gp, cl = self._train_discriminator(real, labels)
                    d_loss_acc += dl
                    gp_acc += gp
                    cls_acc += cl

                d_loss_acc /= self.n_critic
                gp_acc /= self.n_critic
                cls_acc /= self.n_critic

                g_loss = self._train_generator(real.size(0), real, labels)
                # g_loss = self._train_generator(real.size(0), real, labels, accumulate=not is_last_accum)
                # g_loss = self._train_generator(real.size(0), labels)

                if is_last_accum:
                    d_losses.append(d_loss_acc)
                    g_losses.append(g_loss)
                    gps.append(gp_acc)
                    clss.append(cls_acc)

                # Clear cache every 50 batches
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

                pbar.set_postfix(D=f"{d_loss_acc:.4f}", G=f"{g_loss:.4f}", GP=f"{gp_acc:.4f}")
            
            # Clear cache at end of epoch
            torch.cuda.empty_cache()

            # for real, labels in pbar:
            #     real = real.float().to(self.device)
            #     if real.dim() == 2:
            #         real = real.unsqueeze(1)
            #     labels = labels.long().to(self.device)

            #     d_loss_acc, gp_acc, cls_acc = 0.0, 0.0, 0.0
            #     for _ in range(self.n_critic):
            #         dl, gp, cl = self._train_discriminator(real, labels)
            #         d_loss_acc += dl
            #         gp_acc += gp
            #         cls_acc += cl

            #     d_loss_acc /= self.n_critic
            #     gp_acc /= self.n_critic
            #     cls_acc /= self.n_critic
                
            #     g_loss = self._train_generator(real.size(0), real, labels)

            #     d_losses.append(d_loss_acc)
            #     g_losses.append(g_loss)
            #     gps.append(gp_acc)
            #     clss.append(cls_acc)

            #     pbar.set_postfix(D=f"{d_loss_acc:.4f}", G=f"{g_loss:.4f}", GP=f"{gp_acc:.4f}")

            self.history['d_loss'].append(float(np.mean(d_losses)))
            self.history['g_loss'].append(float(np.mean(g_losses)))
            self.history['gp'].append(float(np.mean(gps)))
            self.history['cls_loss'].append(float(np.mean(clss)))

            print(
                f"\nEpoch {epoch+1}: D={self.history['d_loss'][-1]:.4f} | "
                f"G={self.history['g_loss'][-1]:.4f} | GP={self.history['gp'][-1]:.4f} | "
                f"CLS={self.history['cls_loss'][-1]:.4f}"
            )

            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch + 1)
                self.generate_samples(epoch + 1, num_samples=5)
                self.save_training_history(suffix=f'epoch_{epoch+1}')

        self.save_checkpoint('final')
        self.save_training_history(suffix='final')
        self.plot_training_history()

    # def save_checkpoint(self, epoch):
    #     path = os.path.join(self.save_dir, f'generator_epoch_{epoch}.pth')
    #     state = {k: v.cpu() for k, v in self.generator.state_dict().items()}
    #     torch.save(state, path)
    #     print(f"Checkpoint saved: {path}")

    def save_checkpoint(self, epoch):
        # Save generator weights only (800 MB)
        g_path = os.path.join(self.save_dir, f'generator_epoch_{epoch}.pth')
        torch.save(self.generator.state_dict(), g_path)
        
        # Save discriminator weights only (800 MB)
        d_path = os.path.join(self.save_dir, f'discriminator_epoch_{epoch}.pth')
        torch.save(self.discriminator.state_dict(), d_path)
        
        print(f"💾 Checkpoint saved: {g_path} (~800 MB)")
        
        # Save training history separately (every 25 epochs or final)
        is_milestone = isinstance(epoch, int) and (epoch % 250 == 0)
        if is_milestone or epoch == 'final':
            self.save_training_history(suffix=f'epoch_{epoch}')


    def generate_samples(self, epoch, num_samples=5):
        self.generator.eval()
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]

        with torch.no_grad():
            for i in range(num_samples):
                label = torch.tensor([i % self.num_classes], device=self.device)
                label_1h = F.one_hot(label, self.num_classes).float()
                z = torch.randn(1, self.latent_dim, device=self.device)
                fake = self.generator(z, label_1h)
                # map from [-1,1] to approx bpm range [50,200]
                fake_bpm = (fake.cpu().numpy()[0, 0] + 1.0) * 75.0 + 50.0
                axes[i].plot(fake_bpm)
                axes[i].set_title(f"Generated (label={int(label.item())})")
                axes[i].set_ylabel('FHR (bpm)')
                axes[i].set_xlabel('Time (samples)')
                axes[i].grid(True)

        plt.tight_layout()
        out_path = os.path.join(self.save_dir, f'generated_samples_epoch_{epoch}.png')
        plt.savefig(out_path)
        plt.close()
        self.generator.train()

    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.history['d_loss']); axes[0, 0].set_title('Discriminator Loss'); axes[0, 0].grid(True)
        axes[0, 1].plot(self.history['g_loss']); axes[0, 1].set_title('Generator Loss'); axes[0, 1].grid(True)
        axes[1, 0].plot(self.history['gp']); axes[1, 0].set_title('Gradient Penalty'); axes[1, 0].grid(True)
        axes[1, 1].plot(self.history['cls_loss']); axes[1, 1].set_title('Classification Loss'); axes[1, 1].grid(True)
        plt.tight_layout()
        out_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(out_path)
        plt.close()


if __name__ == '__main__':
    # Defaults aligned to your environment
    DATA_FOLDER = '/home/fauzi/Documents/generateGAN_untukSKRIPSI/1APREPROCESSEDFILE/PREPROCESSED_OKTOBER'
    PH_FILE = '/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW/ph_labels.csv'

    trainer = FHRGANTrainer1310(
        data_folder=DATA_FOLDER,
        ph_file=PH_FILE,
        latent_dim=100,
        num_classes=2,
        sequence_length=1000,
        batch_size=32,
        accumulation_steps=1,
        n_critic=5,
        lambda_gp=10.0,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        device='cuda',
        save_dir=os.path.join(BASE_DIR, 'checkpoints_ccwgan_gp_1310v2ph72windowed18NOV'),
        seed=81,
        train_normal=120,
        train_pathological=120,
    )

    trainer.train(epochs=500)
    print('Training completed.')
