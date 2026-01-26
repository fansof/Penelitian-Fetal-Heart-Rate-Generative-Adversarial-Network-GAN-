
import os, json, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from CTGGAN1000 import Generator1000 as Generator, Discriminator1000 as Discriminator
from windowed_dataset import WindowedFromLongDataset

# IMPORTANT: your file must be named exactly 'octoberpreprocessing.py'
# and expose a class/function 'myDataset' that yields (x, y):
#   x: FloatTensor (1, 1000) in [-1, 1]
#   y: int label in {0,1}
from LoadDataset import myDataset

def gradient_penalty(critic, real_x, fake_x, labels, device, gp_lambda=10.0):
    B = real_x.size(0)
    eps = torch.rand(B, 1, 1, device=device)
    xhat = eps * real_x + (1 - eps) * fake_x
    xhat.requires_grad_(True)
    scores, _ = critic(xhat, labels)
    grad = torch.autograd.grad(
        outputs=scores, inputs=xhat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad = grad.view(B, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean() * gp_lambda
    return gp

def save_loss_plot(out_dir, g_hist, d_hist, epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(8,5))
    xs = np.arange(1, len(g_hist)+1)
    plt.plot(xs, g_hist, label="G loss", linewidth=1.5)
    plt.plot(xs, d_hist, label="D loss", linewidth=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    out_png = os.path.join(out_dir, f"losses_e{epoch}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_samples_plot(G, device, out_dir, epoch, test_z, test_labels,
                      label_names=("Pathological","Normal")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    G.eval()
    with torch.no_grad():
        z_vis = test_z.to(device)
        y_vis = test_labels.to(device)
        x_gen = G(z_vis, y_vis).detach().cpu().numpy()  # (2,1,1000), in [-1,1]
    G.train()

    # Denormalize to BPM [50,200]: ((x+1)/2)*(200-50)+50
    gen_bpm = ((x_gen + 1.0) * (200.0 - 50.0) / 2.0) + 50.0  # shape (2,1,1000)

    plt.figure(figsize=(15,8))

    # Sample 0
    plt.subplot(2,2,1)
    plt.plot(gen_bpm[0,0,:1000], linewidth=1.5, alpha=0.9)
    plt.title(f'Generated {label_names[0]} Sample (Epoch {epoch})')
    plt.xlabel('Time Steps'); plt.ylabel('BPM'); plt.grid(True, alpha=0.3); plt.ylim(50,200)

    # Sample 1
    plt.subplot(2,2,2)
    plt.plot(gen_bpm[1,0,:1000], linewidth=1.5, alpha=0.9)
    plt.title(f'Generated {label_names[1]} Sample (Epoch {epoch})')
    plt.xlabel('Time Steps'); plt.ylabel('BPM'); plt.grid(True, alpha=0.3); plt.ylim(50,200)

    # Comparison
    plt.subplot(2,2,3)
    plt.plot(gen_bpm[0,0,:1000], linewidth=1.2, alpha=0.95, label=label_names[0])
    plt.plot(gen_bpm[1,0,:1000], linewidth=1.2, alpha=0.95, label=label_names[1])
    plt.title(f'Comparison (1000 steps) — Epoch {epoch}')
    plt.xlabel('Time Steps'); plt.ylabel('BPM'); plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(50,200)

    # Info panel
    plt.subplot(2,2,4)
    info_text = (
        f"Epoch: {epoch}\n"
        f"Fixed test labels: {test_labels.tolist()}\n"
        f"Min/Max {label_names[0]}: {gen_bpm[0,0].min():.1f}/{gen_bpm[0,0].max():.1f} BPM\n"
        f"Min/Max {label_names[1]}: {gen_bpm[1,0].min():.1f}/{gen_bpm[1,0].max():.1f} BPM\n"
        f"(Saved to samples_e{epoch}.png)"
    )
    plt.axis('off')
    plt.text(0.02, 0.98, info_text, va='top', ha='left', fontsize=10)

    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"samples_e{epoch}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

def train(
    data_path,
    ph_csv,
    out_dir="./checkpoints_CTGGAN_october",
    epochs=500,
    batch_size=32,
    z_len=100,
    n_critic=5,
    lr=1e-4,
    beta1=0.0,
    beta2=0.9,
    gp_lambda=10.0,
    featmatch_alpha=0.0,
    seed=81,
    device_str=None,
    train_normal=70, train_pathological=70,
    eval_normal=30, eval_pathological=30,
    label_names=("Pathological","Normal")   # order must match your label ids [0,1]
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device(device_str or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "epochs"), exist_ok=True)

    # your preprocessing pipeline
    ds = myDataset(
        data_path, ph_csv, sequence_length=4000,
        max_normal=447, max_pathological=105,
        train_normal=train_normal, train_pathological=train_pathological,
        eval_normal=eval_normal,   eval_pathological=eval_pathological,
        mode='train', random_seed=seed
    )

    ds2 = WindowedFromLongDataset(ds, win_len=1000)
    dl = DataLoader(ds2, batch_size=batch_size, shuffle=True, drop_last=True)

    # optional: record splits
    if hasattr(ds, "get_eval_files_info"):
        with open(os.path.join(out_dir, "eval_files_info.json"), "w") as f:
            json.dump(ds.get_eval_files_info(), f, indent=2)
    if hasattr(ds, "get_train_files_info"):
        with open(os.path.join(out_dir, "train_files_info.json"), "w") as f:
            json.dump(ds.get_train_files_info(), f, indent=2)

    # Fixed test noise & labels for periodic visualization
    test_labels = torch.tensor([0, 1], dtype=torch.long)        # assumes 0=Pathological, 1=Normal
    test_z      = torch.randn(test_labels.size(0), 1, z_len)    # (2,1,100)

    G = Generator().to(device)
    D = Discriminator().to(device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    best_g, best_d = float("inf"), float("inf")
    g_hist, d_hist = [], []
    print(f"Start CTGGAN training on {device} for {epochs} epochs, {len(ds)} samples...")

    for epoch in range(1, epochs+1):
        epoch_g, epoch_d = [], []
        for it, (x_real, y) in enumerate(dl):
            x_real, y = x_real.to(device), y.to(device).long()

            # ----- train D: n_critic steps -----
            for _ in range(n_critic):
                z = torch.randn(x_real.size(0), 1, z_len, device=device)
                with torch.no_grad():
                    x_fake = G(z, y)
                D_real, real_feats = D(x_real, y)
                D_fake, fake_feats = D(x_fake, y)

                d_loss = -(D_real.mean() - D_fake.mean())
                gp = gradient_penalty(D, x_real, x_fake, y, device, gp_lambda=gp_lambda)
                (d_loss + gp).backward()
                optD.step()
                optD.zero_grad(set_to_none=True)

            # ----- train G -----
            z = torch.randn(x_real.size(0), 1, z_len, device=device)
            x_fake = G(z, y)
            D_fake, fake_feats = D(x_fake, y)
            _, real_feats = D(x_real.detach(), y)
            g_loss = -D_fake.mean()
            if featmatch_alpha > 0:
                g_loss = g_loss + ((real_feats.detach() - fake_feats) ** 2).mean() * featmatch_alpha

            g_loss.backward()
            optG.step()
            optG.zero_grad(set_to_none=True)

            epoch_d.append((d_loss + gp).item())
            epoch_g.append(g_loss.item())

        avg_d = float(np.mean(epoch_d))
        avg_g = float(np.mean(epoch_g)) if epoch_g else 0.0
        d_hist.append(avg_d); g_hist.append(avg_g)

        # save “best” (lower is better surrogate here)
        if avg_d < best_d:
            best_d = avg_d
            torch.save(D.state_dict(), os.path.join(out_dir, "best_D.pth"))
        if avg_g < best_g:
            best_g = avg_g
            torch.save(G.state_dict(), os.path.join(out_dir, "best_G.pth"))

        # periodic snapshots
        if epoch % 5 == 0 or epoch == epochs:
            torch.save(G.state_dict(), os.path.join(out_dir, "epochs", f"G_e{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(out_dir, "epochs", f"D_e{epoch}.pth"))

        print(f"[{epoch:03d}/{epochs}] D={avg_d:.4f}  G={avg_g:.4f}  (best D={best_d:.4f}, best G={best_g:.4f})")

        # ====== plots every 50 epochs (and at 1 & final) ======
        if (epoch % 50 == 0) or (epoch == 1) or (epoch == epochs):
            save_loss_plot(out_dir, g_hist, d_hist, epoch)
            save_samples_plot(G, device, out_dir, epoch, test_z, test_labels, label_names=label_names)

    np.save(os.path.join(out_dir, "g_losses.npy"), np.array(g_hist, dtype=np.float32))
    np.save(os.path.join(out_dir, "d_losses.npy"), np.array(d_hist, dtype=np.float32))
    torch.save(G.state_dict(), os.path.join(out_dir, "final_G.pth"))
    torch.save(D.state_dict(), os.path.join(out_dir, "final_D.pth"))
    print("Training finished.")

if __name__ == "__main__":
    # EDIT THESE TWO PATHS to match your machine
    data_path = "/path/to/.csv"
    ph_csv    = "/path/to/ph_labels.csv"

    train(
        data_path=data_path,
        ph_csv=ph_csv,
        out_dir="./checkpoints_CTGGAN_october2170n70p4000WINDOW",
        epochs=500,
        batch_size=32,
        n_critic=5,
        lr=1e-4, beta1=0.0, beta2=0.9,
        gp_lambda=10.0,
        featmatch_alpha=0.0,
        seed=81
    )

