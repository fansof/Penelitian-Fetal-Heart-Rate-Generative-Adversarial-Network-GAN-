import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    1D adaptation of CCWGAN-GP Generator as described in the paper:
    - Concatenate noise z and one-hot label c
    - Fully-connected map to (channels=2048, length=T)
    - Six convolutional blocks with BN(momentum=0.9) + LeakyReLU(0.2)
    - Final Conv1d to 1 channel with tanh

    Note: Paper lists 4x4 kernels/stride 2 (2D view). In 1D, we keep
    time length T constant (no pooling) and reduce channels stepwise
    2048→512→256→128→64→32→1 using Conv1d(kernel_size=3, stride=1, padding='same').
    """

    def __init__(self, latent_dim=100, num_classes=2, sequence_length=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.fc = nn.Linear(latent_dim + num_classes, sequence_length * 2048)
        self.bn_fc = nn.BatchNorm1d(2048, momentum=0.9)

        self.conv1 = nn.Conv1d(2048, 512, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(512, momentum=0.9)

        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(256, momentum=0.9)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128, momentum=0.9)

        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(64, momentum=0.9)

        self.conv5 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding='same')
        self.bn5 = nn.BatchNorm1d(32, momentum=0.9)

        self.conv6 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding='same')

        self.act = nn.LeakyReLU(0.2)

    def forward(self, noise, labels_onehot):
        b = noise.size(0)
        x = torch.cat([noise, labels_onehot], dim=1)
        x = self.fc(x)
        x = x.view(b, 2048, self.sequence_length)
        x = self.act(self.bn_fc(x))

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))

        x = torch.tanh(self.conv6(x))

        # if x.shape[2] != self.sequence_length:
        #     x = x[:, :, : self.sequence_length]
        return x


class Discriminator(nn.Module):
    """
    1D adaptation of the CCWGAN-GP Discriminator + Auxiliary Classifier:
    - Five Conv1d layers; BN on first four; LeakyReLU activations
    - Flatten → FC(64) → Dropout(0.4) → two heads
      * validity head: raw score (WGAN-GP critic)
      * classifier head: logits for CrossEntropy

    Note: No sigmoid on validity for WGAN-GP.
    """

    def __init__(self, num_classes=2, sequence_length=1000):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(32, momentum=0.9)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64, momentum=0.9)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(128, momentum=0.9)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(256, momentum=0.9)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding='same')

        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.4)

        # compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, sequence_length)
            h = self._forward_conv(dummy)
            flat = h.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat, 64)
        self.validity_head = nn.Linear(64, 1)
        self.classifier_head = nn.Linear(64, num_classes)
        # Probabilistic outputs per paper description
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def _forward_conv(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.conv5(x))
        return x

    def forward(self, x, return_logits: bool = False):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.dropout(self.fc1(x)))
        v_logit = self.validity_head(x)
        c_logit = self.classifier_head(x)
        if return_logits:
            return v_logit, c_logit
        # default: probabilities for inspection
        # validity = v_logit
        # class_probs = c_logit
        validity = self.sigmoid(v_logit)
        class_probs = self.softmax(c_logit)
        return validity, class_probs
