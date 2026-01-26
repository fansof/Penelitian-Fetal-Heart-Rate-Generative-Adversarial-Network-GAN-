import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W = x.size()
        q = self.query(x).view(B, -1, W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, W)
        v = self.value(x).view(B, -1, W)
        attn = torch.bmm(q, k)
        attn = torch.nn.functional.softmax(attn, dim=2)
        o = torch.bmm(v, attn.permute(0, 2, 1))
        o = self.gamma * o.view(B, C, W) + x
        return o


class ResidualBlock(nn.Module):
    """Residual block for Generator"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return nn.ReLU()(self.block(x) + self.residual(x))


class ResidualBlockDisc(nn.Module):
    """Residual block for Discriminator"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockDisc, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        return nn.LeakyReLU(0.2, True)(self.block(x) + self.residual(x))


class Generator1000(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, num_classes=2):
        super(Generator1000, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Input: (B, 1+2, 100) -> (B, 256, 125)
        self.dc1 = nn.Sequential(
            nn.Conv1d(1 + num_classes, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(256)
        )
        
        # (B, 256, 125) -> (B, 128, 250)  
        self.rb2_up = nn.Upsample(size=250, mode='linear', align_corners=False)
        self.rb2_conv1 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(128)
        )
        self.rb2_conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(128)
        )
        
        # (B, 128, 250)
        self.sa3 = SelfAttention(128)
        
        # (B, 128, 250) -> (B, 64, 500)
        self.rb4_up = nn.Upsample(size=500, mode='linear', align_corners=False)
        self.rb4_conv = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(64)
        )
        
        # (B, 64, 500) -> (B, 32, 1000)
        self.rb5_up = nn.Upsample(size=1000, mode='linear', align_corners=False)
        self.rb5_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(32)
        )
        
        # (B, 32, 1000) -> (B, 1, 1000)
        self.dc6 = nn.Sequential(
            nn.Conv1d(32, output_dim, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (B, input_dim) -> reshape to (B, 1, 100)
        B = z.size(0)
        z = z.view(B, 1, -1)  # (B, 1, 100)
        
        # Create one-hot labels and expand to match z's temporal dimension
        labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(device)
        labels_onehot = labels_onehot.unsqueeze(2).expand(-1, -1, z.size(2))  # (B, 2, 100)
        
        x = torch.cat([z, labels_onehot], dim=1)  # (B, 3, 100)
        
        # First upsample 100 -> 125
        x = F.interpolate(x, size=125, mode='linear', align_corners=False)
        
        x = self.dc1(x)           # (B, 256, 125)
        x = self.rb2_up(x)        # (B, 256, 250)
        x = self.rb2_conv1(x)     # (B, 128, 250)
        x = self.rb2_conv2(x)     # (B, 128, 250)
        x = self.sa3(x)           # (B, 128, 250)
        x = self.rb4_up(x)        # (B, 128, 500)
        x = self.rb4_conv(x)      # (B, 64, 500)
        x = self.rb5_up(x)        # (B, 64, 1000)
        x = self.rb5_conv(x)      # (B, 32, 1000)
        x = self.dc6(x)           # (B, 1, 1000)
        
        return x


class Discriminator1000(nn.Module):
    def __init__(self, input_dim=1, num_classes=2):
        super(Discriminator1000, self).__init__()
        self.num_classes = num_classes
        
        # (B, 3, 1000) -> (B, 32, 1000)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim + num_classes, 32, kernel_size=25, stride=1, padding=12),
            nn.LeakyReLU(0.2, True)
        )
        
        # (B, 32, 1000) -> (B, 64, 500)
        self.rb2 = ResidualBlockDisc(32, 64, stride=2)
        
        # (B, 64, 500)
        self.sa3 = SelfAttention(64)
        
        # (B, 64, 500) -> (B, 128, 250)
        self.rb4 = ResidualBlockDisc(64, 128, stride=2)
        
        # (B, 128, 250) -> (B, 256, 125)
        self.rb5 = ResidualBlockDisc(128, 256, stride=2)
        
        # (B, 256, 125) -> (B, 1, 1)
        self.fc6 = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=20, stride=1, padding=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = torch.nn.functional.one_hot(labels, self.num_classes).float().view(-1, 2, 1).expand(-1, 2, x.size(2)).to(device)
        x = torch.cat([x, labels], 1)
        x = self.conv1(x)
        x = self.rb2(x)
        x = self.sa3(x)
        x = self.rb4(x)
        features = self.rb5(x)  # Save features before final layer
        score = self.fc6(features)
        
        return score, features  # Return BOTH score and features
