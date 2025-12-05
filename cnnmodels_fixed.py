import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=25, stride=1, padding=24, dilation=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 1, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet1D(nn.Module):

    def __init__(self, num_classes=2, layers=[2, 2, 2, 2]):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(BasicBlock1D, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1D, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Bottleneck1D(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet50V1_1D(nn.Module):

    def __init__(self, num_classes=2):
        super(ResNet50V1_1D, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet50 layers: [3, 4, 6, 3]
        self.layer1 = self._make_layer(Bottleneck1D, 64, 3)
        self.layer2 = self._make_layer(Bottleneck1D, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck1D, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck1D, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * Bottleneck1D.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for MobileNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNet1D(nn.Module):
    """
    1D MobileNet adapted for time series classification
    """
    def __init__(self, num_classes=2):
        super(MobileNet1D, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        # Depthwise separable convolutions
        self.features = nn.Sequential(
            DepthwiseSeparableConv1d(32, 64, stride=1),
            DepthwiseSeparableConv1d(64, 128, stride=2),
            DepthwiseSeparableConv1d(128, 128, stride=1),
            DepthwiseSeparableConv1d(128, 256, stride=2),
            DepthwiseSeparableConv1d(256, 256, stride=1),
            DepthwiseSeparableConv1d(256, 512, stride=2),
            DepthwiseSeparableConv1d(512, 512, stride=1),
            DepthwiseSeparableConv1d(512, 512, stride=1),
            DepthwiseSeparableConv1d(512, 512, stride=1),
            DepthwiseSeparableConv1d(512, 512, stride=1),
            DepthwiseSeparableConv1d(512, 512, stride=1),
            DepthwiseSeparableConv1d(512, 1024, stride=2),
            DepthwiseSeparableConv1d(1024, 1024, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D"""
    def __init__(self, channels, middle_channels):  # Accept middle_channels
        super().__init__()
        self.fc1 = nn.Linear(channels, middle_channels)
        self.fc2 = nn.Linear(middle_channels, channels)

    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1)
        return x * y


class MBConv1D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for EfficientNet"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 expand_ratio=6, se_ratio=0.25):
        super(MBConv1D, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        # Expansion
        hidden_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, 
                                     bias=False) if expand_ratio != 1 else nn.Identity()
        self.expand_bn = nn.BatchNorm1d(hidden_channels) if expand_ratio != 1 else nn.Identity()

        # Depthwise
        self.dw_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                                stride=stride, padding=kernel_size//2, 
                                groups=hidden_channels, bias=False)
        self.dw_bn = nn.BatchNorm1d(hidden_channels)

        # SE block
        # self.se = SEBlock1D(hidden_channels, int(in_channels * se_ratio)) if se_ratio > 0 else nn.Identity()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SEBlock1D(hidden_channels, se_channels)

        # Projection
        self.project_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x

        # Expansion
        if not isinstance(self.expand_conv, nn.Identity):
            x = F.silu(self.expand_bn(self.expand_conv(x)))

        # Depthwise
        x = F.silu(self.dw_bn(self.dw_conv(x)))

        # SE
        x = self.se(x)

        # Projection
        x = self.project_bn(self.project_conv(x))

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x


class EfficientNetB0_1D(nn.Module):
    """
    1D EfficientNetB0 adapted for time series classification
    """
    def __init__(self, num_classes=2):
        super(EfficientNetB0_1D, self).__init__()

        # Initial convolution
        self.conv_stem = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm1d(32)

        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConv1D(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv1D(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv1D(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv1D(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv1D(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv1D(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv1D(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv1D(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv1D(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv1D(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )

        # Head
        self.conv_head = nn.Conv1d(320, 1280, kernel_size=1, bias=False)
        self.bn_head = nn.BatchNorm1d(1280)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = F.silu(self.bn_stem(self.conv_stem(x)))
        x = self.blocks(x)
        x = F.silu(self.bn_head(self.conv_head(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class DenseBlock1D(nn.Module):
    """Dense block for DenseNet"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0):
        super(DenseBlock1D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_dense_layer(in_channels + i * growth_rate, 
                                          growth_rate, bn_size, drop_rate)
            self.layers.append(layer)

    def _make_dense_layer(self, in_channels, growth_rate, bn_size, drop_rate):
        layers = []
        layers.append(nn.BatchNorm1d(in_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm1d(bn_size * growth_rate))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(bn_size * growth_rate, growth_rate, 
                               kernel_size=3, padding=1, bias=False))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class Transition1D(nn.Module):
    """Transition layer for DenseNet"""
    def __init__(self, in_channels, out_channels):
        super(Transition1D, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layers(x)


class DenseNet201_1D(nn.Module):
    """
    1D DenseNet201 adapted for time series classification
    """
    def __init__(self, num_classes=2, growth_rate=32, block_config=(6, 12, 48, 32)):
        super(DenseNet201_1D, self).__init__()

        # Initial convolution
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv1d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock1D(num_layers, num_features, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition1D(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def create_model(model_name, num_classes=2):
    """
    Factory function to create models
    """
    model_name = model_name.lower()

    if model_name == 'cnn1d':
        return CNN1D(num_classes)
    elif model_name == 'resnet1d':
        return ResNet1D(num_classes)
    elif model_name == 'resnet50v1':
        return ResNet50V1_1D(num_classes)
    elif model_name == 'mobilenet':
        return MobileNet1D(num_classes)
    elif model_name == 'efficientnetb0':
        return EfficientNetB0_1D(num_classes)
    elif model_name == 'densenet201':
        return DenseNet201_1D(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Test function to verify models work
def test_models():
    """Test all models with sample data"""
    batch_size = 4
    sequence_length = 1000
    sample_input = torch.randn(batch_size, 1, sequence_length)

    models = ['cnn1d', 'resnet1d', 'resnet50v1', 'mobilenet', 'efficientnetb0', 'densenet201']

    for model_name in models:
        print(f"Testing {model_name}...")
        try:
            model = create_model(model_name, num_classes=2)
            output = model(sample_input)
            print(f"  ✓ {model_name}: Input {sample_input.shape} -> Output {output.shape}")
        except Exception as e:
            print(f"  ✗ {model_name}: Error - {e}")

    # Test with different sequence lengths
    print(f"\nTesting variable sequence lengths...")
    for seq_len in [500, 1000, 2000]:
        sample_input = torch.randn(2, 1, seq_len)
        try:
            model = create_model('cnn1d', num_classes=2)
            output = model(sample_input)
            print(f"  ✓ CNN1D with seq_len={seq_len}: Input {sample_input.shape} -> Output {output.shape}")
        except Exception as e:
            print(f"  ✗ CNN1D with seq_len={seq_len}: Error - {e}")


if __name__ == "__main__":
    test_models()
