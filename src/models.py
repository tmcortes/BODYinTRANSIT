import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_proj = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, N, D]
        attn_logits = self.attn_proj(x).squeeze(-1)   # [B, N]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, N]
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled, attn_weights



def weight_init(module, mode="fan_out", nonlinearity="relu"):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Downsample(nn.Module):
    def __init__(self, channels=None, factor=2, order=1):
        super().__init__()
        assert factor > 1
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0
        padding = total_padding // 2
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, np.ones(factor))
        kernel = torch.from_numpy(kernel / kernel.sum()).float()
        self.register_buffer("kernel", kernel[None, None, :].repeat((channels, 1, 1)))
        self.stride = factor
        self.padding = padding

    def forward(self, x):
        return F.conv1d(x, self.kernel, stride=self.stride,
                        padding=self.padding, groups=x.shape[1])

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                               padding, bias=False, padding_mode="circular")
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride,
                               padding, bias=False, padding_mode="circular")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)
        return x + identity

class Resnet(nn.Module):
    def __init__(self,
                 output_size: int = 2,
                 n_channels: int = 3,
                 is_eva: bool = False,
                 resnet_version: int = 1,
                 epoch_len: int = 5):
        super().__init__()

        # build architecture config
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [(64,5,2,5,2,2),(128,5,2,5,2,2),
                       (256,5,2,5,3,1),(256,5,2,5,3,1),(512,5,0,5,3,1)]
            elif epoch_len == 10:
                cgf = [(64,5,2,5,2,2),(128,5,2,5,2,2),
                       (256,5,2,5,5,1),(512,5,2,5,5,1),(1024,5,0,5,3,1)]
            else:
                cgf = [(64,5,2,5,3,1),(128,5,2,5,3,1),
                       (256,5,2,5,5,1),(512,5,2,5,5,1),(1024,5,0,5,4,0)]
        else:
            cgf = [(64,5,2,5,3,1),(64,5,2,5,3,1),
                   (128,5,2,5,5,1),(128,5,2,5,5,1),
                   (256,5,2,5,4,0)]

        in_ch = n_channels
        layers = []
        for idx, (out_ch, ksz, nblk, rksz, downfac, downord) in enumerate(cgf):
            layers.append(self.make_layer(in_ch, out_ch, ksz, nblk, rksz, downfac, downord))
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)
        self.is_eva = is_eva
        if is_eva:
            self.classifier = nn.Sequential(
                nn.Linear(in_ch, 512),
                nn.ReLU(True),
                nn.Linear(512, output_size)
            )
        else:
            self.classifier = nn.Linear(in_ch, output_size)

        weight_init(self)

    @staticmethod
    def make_layer(in_ch, out_ch, ksz, nblk, rksz, downfac, downord):
        modules = [nn.Conv1d(in_ch, out_ch, ksz, 1, (ksz-1)//2,
                             bias=False, padding_mode="circular")]
        for _ in range(nblk):
            modules.append(ResBlock(out_ch, out_ch, rksz, 1, (rksz-1)//2))
        modules += [nn.BatchNorm1d(out_ch), nn.ReLU(True), Downsample(out_ch, downfac, downord)]
        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(x.size(0), -1)
        return self.classifier(feats)

def harnet5(pretrained: bool = False, weight_path: str = None,
            output_size: int =2, epoch_len: int =5, device="cpu"):
    model = Resnet(output_size=output_size, is_eva=True, epoch_len=epoch_len)
    if pretrained:
        assert weight_path is not None, "Must provide weight_path if pretrained=True"
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    return model


# Out model RoSA: Robust Sensor Aggregation Architecture
class RoSA(nn.Module):
    def __init__(self, 
                 num_accels=17, 
                 embed_dim=512, 
                 transformer_depth=2, 
                 n_heads=4,
                 num_classes=3,
                 freeze_harnet=True):
        
        super().__init__()
        self.num_accels = num_accels
        self.embed_dim = embed_dim
        self.freeze_harnet_flag = freeze_harnet

        # Load pretrained HarNet5
        harnet5_full = harnet5(pretrained=True,
                weight_path="weights/harnet5_weights.pth",
                output_size=2,
                epoch_len=5,
                device="cpu")

        self.harnet = harnet5_full.feature_extractor

        # Freeze HarNet if needed
        if freeze_harnet:
            self.freeze_feature_extractor()

        # # Use standard sinusoidal positional encoding
        # self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=num_accels)

        # # Transformer encoder
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

        # Attention pooling
        self.attn_pool = AttentionPooling(embed_dim)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def freeze_feature_extractor(self):
        for param in self.harnet.parameters():
            param.requires_grad = False

    def forward(self, x, dum=None):
        B, T, C = x.shape
        assert C == self.num_accels * 3, f"Expected {self.num_accels*3} channels, got {C}"

        # Ensure eval mode for frozen HarNet (important for BatchNorm etc.)
        if self.freeze_harnet_flag:
            self.harnet.eval()

        # Reshape input to [B*17, 3, T] for HarNet
        x = x.view(B, T, self.num_accels, 3)        # [B, T, 17, 3]
        x = x.permute(0, 2, 3, 1)                   # [B, 17, 3, T]
        x = x.reshape(B * self.num_accels, 3, T)    # [B*17, 3, T]

        # Feature extraction
        with torch.no_grad() if self.freeze_harnet_flag else torch.enable_grad():
            features = self.harnet(x)               # [B*17, 512]

        features = features.view(B, self.num_accels, self.embed_dim)  # [B, 17, 512]

        # Add positional encoding
        #features = self.pos_encoder(features.transpose(0, 1))         # [17, B, 512]
        #features = features.transpose(0, 1)                           # [B, 17, 512]

        # Transformer
        #encoded = self.transformer(features)        # [B, 17, 512]

        # Attention pooling
        pooled, attn_weights = self.attn_pool(features)  # [B, 512], [B, 17]

        # Classifier
        logits = self.classifier(pooled)            # [B, num_classes]
        return logits, attn_weights



if __name__ == "__main__":

    model = RoSA(freeze_harnet=True)
    x = torch.randn(8, 150, 51)
    logits, attn_weights = model(x)
