import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from portable_udh import get_hiding_unet, get_reveal_net, count_params
from config import device
# from options import HiDDenConfiguration
# from model.conv_bn_relu import ConvBNRelu
import torch.nn.functional as F
import math
from os.path import join


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out, stride=1):

        super(ConvBNRelu, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class HiDDenConfiguration():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, H: int, W: int, message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 use_vgg: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float,
                 cover_dependent: int,
                 residual: int = 0, 
                 enable_fp16: bool = False):
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.cover_dependent = cover_dependent
        self.residual = residual
        self.enable_fp16 = enable_fp16

class TestEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        config = HiDDenConfiguration(
            H=32, W=32, 
            message_length=100, 
            encoder_blocks=5, encoder_channels=64,
            decoder_blocks=5, decoder_channels=64,
            use_discriminator=False,
            use_vgg=False,
            discriminator_blocks=4, discriminator_channels=64,
            decoder_loss=1.0,
            encoder_loss=1.0,
            adversarial_loss=0.001,
            cover_dependent=1,
            residual=0,
            enable_fp16=False
        )
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks
        self.cover_dependent = config.cover_dependent
        # self.residual = config.residual

        if self.cover_dependent == 1:
            layers = [ConvBNRelu(3, self.conv_channels)]
        else:
            layers = [ConvBNRelu(config.message_length, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        if self.cover_dependent == 1:
            self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)
        else:
            self.after_concat_layer = ConvBNRelu(config.message_length,
                                             self.conv_channels)            

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        self.final_tanh = nn.Tanh()
        self.factor = 10/255

    def forward(self, image, message):

        # # First, add two dummy dimensions in the end of the message.
        # # This is required for the .expand to work correctly
        # expanded_message = message.unsqueeze(-1)
        # expanded_message.unsqueeze_(-1)

        # expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        # encoded_image = self.conv_layers(image)
        # # concatenate expanded message and image
        # concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        # im_w = self.after_concat_layer(concat)
        # im_w = self.final_layer(im_w)

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        if self.cover_dependent:
            encoded_image = self.conv_layers(image)
            # concatenate expanded message and image
            concat = torch.cat([expanded_message, encoded_image, image], dim=1)
            im_w = self.after_concat_layer(concat)
            im_w = self.final_layer(im_w)
            # if self.residual:
            #     im_w = self.factor * self.final_tanh(im_w) + image
        else:
            #import pdb; pdb.set_trace()
            #encoded_message = self.conv_layers(expanded_message)
            # concatenate expanded message and image
            #concat = encoded_message # torch.cat([expanded_message, encoded_image, image], dim=1)
            im_w = self.after_concat_layer(expanded_message)
            im_w = self.final_layer(im_w) + image
        return im_w


def load_models(encoder, decoder, ckpt_dir):
    print(f"load_models from {ckpt_dir}")
    encoder.load_state_dict(torch.load(join(ckpt_dir, "encoder.ckpt")))
    decoder.load_state_dict(torch.load(join(ckpt_dir, "decoder.ckpt")))
    logger.info("loader models")

def get_models(hparams):
    if hparams.model_type == 'normal':
        dec_um_conv_dim = 1 + 64
        encoder = FullEncoder(
            block_type=hparams.block_type,
            enc_n_layers=hparams.enc_n_layers,
            dec_um_conv_dim=dec_um_conv_dim,
            dec_c_n_layers=hparams.dec_c_n_layers
        ).to(device)
        decoder = MsgDecoder(conv_dim=1, block_type=hparams.block_type).to(device)

    elif hparams.model_type == 'unet':
        encoder = get_hiding_unet(input_nc=1+1, output_nc=1, num_downs=5, norm='instance', use_dropout=False).to(device)
        decoder = get_reveal_net(input_nc=1, output_nc=1, norm='instance').to(device)

    elif hparams.model_type == 'transformer':
        # --- 定义 En_wm: 水印编码器 ---
        class WatermarkEncoder(nn.Module):
            def __init__(self, watermark_size, feat_dim=512):
                super().__init__()
                self.feat_dim = feat_dim
                self.watermark_size = watermark_size
                # maps either a vector of length `watermark_size` or a pooled spectrogram
                self.net = nn.Sequential(
                    nn.Linear(watermark_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, feat_dim)
                )

            def forward(self, msg):
                """Accepts either:
                - msg: [B, watermark_size] (vector), or
                - msg: [B, C, H, W] (spectrogram). In the latter case we adaptively
                  pool across spatial dims to a vector of length `watermark_size`.
                Returns: [B, feat_dim]
                """
                # If message is a spectrogram, pool to (1, watermark_size) then flatten
                if msg.dim() == 4:  # [B, C, H, W]
                    B = msg.shape[0]
                    # pool to (1, watermark_size) -> [B, C, 1, watermark_size]
                    pooled = F.adaptive_avg_pool2d(msg, (1, self.watermark_size))
                    vec = pooled.view(B, -1)  # [B, C * watermark_size] (C typically 1)
                    # if C>1, ensure length matches watermark_size by projecting if needed
                    if vec.shape[1] != self.watermark_size:
                        # project or trim/pad to required size
                        vec = vec[:, :self.watermark_size]
                    return self.net(vec)

                # Otherwise assume it's already a vector [B, watermark_size]
                return self.net(msg)  # [B, feat_dim]

        # --- 定义 En_ac: 音频编码器 (提取 STFT 特征) ---
        class AudioEncoder(nn.Module):
            def __init__(self, in_channels=1, out_channels=64):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
                self.transformers = nn.Sequential(
                    *[MultiScaleTransformerBlock(dim=32) for _ in range(3)]
                )
                self.conv2 = nn.Conv2d(32, out_channels, kernel_size=1)
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.transformers(x)
                x = self.conv2(x)
                return x  # [B, 64, H, W]

        # --- 定义 GeneratorG: 生成扰动 S ---
        # PerturbationGenerator was moved to module scope (class `PerturbationGenerator`) to allow
        # importing directly via `from model import PerturbationGenerator`.
        # Use: self.GeneratorG = PerturbationGenerator().to(device)
        pass

        # --- 组装 encoder ---
        class TransformerEncoderWrapper(nn.Module):
            def __init__(self, watermark_size):
                super().__init__()
                self.En_wm = WatermarkEncoder(watermark_size).to(device)
                self.En_ac = AudioEncoder().to(device)
                self.GeneratorG = PerturbationGenerator().to(device)

            def forward(self, carrier, msg):
                # This forward is not used in solver.py; solver calls submodules directly
                ac_feat = self.En_ac(carrier)
                wm_feat = self.En_wm(msg)
                S = self.GeneratorG(ac_feat, wm_feat)
                watermarked = carrier + S
                return watermarked, S

        encoder = TransformerEncoderWrapper(hparams.watermark_size).to(device)
        decoder = WatermarkDecoder(watermark_size=hparams.watermark_size, input_channels=1, hidden_dim=512).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {hparams.model_type}")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=hparams.lr, betas=(0.5, 0.999))
    scheduler =StepLR(optimizer, step_size=20, gamma=0.5)
    # if hparams.load_ckpt:
    #     load_models(encoder, decoder, hparams.load_ckpt)

    # logger.debug(encoder)
    # logger.debug(decoder)

    return encoder, decoder, optimizer, scheduler


class FullEncoder(nn.Module):
    def __init__(self, block_type, enc_n_layers, dec_um_conv_dim, dec_c_n_layers) -> None:
        super().__init__()
        self.encoder_first = Encoder(block_type=block_type,
                                n_layers=enc_n_layers)

        self.encoder_second = CarrierDecoder(conv_dim=dec_um_conv_dim,
                                        block_type=block_type,
                                        n_layers=dec_c_n_layers)
        
    def forward(self, carrier, msg):
        # msg_emb = self.encoder_first(msg)
        # msg_u = self.encoder_second(msg_emb)
        # carrier_reconst = carrier + msg_u
        
        msg_emb = self.encoder_first(msg)
        msg_merged = torch.cat((msg, msg_emb), dim=1)
        msg_u = self.encoder_second(msg_merged)
        carrier_reconst = carrier + msg_u
        

        # print(f"msg_u.shape={msg_u.shape},carrier.shape={carrier.shape}")
        # print(f"msg.shape={msg.shape},msg_emb.shape={msg_emb.shape}")
        # print(f"carrier_reconst.shape={carrier_reconst.shape}")
        return carrier_reconst
    


class GatedBlockBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(GatedBlockBN, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        conv = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.BatchNorm2d(c_out)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = torch.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class GatedBlockIN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(GatedBlockIN, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_conv = nn.InstanceNorm2d(c_out)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.InstanceNorm2d(c_out)

    def forward(self, x):
        x1 = self.bn_conv(self.conv(x))
        x2 = torch.sigmoid(self.bn_gate(self.gate(x)))
        out = x1 * x2
        return out

class GatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(GatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        return out

class SkipGatedBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(SkipGatedBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        self.conv = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.gate = conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.skip = True if c_in == c_out else False

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.sigmoid(self.gate(x))
        out = x1 * x2
        if self.skip: 
            out += x
        return out

class ReluBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, deconv=False, conv_dim=2):
        super(ReluBlock, self).__init__()
        conv = {(False, 1): nn.Conv1d,
                (True, 1): nn.ConvTranspose1d,
                (False, 2): nn.Conv2d,
                (True, 2): nn.ConvTranspose2d}[(deconv, conv_dim)]
        bn = {1: nn.BatchNorm1d,
              2: nn.BatchNorm2d}[conv_dim]
        self.conv = nn.Sequential(
            conv(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            bn(c_out),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class PrintShapeLayer(nn.Module):
    def __init__(self, str=None):
        super(PrintShapeLayer, self).__init__()
        self.str = str

    def forward(self, input):
        if self.str: 
            logger.debug(f"{self.str}")
        logger.debug(f"{input.shape}")
        return input

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Encoder(nn.Module):
    def __init__(self, conv_dim=1, block_type='normal', n_layers=3):
        super(Encoder, self).__init__()
        block = {'normal': GatedBlock,
                 'skip': SkipGatedBlock,
                 'bn': GatedBlockBN,
                 'in': GatedBlockIN,
                 'relu': ReluBlock}[block_type]

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers-1):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class CarrierDecoder(nn.Module):
    def __init__(self, conv_dim, block_type='normal', n_layers=4):
        super(CarrierDecoder, self).__init__()
        block = {'normal': GatedBlock,
                 'skip': SkipGatedBlock,
                 'bn': GatedBlockBN,
                 'in': GatedBlockIN,
                 'relu': ReluBlock}[block_type]

        layers = [block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False)]

        for i in range(n_layers-2):
            layers.append(block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False))

        layers.append(block(c_in=64, c_out=1, kernel_size=1, stride=1, padding=0, deconv=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        h = self.main(x)
        return h

class MsgDecoder(nn.Module):
    def __init__(self, conv_dim=1, block_type='normal'):
        super(MsgDecoder, self).__init__()
        block = {'normal': GatedBlock,
                 'skip': SkipGatedBlock,
                 'bn': GatedBlockBN,
                 'in': GatedBlockIN,
                 'relu': ReluBlock}[block_type]

        self.main = nn.Sequential(
                block(c_in=conv_dim, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=64, kernel_size=3, stride=1, padding=1, deconv=False),
                block(c_in=64, c_out=1, kernel_size=3, stride=1, padding=1, deconv=False)
                )

    def forward(self, x):
        h = self.main(x)
        return h

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                GatedBlockBN(1,16,3,1,1),
                GatedBlockBN(16,32,3,1,1),
                GatedBlockBN(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x


def compute_BER_Tensor(original: torch.Tensor, recovered: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    计算比特错误率 (BER)，适用于浮点概率输出。
    假设输入是二维张量 (batch_size, feature_dim)。
    """
    # Thresholding to convert probabilities to binary (0 or 1)
    original_binary = (original > threshold).float()
    recovered_binary = (recovered > threshold).float()

    # Calculate differing bits
    error_bits = (original_binary != recovered_binary).sum(dtype=torch.float32) # Use float32 for division
    # Calculate total bits
    total_bits = original.numel() # Use numel() for total number of elements

    if total_bits == 0:
        return torch.tensor(0.0, device=original.device) # Avoid division by zero
    ber_percentage = (error_bits / total_bits) * 100
    return ber_percentage

# 2. Audio Encoder using Transformer Encoder Layers
class TransformerAudioEncoder(nn.Module):
    def __init__(self, feature_size, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super(TransformerAudioEncoder, self).__init__()
        # Ensure feature_size is divisible by nhead
        assert feature_size % nhead == 0, f"feature_size ({feature_size}) must be divisible by nhead ({nhead})"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, # Feature dimension per time step
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # Input shape: (seq_len, batch, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Optional: Add LayerNorm after the encoder
        self.norm = nn.LayerNorm(feature_size)

    def forward(self, x): # x shape: (seq_len, batch, feature)
        # TransformerEncoder expects src_mask for optional masking (we pass None)
        out = self.transformer_encoder(x, src_key_padding_mask=None, mask=None)
        out = self.norm(out)
        return out

# 3. Watermark Encoder (Simple FC layers to match feature size)
class TransformerWatermarkEncoder(nn.Module):
    def __init__(self, watermark_size, hidden_size=256, output_feature_size=512):
        super(TransformerWatermarkEncoder, self).__init__()
        # Assumes watermark input is flattened (batch, watermark_size)
        self.net = nn.Sequential(
            nn.Linear(watermark_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_feature_size)
        )

    def forward(self, x): # x shape: (batch, watermark_size) or (watermark_size,)
        if x.dim() == 1:
             x = x.unsqueeze(0) # Add batch dimension if missing
        return self.net(x) # (batch, output_feature_size)


class WatermarkDecoder(nn.Module):
    """
    Watermark Decoder for audio steganography.
    Input:  STFT spectrogram of watermarked audio, shape [B, C, H, W]
    Output: Recovered watermark message, shape [B, watermark_size]
    
    Compatible with paper: Tong et al., "Robust Audio Watermarking via Multi-scale Transformer", TASLP 2024.
    """
    def __init__(self, watermark_size, input_channels=2, hidden_dim=512):
        super().__init__()
        self.watermark_size = watermark_size

        # Stage 1: Multi-scale feature extraction (类似 U-Net encoder)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # downsample
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # downsample
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # downsample

        # Stage 2: Global context modeling (lightweight attention over spatial dims)
        # Use a small custom module to robustly produce a [B, hidden_dim] vector
        class GlobalAttn(nn.Module):
            def __init__(self, in_channels, hidden_dim, dropout=0.3):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                self.ln = nn.LayerNorm(hidden_dim)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
                self.act = nn.GELU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x: [B, C, H, W]
                # 1) normalize input spatially to 8x8
                x = F.adaptive_avg_pool2d(x, (8, 8))
                # 2) project channels
                x = self.conv(x)  # [B, hidden_dim, 8, 8]
                # 3) pool to (1,1) robustly
                x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, hidden_dim, 1, 1]
                x = x.view(x.size(0), -1)  # [B, hidden_dim]
                # 4) normalization and MLP
                x = self.ln(x)
                x = self.fc(x)
                x = self.act(x)
                x = self.dropout(x)
                return x

        self.global_attn = GlobalAttn(in_channels=256, hidden_dim=hidden_dim)

        # Stage 3: Message reconstruction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, watermark_size),
            nn.Sigmoid()  # assuming watermark is in [0, 1]; use nn.Tanh() if in [-1, 1]
        )

    def forward(self, x):
        # Accept either [B, C, H, W] or [B, H, W] (if C=1 omitted). Normalize to 4D.
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # x: [B, C, H, W], e.g., [B, 2, 128, 128]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # [B, 64, H/2, W/2]
        x = F.relu(self.conv3(x))  # [B, 128, H/4, W/4]
        x = F.relu(self.conv4(x))  # [B, 256, H/8, W/8]

        x = self.global_attn(x)    # [B, hidden_dim]
        msg = self.head(x)         # [B, watermark_size]
        return msg

# 4. Generator: Creates perturbation vector from fused features
class TransformerGenerator(nn.Module):
    def __init__(self, input_feature_size, hidden_size=512, output_feature_size=512):
        super(TransformerGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_feature_size)
        )

    def forward(self, x): # x shape: (batch, input_feature_size)
        return self.net(x) # (batch, output_feature_size)

# 5. Main Transformer-based Watermarking Model
class TransformerSpeechWatermarkingModel(nn.Module):
    def __init__(self, feature_size, watermark_size):
        super(TransformerSpeechWatermarkingModel, self).__init__()
        self.feature_size = feature_size
        self.watermark_size = watermark_size

        self.audio_encoder = TransformerAudioEncoder(feature_size=feature_size)
        # WatermarkEncoder output dim matches feature_size for fusion
        self.watermark_encoder = TransformerWatermarkEncoder(watermark_size=watermark_size, output_feature_size=feature_size)
        # Generator input is fused feature size (here, just feature_size)
        self.generator = TransformerGenerator(input_feature_size=feature_size, output_feature_size=feature_size)

        # Placeholder for extractor network (not implemented here)
        # In practice, you'd have a separate network to extract watermark from watermarked audio features.
        # For training consistency with existing framework (which compares input msg with decoded msg),
        # we'll simplify and return the encoded watermark.
        # A more realistic setup would have a dedicated decoder/extraction network.

    def forward(self, audio_features, watermark_features):
        """
        Args:
            audio_features: (seq_len, batch, feature_size)
                            Features from STFT (e.g., magnitude).
            watermark_features: (batch, watermark_size)
                                Flattened watermark data.
        Returns:
            watermarked_audio_features: (seq_len, batch, feature_size)
                                        Modified audio features containing watermark.
            extracted_watermark_features_placeholder: (batch, feature_size)
                                                     Simplified output matching encoder's output dim.
        """
        # 1. Encode audio features
        encoded_audio = self.audio_encoder(audio_features) # (seq_len, batch, feature_size)

        # 2. Encode watermark features
        encoded_watermark = self.watermark_encoder(watermark_features) # (batch, feature_size)

        # 3. Fuse features (example: broadcast watermark to sequence length and add)
        # encoded_watermark: (batch, feature_size) -> unsqueeze to (1, batch, feature_size) for broadcasting
        encoded_watermark_expanded = encoded_watermark.unsqueeze(0) # (1, batch, feature_size)
        fused_features = encoded_audio + encoded_watermark_expanded # Broadcasting add: (seq_len, batch, feature_size)

        # 4. Generate perturbation vector (pool fused features across time)
        pooled_features = torch.mean(fused_features, dim=0) # (batch, feature_size)
        perturbation_vector = self.generator(pooled_features) # (batch, feature_size)
        # Broadcast perturbation back to sequence dimension
        perturbation_expanded = perturbation_vector.unsqueeze(0) # (1, batch, feature_size)

        # 5. Apply perturbation to encoded audio to get watermarked features
        watermarked_audio_features = encoded_audio + perturbation_expanded # (seq_len, batch, feature_size)

        # 6. Return watermarked features and a placeholder for extracted watermark
        # Simplification: Return the encoded watermark as the "extracted" one.
        # In reality, an extraction network would process watermarked_audio_features.
        extracted_watermark_features_placeholder = encoded_watermark # (batch, feature_size)
        # If the goal is to recover the original watermark_size vector,
        # an additional decoder layer would be needed here.
        # e.g., self.watermark_decoder = nn.Linear(feature_size, watermark_size)
        # extracted_watermark_features = self.watermark_decoder(extracted_watermark_features_placeholder)

        return watermarked_audio_features, extracted_watermark_features_placeholder


# --- Updated Factory Function for Models ---

# Define default hyperparameters for the transformer model (can be overridden via config/hparams)
TRANSFORMER_DEFAULTS = {
    'feature_size': 129, # Match spectrogram feature dimension (Freq bins) - CHECK YOUR DATA
    'watermark_size': 128, # Define your watermark size
    'audio_encoder_layers': 2,
    'audio_encoder_nhead': 3, # Must divide feature_size (129 is not divisible by many, 3 works for 129)
    'audio_encoder_dim_feedforward': 512,
    'audio_encoder_dropout': 0.1,
    'wm_encoder_hidden_size': 256,
    'generator_hidden_size': 512,
}

def get_transformer_models(config):
    """Factory function to create transformer-based models."""
    # Use defaults, override with config values if present
    feature_size = getattr(config, 'feature_size', TRANSFORMER_DEFAULTS['feature_size'])
    watermark_size = getattr(config, 'watermark_size', TRANSFORMER_DEFAULTS['watermark_size'])

    # Validate nhead divides feature_size
    nhead = getattr(config, 'audio_encoder_nhead', TRANSFORMER_DEFAULTS['audio_encoder_nhead'])
    assert feature_size % nhead == 0, f"feature_size ({feature_size}) must be divisible by nhead ({nhead})"

    # Create the main model
    model = TransformerSpeechWatermarkingModel(
        feature_size=feature_size,
        watermark_size=watermark_size
    )

    # For compatibility with existing solver expecting encoder/decoder,
    # we treat the whole model as 'encoder' and create a dummy 'decoder'
    # that simply passes through the extracted watermark part.
    # This is a workaround for the existing structure.
    class DummyDecoder(nn.Module):
        def __init__(self, model_ref):
            super().__init__()
            self.model_ref = model_ref # Keep reference if needed (though not used here)
        def forward(self, watermarked_features_and_encoded_wm):
            # Input is the second output from model.forward (extracted_watermark_placeholder)
            # In our case, it's already shaped correctly (batch, feature_size)
            # If it were (seq_len, batch, feature_size), we'd need to pool/process it.
            # For simplicity, assume it's ready.
            return watermarked_features_and_encoded_wm # Return the "extracted" watermark

    encoder = model # The combined model acts as the encoder
    decoder = DummyDecoder(model) # Dummy decoder extracts the watermark part

    # Optimizer and Scheduler (using config values)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr) # Use config.lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # Example scheduler

    return encoder, decoder, optimizer, scheduler


class MultiScaleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 注意：使用 batch_first=True，输入应为 [B, N, C]
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def __init__(self, dim, num_heads=4, max_tokens=4096):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_tokens = max_tokens

        # 注意：使用 batch_first=True，输入应为 [B, N, C]
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        """
        Memory-conservative forward using optional adaptive downsampling:
        - If H*W <= self.max_tokens: perform normal full attention.
        - Else: adaptively pool spatial dims to (h', w') such that h'*w' <= max_tokens,
          run attention on pooled tokens, then upsample attention output back to (H, W)
          and add as residual before the MLP.
        """
        B, C, H, W = x.shape
        N = H * W

        # Fast path: if small enough, do full attention (same as before)
        if N <= self.max_tokens:
            x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
            x_norm = self.norm1(x_flat)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # [B, N, C]
            x_res = x_flat + attn_out
            x_mlp = self.norm2(x_res)
            x_mlp = self.mlp(x_mlp)
            x_out = x_res + x_mlp  # [B, N, C]
            x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return x_out

        # Otherwise, compute pooling target so that pooled_N <= max_tokens
        factor = math.ceil((N / float(self.max_tokens)) ** 0.5)
        h_p = max(1, H // factor)
        w_p = max(1, W // factor)

        # Pool (downsample) spatially
        x_pooled = F.adaptive_avg_pool2d(x, (h_p, w_p))  # [B, C, h_p, w_p]

        # Flatten pooled tokens and apply attention
        Np = h_p * w_p
        x_p_flat = x_pooled.permute(0, 2, 3, 1).reshape(B, Np, C)
        x_p_norm = self.norm1(x_p_flat)
        attn_p, _ = self.attn(x_p_norm, x_p_norm, x_p_norm)  # [B, Np, C]
        x_p_res = x_p_flat + attn_p
        x_p_mlp = self.norm2(x_p_res)
        x_p_mlp = self.mlp(x_p_mlp)
        x_p_out = x_p_res + x_p_mlp  # [B, Np, C]

        # Reshape pooled output and upsample back to original spatial size
        x_p_out = x_p_out.reshape(B, h_p, w_p, C).permute(0, 3, 1, 2)  # [B, C, h_p, w_p]
        x_up = F.interpolate(x_p_out, size=(H, W), mode='bilinear', align_corners=False)  # [B, C, H, W]

        # Combine pooled-attention result with original via residual, then MLP
        x_orig_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        x_up_flat = x_up.permute(0, 2, 3, 1).reshape(B, N, C)

        x_res = x_orig_flat + x_up_flat
        x_mlp = self.norm2(x_res)
        x_mlp = self.mlp(x_mlp)
        x_out = x_res + x_mlp

        x_out = x_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x_out


# ----------------- Module-level PerturbationGenerator -----------------
class PerturbationGenerator(nn.Module):
    """Generates perturbation S (module-level) from acoustic features and watermark features.
    Defaults mirror previous local implementation: ac_channels=64, wm_channels=512.
    """
    def __init__(self, ac_channels=64, wm_channels=512, proj_channels=64, mid_channels=128, out_channels=1, n_blocks=3):
        super().__init__()
        self.proj_wm = nn.Conv2d(wm_channels, proj_channels, kernel_size=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(ac_channels + proj_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            *[MultiScaleTransformerBlock(dim=mid_channels) for _ in range(n_blocks)],
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, ac_feat, wm_feat):
        B, _, H, W = ac_feat.shape
        # if wm_feat is a vector, expand
        if wm_feat.dim() == 2:
            wm_map = wm_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
        elif wm_feat.dim() == 4:
            # interpolate spatial watermark to match acoustic dims
            _, _, h, w = wm_feat.shape
            if (h, w) != (H, W):
                wm_map = F.interpolate(wm_feat, size=(H, W), mode='bilinear', align_corners=False)
            else:
                wm_map = wm_feat
        else:
            wm_map = wm_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)

        wm_map = self.proj_wm(wm_map)
        fused = torch.cat([ac_feat, wm_map], dim=1)
        S = self.encoder(fused)
        return S  # [B, out_channels, H, W]

class TransformerEncoder(nn.Module):
    def __init__(self, input_nc=1, watermark_nc=1, n_downsampling=2, ngf=64, n_blocks=6, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """Encoder that takes carrier and watermark, outputs watermarked features."""
        super(TransformerEncoder, self).__init__()
        self.input_nc = input_nc
        self.watermark_nc = watermark_nc
        self.ngf = ngf
        
        # Initial convolutional layers to increase channels and downsample slightly
        # Mimicking the initial layers before transformer blocks in Fig. 2
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc + watermark_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
                 
        # Downsampling layers (optional, depends on desired feature map size)
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        # Transformer Blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [MultiScaleTransformerBlock(dim=ngf * mult)]

        # Upsampling/Perturbation Generation Layers (Paper uses G to generate S)
        # This part is tricky. The paper adds perturbation S to original X_ac.
        # Our approach: Encoder outputs the perturbation S.
        # The final watermarked feature X_w_ac = X_ac + S is computed outside (e.g., in forward_encoder).
        # So, this encoder's final layers should produce a tensor S of same shape as input feature (after initial convs).
        # Let's adjust the architecture accordingly.
        
        # We'll assume the input after initial conv/down is (B, ngf*mult, H', W')
        # And we want to output a perturbation S of the same shape.
        # We can add more transformer blocks or conv layers here.
        # For simplicity, let's add a few conv layers to refine the perturbation.
        # Note: Paper uses upsampling in Generator G, but our input is already downsampled.
        # If we want to match dimensions exactly, we might need to upsample back.
        # However, for direct integration, let's design it to output a residual-like perturbation.
        
        # Add final layers to refine perturbation (assuming no further downsampling)
        # Output channels should match the feature dimension after initial processing
        model += [
            nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1),
            norm_layer(ngf * mult),
            nn.ReLU(True),
            nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, padding=1),
            norm_layer(ngf * mult),
            nn.ReLU(True),
            # Final layer to produce perturbation S with tanh activation to bound it
            # Output channels = input_nc (typically 1 for magnitude spectrogram)
            nn.Conv2d(ngf * mult, input_nc, kernel_size=3, padding=1),
            nn.Tanh() # Bounded perturbation
        ]
        
        self.model = nn.Sequential(*model)
        
        # Store downsampling factor for later use if needed
        self.downsample_factor = 2 ** n_downsampling 

    def forward(self, carrier, msg):
        # Concatenate carrier and message along the channel dimension
        x = torch.cat([carrier, msg], dim=1) # Shape: (B, input_nc + watermark_nc, H, W)
        perturbation_S = self.model(x) # Shape: (B, input_nc, H, W)
        
        if perturbation_S.shape != carrier.shape:
            perturbation_S = torch.nn.functional.interpolate(
            perturbation_S, size=carrier.shape[2:], mode='bilinear', align_corners=False
        )
        # The actual watermarked feature is computed as X_w_ac = carrier + S
        # This addition happens outside this encoder (e.g., in solver.forward_encoder or main.py logic)
        # Returning S allows flexibility in how it's added.
        return perturbation_S # This is S, not X_w_ac yet


# --- 新增：Transformer Decoder (参考论文Fig. 1 Extraction Process) ---
class TransformerDecoder(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_downsampling=2, ngf=64, n_blocks=6, norm_layer=nn.BatchNorm2d):
        """Decoder that takes watermarked features, outputs extracted watermark."""
        super(TransformerDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        
        # Initial convolutional layers (similar to encoder's beginning)
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
                 
        # Downsampling layers (optional, to extract high-level features)
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        # Transformer Blocks for feature extraction
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [MultiScaleTransformerBlock(dim=ngf * mult)]

        # Upsampling layers to restore original dimensions
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # Use ConvTranspose2d for upsampling
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        # Final layer to produce the watermark
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()] # Watermark is typically bounded [0, 1]

        self.model = nn.Sequential(*model)

    # def forward(self, watermarked_features):
    #     return self.model(watermarked_features)
    def forward(self, watermarked_features, target_size=None):
        out = self.model(watermarked_features)
        if target_size is not None:
            _, _, H, W = target_size
            h, w = out.shape[2], out.shape[3]
            if h != H or w != W:
                dh = h - H
                dw = w - W
                # 居中裁剪（更合理）
                out = out[:, :, dh//2 : dh//2 + H, dw//2 : dw//2 + W]
        return out