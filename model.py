import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from portable_udh import get_hiding_unet, get_reveal_net, count_params
from config import device
# from options import HiDDenConfiguration
# from model.conv_bn_relu import ConvBNRelu
import torch.nn.functional as F

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
    encoder.load_state_dict(torch.load(join(ckpt_dir, "encoder.ckpt")))
    decoder.load_state_dict(torch.load(join(ckpt_dir, "decoder.ckpt")))
    logger.info("loader models")

def get_models(hparams):
    
    if hparams.model_type == 'normal':
        dec_um_conv_dim = 1 + 64
        # dec_um_conv_dim=64
        encoder = FullEncoder(
            block_type=hparams.block_type,
            enc_n_layers=hparams.enc_n_layers,
            dec_um_conv_dim=dec_um_conv_dim,
            dec_c_n_layers=hparams.dec_c_n_layers
        ).to(device)

        decoder = MsgDecoder(
            conv_dim=1,
            block_type=hparams.block_type
        ).to(device)

    elif hparams.model_type == 'unet':
        encoder = get_hiding_unet(input_nc=1+1, output_nc=1, num_downs=5, norm='instance', use_dropout=False).to(device)
        decoder = get_reveal_net(input_nc=1, output_nc=1, norm='instance').to(device)

    else:
        raise ValueError(f"Unsupported model_type: {hparams.model_type}")

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=hparams.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    if hparams.load_ckpt:
        load_models(encoder, decoder, hparams.load_ckpt)

    logger.debug(encoder)
    logger.debug(decoder)

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
