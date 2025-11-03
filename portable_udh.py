# encoding: utf-8
"""
Audio-grade UNet without skip connection for steganography.
- All skip cat removed -> hidden layer must carry all info.
- Default 1-channel I/O (audio spectrogram).
Compatible with original API.
"""
from typing import Optional, Union
import torch
import torch.nn as nn
import functools

# ---------- norm helper ----------
def _norm_layer_by_name(name: Optional[str]):
    if name is None or name == 'none':
        return None
    if name == 'instance':
        return nn.InstanceNorm2d
    if name == 'batch':
        return nn.BatchNorm2d
    raise ValueError('Unknown norm: %s' % name)

# ---------- UNet block (NO skip concat) ----------
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
                 innermost=False, norm_layer=None, use_dropout=False,
                 output_function=nn.Sigmoid):
        super().__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer is None:
            use_bias = True
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu   = nn.ReLU(True)

        downnorm = norm_layer(inner_nc) if norm_layer else None
        upnorm   = norm_layer(outer_nc) if norm_layer else None

        if outermost:               # 最外层：下 -> 子模块 -> 上
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up   = [uprelu, upconv, output_function()]
            model = down + [submodule] + up
        elif innermost:             # 最内层：下 -> 上
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up   = [uprelu, upconv] + ([upnorm] if upnorm else [])
            model = down + up
        else:                       # 中间层：下 -> 子模块 -> 上
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv] + ([downnorm] if downnorm else [])
            up   = [uprelu, upconv] + ([upnorm]   if upnorm   else [])
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    # ******** 关键：无 skip ********
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_model = self.model(x)
            # 不做 torch.cat([x, x_model], 1)
            return x_model

# ---------- 生成器 ----------
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer: Optional[str] = 'instance',
                 use_dropout=False, output_function=nn.Sigmoid,
                 scale_tanh: bool = False):
        super().__init__()
        nl = _norm_layer_by_name(norm_layer)
        # 从内向外搭积木
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                             submodule=None, norm_layer=nl, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None,
                                                 submodule=unet_block, norm_layer=nl,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None,
                                             submodule=unet_block, norm_layer=nl)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None,
                                             submodule=unet_block, norm_layer=nl)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None,
                                             submodule=unet_block, norm_layer=nl)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc,
                                             submodule=unet_block, outermost=True,
                                             norm_layer=nl, output_function=output_function)

        self.model = unet_block
        self.tanh = (output_function == nn.Tanh)
        self.scale_tanh = scale_tanh
        if self.tanh and self.scale_tanh:
            self.factor = 10.0 / 255.0
        else:
            self.factor = 1.0

    def forward(self, x):
        out = self.factor * self.model(x)
        # 自动裁剪宽度到输入x的宽度
        if out.shape[-1] != x.shape[-1]:
            min_width = min(out.shape[-1], x.shape[-1])
            out = out[..., :min_width]
        return out
        # return self.factor * self.model(x)

# ---------- 揭示网络 ----------
class RevealNet(nn.Module):
    def __init__(self, input_nc, output_nc, nhf=64,
                 norm_layer: Optional[str] = 'instance',
                 output_function=nn.Sigmoid):
        super().__init__()
        nl = _norm_layer_by_name(norm_layer)
        layers = []
        # conv1
        layers += [nn.Conv2d(input_nc, nhf, 3, 1, 1), nn.ReLU(inplace=True)]
        if nl: layers.append(nl(nhf))
        # conv2
        layers += [nn.Conv2d(nhf, nhf * 2, 3, 1, 1), nn.ReLU(inplace=True)]
        if nl: layers.append(nl(nhf * 2))
        # conv3
        layers += [nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1), nn.ReLU(inplace=True)]
        if nl: layers.append(nl(nhf * 4))
        # conv4
        layers += [nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1), nn.ReLU(inplace=True)]
        if nl: layers.append(nl(nhf * 2))
        # conv5
        layers += [nn.Conv2d(nhf * 2, nhf, 3, 1, 1), nn.ReLU(inplace=True)]
        if nl: layers.append(nl(nhf))
        # final
        layers.append(nn.Conv2d(nhf, output_nc, 3, 1, 1))
        if output_function == nn.Sigmoid:
            layers.append(nn.Sigmoid())
        elif output_function == nn.Tanh:
            layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ---------- 工厂函数 ----------
def get_hiding_unet(input_nc: int, output_nc: int, num_downs: int = 5, ngf: int = 64,
                    norm: Optional[str] = 'instance', use_dropout: bool = False,
                    output_function: Union[str, nn.Module] = 'sigmoid', scale_tanh: bool = False):
    if isinstance(output_function, str):
        of = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[output_function.lower()]
    else:
        of = output_function
    return UnetGenerator(input_nc=input_nc, output_nc=output_nc, num_downs=num_downs,
                         ngf=ngf, norm_layer=norm, use_dropout=use_dropout,
                         output_function=of, scale_tanh=scale_tanh)

def get_reveal_net(input_nc: int, output_nc: int, nhf: int = 64,
                   norm: Optional[str] = 'instance',
                   output_function: Union[str, nn.Module] = 'sigmoid'):
    if isinstance(output_function, str):
        of = {'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[output_function.lower()]
    else:
        of = output_function
    return RevealNet(input_nc=input_nc, output_nc=output_nc, nhf=nhf,
                     norm_layer=norm, output_function=of)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())