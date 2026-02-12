from collections import defaultdict
from os import makedirs
from os.path import join
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm, trange

from experiment import Experiment
from hparams import AUDIO_LEN, HOP_LENGTH, N_FFT
from stft.stft import STFT
from config import get_hparams
from config import gl_hparams
from config import device
# from pesq import pesq as pesq_fn
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import warnings
import math

if(gl_hparams==None): 
    gl_hparams = get_hparams()

spect_audio_shape = (gl_hparams.batch_size, 1, 129, 378)



class WatermarkAutoEncoder(nn.Module):
    def __init__(self, encoder_wm, decoder):
        super().__init__()
        self.encoder_wm = encoder_wm
        self.decoder = decoder

    def forward(self, msg):
        # I'_ae = De(En_wm(I))
        encoded = self.encoder_wm(msg)
        decoded = self.decoder(encoded)
        return decoded

class TransformerWatermarkEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.En_ac = MultiScaleTransformerEncoder(...)      # 声学编码器
        self.En_wm = VGGStyleWatermarkEncoder(...)         # 水印编码器
        self.GeneratorG = MultiScaleTransformerGenerator(...)  # 扰动生成器



# def forward_encoder(encoder, carrier, msg, model_type):
#     if model_type == 'unet':
#         x = torch.cat([carrier, msg], dim=1)
#         return encoder(x)
#     elif model_type == 'transformer':
#          # Encoder outputs the perturbation S
#         perturbation_S = encoder(carrier, msg)
#         # Calculate the final watermarked acoustic feature X_w_ac = X_ac + S
#         # This aligns with formula (4) in the paper: X_w_ac = Xen_ac ⊕ S (where ⊕ is addition)
#         # Ensure S matches carrier dimensions
#         if perturbation_S.shape[2:] != carrier.shape[2:]:
#             # Resize perturbation if needed (should ideally match inside encoder)
#             perturbation_S = F.interpolate(perturbation_S, size=carrier.shape[2:], mode='bilinear', align_corners=False)
#         container = carrier + perturbation_S
#         # Optional: Clamping might be needed if S isn't strictly bounded correctly
#         # container = torch.clamp(container, min=0, max=1) # Adjust clamp range if data isn't [0,1]
#         return container

#     else:  # original
#         return encoder(carrier, msg)

def forward_encoder(encoder, carrier, msg, model_type):
    """Forward wrapper that accepts carrier/msg with or without channel dim.
    - Inputs can be [B, H, W] (no channel) or [B, C, H, W]. Internally we ensure 4D where needed.
    - Returns carrier container and perturbation, both possibly squeezed to 3D if they have single channel.
    """
    if model_type == 'unet':
        # accept [B,H,W] or [B,C,H,W]
        c_in = carrier.unsqueeze(1) if carrier.dim() == 3 else carrier
        m_in = msg.unsqueeze(1) if msg.dim() == 3 else msg
        container = encoder(torch.cat([c_in, m_in], dim=1))
        # squeeze channel dim for external API if single-channel
        container_out = container.squeeze(1) if container.dim() == 4 and container.shape[1] == 1 else container
        return container_out, None  # 返回 (container, perturbation_S)

    elif model_type == 'transformer':
        # 兼容性检查：期望 encoder 是包含子模块的对象
        if not hasattr(encoder, 'En_ac') or not hasattr(encoder, 'En_wm') or not hasattr(encoder, 'GeneratorG'):
            raise ValueError(
                "For 'transformer' mode, encoder must have attributes: En_ac, En_wm, GeneratorG"
            )
        
        En_ac = encoder.En_ac
        En_wm = encoder.En_wm
        GeneratorG = encoder.GeneratorG

        # Accept 3D carrier by adding channel dim for internal processing
        c_in = carrier.unsqueeze(1) if carrier.dim() == 3 else carrier
        x_en_ac = En_ac(c_in)      # Eq.(2)

        # msg can be vector [B, L] or spectrogram [B, C, H, W] — En_wm handles both
        # i_en = En_wm(msg)             # Eq.(3)

        # Ensure watermark features have spatial dims compatible with acoustic features
        # - If i_en is a vector [B, feat_dim], expand to [B, feat_dim, H, W]
        # - If i_en is [B, C, h, w], interpolate to match
        # if i_en.dim() == 2:
        #     B = i_en.shape[0]
        #     feat = i_en.shape[1]
        #     H, W = x_en_ac.shape[2], x_en_ac.shape[3]
        #     i_en = i_en.view(B, feat, 1, 1).expand(-1, -1, H, W).contiguous()
        # elif i_en.dim() == 4:
        #     if x_en_ac.shape[2:] != i_en.shape[2:]:
        #         i_en = F.interpolate(i_en, size=x_en_ac.shape[2:], mode='bilinear', align_corners=False)
        # else:
        #     # Try to reshape generically, otherwise raise informative error
        #     try:
        #         B = i_en.shape[0]
        #         C = i_en.shape[1]
        #         H, W = x_en_ac.shape[2], x_en_ac.shape[3]
        #         i_en = i_en.view(B, C, 1, 1).expand(-1, -1, H, W).contiguous()
        #     except Exception:
        #         raise ValueError(f"Unsupported watermark encoding shape: {i_en.shape}")

        # Fuse and generate perturbation S
        # GeneratorG expects acoustic features and watermark features separately
        perturbation_S = GeneratorG(x_en_ac, msg)

        # Ensure S matches original carrier shape (not encoded!)
        # carrier may be 3D ([B,H,W]) or 4D ([B,C,H,W]); use c_in for spatial reference
        # if perturbation_S.shape[2:] != c_in.shape[2:]:
        #     perturbation_S = F.interpolate(perturbation_S, size=c_in.shape[2:], mode='bilinear', align_corners=False)

        container =c_in + perturbation_S #x_en_ac #

        # squeeze channel axis for external API if single-channel
        container_out = container.squeeze(1) if container.dim() == 4 and container.shape[1] == 1 else container
        S_out = perturbation_S.squeeze(1) if perturbation_S.dim() == 4 and perturbation_S.shape[1] == 1 else perturbation_S
        return container_out, S_out

    else:  # original / default mode
        container = encoder(carrier, msg)
        return container, None


def snr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    try:
        assert orig.shape == recon.shape == spect_audio_shape
    except AssertionError:
        print(f"orig.shape:{orig.shape},recon.shape:{recon.shape},spect_audio_shape:{spect_audio_shape}")
    N = orig.shape[-1] * orig.shape[-2]
    orig, recon = orig.cpu(), recon.cpu()
    # rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
    # rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
    # snr = 10 * torch.log10((rms1 / rms2) ** 2)



    rms1 = torch.sqrt(torch.mean(orig ** 2))
    rms2 = torch.sqrt(torch.mean((orig - recon) ** 2))
    snr_val = 10 * torch.log10((rms1 / (rms2 + 1e-8)) ** 2)
    return snr_val.item()  # only .item() at end
    # return snr

# def training_step(carrier: torch.Tensor, carrier_reconst: torch.Tensor, msg: torch.Tensor, msg_reconst: torch.Tensor, lambda_carrier, lambda_msg, loss_type) -> Tuple[torch.Tensor, defaultdict]:
#     try:
#         assert carrier.shape == carrier_reconst.shape == msg.shape == msg_reconst.shape == spect_audio_shape
#     except AssertionError:
#         print(f"carrier.shape:{carrier.shape},carrier_reconst.shape:{carrier_reconst.shape},msg.shape:{msg.shape},msg_reconst.shape:{msg_reconst.shape},spect_audio_shape:{spect_audio_shape}")
#     losses_log = defaultdict(int)
#     carrier, msg = carrier.to(device), msg.to(device)
#     loss = F.mse_loss if loss_type == 'mse' else F.l1_loss
#     carrier_loss = loss(carrier_reconst, carrier)
#     msg_loss = loss(msg_reconst, msg)
#     losses_log['carrier_loss'] = carrier_loss.item()
#     losses_log['msg_loss'] = msg_loss.item()
#     loss = lambda_carrier * carrier_loss + lambda_msg * msg_loss
#     return loss, losses_log

def training_step(
    carrier: torch.Tensor,
    carrier_reconst: torch.Tensor,
    msg: torch.Tensor,
    msg_reconst: torch.Tensor,
    msg_ae_reconst: torch.Tensor = None,   # optional
    perturbation_S: torch.Tensor = None,   # optional
    lambda_carrier=1.0,
    lambda_msg=1.0,
    lambda_ae=1.0,
    lambda_pert=0.1,
    epsilon=0.01,
    loss_type='mse'
) -> Tuple[torch.Tensor, defaultdict]:
    # Normalize inputs to 4D tensors for loss computations (support external 3D tensors where C==1)
    if carrier.dim() == 3:
        carrier = carrier.unsqueeze(1)
    if carrier_reconst.dim() == 3:
        carrier_reconst = carrier_reconst.unsqueeze(1)
    if msg.dim() == 3:
        msg = msg.unsqueeze(1)
    if msg_reconst.dim() == 3:
        msg_reconst = msg_reconst.unsqueeze(1)
    # Only call .dim() on tensor-like objects. Some code paths may supply floats/None.
    if torch.is_tensor(msg_ae_reconst) and msg_ae_reconst.dim() == 3:
        msg_ae_reconst = msg_ae_reconst.unsqueeze(1)
    if torch.is_tensor(perturbation_S) and perturbation_S.dim() == 3:
        perturbation_S = perturbation_S.unsqueeze(1)

    losses_log = defaultdict(float)
    loss_fn = F.mse_loss if loss_type == 'mse' else F.l1_loss

    carrier_loss = loss_fn(carrier_reconst, carrier)
    msg_loss = loss_fn(msg_reconst, msg)
    total_loss = lambda_carrier * carrier_loss + lambda_msg * msg_loss

    losses_log['carrier_loss'] = carrier_loss.item()
    losses_log['msg_loss'] = msg_loss.item()

    # # Optional: AE loss (for transformer mode)
    # if msg_ae_reconst is not None:
    #     ae_loss = loss_fn(msg_ae_reconst, msg)
    #     total_loss += lambda_ae * ae_loss
    #     losses_log['ae_loss'] = ae_loss.item()

    # # Optional: Perturbation loss
    # if perturbation_S is not None:
    #     pert_loss = F.mse_loss(perturbation_S, torch.full_like(perturbation_S, epsilon))
    #     total_loss += lambda_pert * pert_loss
    #     losses_log['pert_loss'] = pert_loss.item()

    return total_loss, losses_log

def save_models(ckpt_dir, encoder, decoder, suffix=''):
    logger.info(f"saving model to: {ckpt_dir}\n==> suffix: {suffix}")
    makedirs(join(ckpt_dir, suffix), exist_ok=True)
    torch.save(encoder.state_dict(), join(ckpt_dir, suffix, "encoder.ckpt"))
    torch.save(decoder.state_dict(), join(ckpt_dir, suffix, "decoder.ckpt"))

class Solver(object):
    def __init__(self, config):
        self.config = config
        self.num_iters = config.num_iters
        self.cur_iter = 0
        self.dataset= config.dataset
        self.num_samples = int({'timit': AUDIO_LEN * 16000,
                                'mini':  AUDIO_LEN * 16000}[self.dataset])

        # create experimentsamples_dir
        self.experiment    = Experiment(config.run_dir, use_comet=False, use_wandb=getattr(config, 'use_wandb', False))
        self.run_dir       = self.experiment.dir
        self.ckpt_dir      = self.experiment.ckpt_dir
        self.code_dir      = self.experiment.code_dir
        self.load_ckpt_dir = config.load_ckpt
        self.samples_dir   = join(self.run_dir, 'samples')
        self.experiment.save_hparams(config)

        self.num_workers        = config.num_workers
        self.save_model_every   = config.save_model_every
        self.sample_every       = config.sample_every
        self.print_every        = 10
        self.mode               = 'test'
        self.model_type          = config.model_type

        self.create_dirs()
        torch.manual_seed(10)

        self.stft = STFT(N_FFT, HOP_LENGTH)
        self.stft.num_samples = self.num_samples
        # self._window = torch.hann_window(N_FFT).to(device)
        # self.carrier_pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb',  # 'wb' for wideband, 'nb' for narrowband   
        #                                                                   ).to(device)
        
        # self.message_pesq_metric = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
        # torch.autograd.set_detect_anomaly(True)        
        # logging
        logger.add(join(self.run_dir, "stdout.log"))
        self.global_step = 0

  
    def log_losses(self, losses, iteration=None):
        if iteration is None:
            iteration = self.cur_iter
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        try:
            self.global_step += 1
            step_for_logging = self.global_step
        except Exception:
            step_for_logging = iteration

        self.experiment.log_metric(losses, step=step_for_logging)

    def create_dirs(self):
        makedirs(self.samples_dir, exist_ok=True)
        logger.info("created dirs")

    def train_gan(self, train_dataloader, val_dataloader, encoder, decoder, discriminator, optimizer_G, optimizer_D, scheduler):
        epoch_it = trange(self.num_iters)

        for epoch in epoch_it:
            lr = optimizer_G.param_groups[0]['lr'] 
            epoch_it.set_description(f"Epoch {epoch}, LR={lr}")
            epoch_loss = defaultdict(list)
            it = tqdm(train_dataloader)
            logger.debug("train mode")
            self.mode = 'train'

            encoder.train()
            decoder.train()

            for cur_iter, (carrier, msg) in enumerate(it):
                try:
                    assert carrier.shape == msg.shape == spect_audio_shape
                except AssertionError:
                    print(f"carrier.shape:{carrier.shape},msg.shape:{msg.shape},spect_audio_shape:{spect_audio_shape}")
                
                carrier, msg = carrier.to(device), msg.to(device)

                if cur_iter % gl_hparams.freeze_num == 0:
                    for p in discriminator.parameters():
                        p.requires_grad = True
                    # === 1. 训练判别器 ===
                    optimizer_D.zero_grad()
                    # 真实数据
                    real_out = discriminator(carrier)
                    loss_D_real = F.binary_cross_entropy_with_logits(real_out, torch.ones_like(real_out))#loss = -log(D(G(z)))
                    # 生成数据
                    carrier_reconst = encoder(carrier, msg)
                    fake_out = discriminator(carrier_reconst.detach())
                    loss_D_fake = F.binary_cross_entropy_with_logits(fake_out, torch.zeros_like(fake_out))
                    # 判别器总损失
                    loss_D = (loss_D_real + loss_D_fake) / 2
                    loss_D.backward()
                    optimizer_D.step()
                else:
                    # 冻结判别器权重
                    for p in discriminator.parameters():
                        p.requires_grad = False

                # === 2. 训练生成器 ===
                optimizer_G.zero_grad()
                carrier_reconst = encoder(carrier, msg)
                fake_out = discriminator(carrier_reconst)
                # 生成器希望判别器判为真
                loss_G_gan = F.binary_cross_entropy_with_logits(fake_out, torch.ones_like(fake_out))
                # 加上原有的重建损失
                msg_reconst = decoder(carrier_reconst)

                loss_recon, losses_log = training_step(carrier, carrier_reconst, msg, msg_reconst, self.config.lambda_carrier_loss, self.config.lambda_msg_loss, self.config.loss_type)
                loss_G = loss_G_gan + loss_recon
                loss_G.backward()
                optimizer_G.step()

                # 在train_gan的for epoch in epoch_it:循环内，每个epoch结束时添加如下代码
            # log lambda values via Solver logging helper to ensure monotonic steps
            self.log_losses({
                "lambda_carrier": self.config.lambda_carrier_loss,
                "lambda_msg": self.config.lambda_msg_loss
            }, iteration=epoch)

            for k, v in list(epoch_loss.items()):
                epoch_loss["epoch_" + k] = np.mean(v)
                epoch_loss.pop(k)
            epoch_loss['lr'] = lr
            self.log_losses(epoch_loss, iteration=epoch)

            # epoch stats already logged via self.log_losses above; nothing extra needed here

            # 保存模型


            save_models(self.ckpt_dir, encoder, decoder, suffix=str(epoch+1) + "_epoch")

            # 验证集评估
            self.log_losses(self.test(val_dataloader, encoder, decoder, data='val'), iteration=epoch)
        logger.info("finished training!")


    # def train(self, train_dataloader, val_dataloader, encoder, decoder, optimizer, scheduler):
    #     # start of training loop
    #     # logger.info("start training...")
    #     epoch_it = trange(self.num_iters)

    #     for epoch in epoch_it:
    #         lr = optimizer.param_groups[0]['lr']
    #         epoch_it.set_description(f"Epoch {epoch}, LR={lr}")
    #         epoch_loss = defaultdict(list)
    #         it = tqdm(train_dataloader)
    #         logger.debug("train mode")
    #         self.mode = 'train'
            
    #         encoder.train()
    #         decoder.train()

    #         # inner epoch loop
    #         for cur_iter, (carrier, msg) in enumerate(it):
    #             try:
    #                 assert carrier.shape == msg.shape == spect_audio_shape
    #             except AssertionError:
    #                 print(f"carrier.shape:{carrier.shape},msg.shape:{msg.shape},spect_audio_shape:{spect_audio_shape}")
    #             # feedforward and suffer loss
    #             carrier, msg = carrier.to(device), msg.to(device)
    #             # carrier_reconst = encoder(carrier, msg)
    #             carrier_reconst = forward_encoder(encoder, carrier, msg, self.model_type)

    #             if self.model_type == 'transformer':
    #                 msg_reconst = decoder(carrier_reconst, target_size=msg.shape)
    #             else:
    #                 msg_reconst = decoder(carrier_reconst)

    #             loss, losses_log = training_step(carrier, carrier_reconst, msg, msg_reconst, self.config.lambda_carrier_loss, self.config.lambda_msg_loss, self.config.loss_type)

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             if cur_iter % len(train_dataloader) == 0:
    #                 scheduler.step()

    #             # log stuff
    #             if cur_iter % self.print_every == 0:
    #                 log = f"[{cur_iter}/{len(train_dataloader)}]"
    #                 for loss_name, loss_value in losses_log.items():
    #                     log += f", {loss_name}: {loss_value:.4f}"
    #                 it.set_description(log)
    #             self.log_losses(losses_log, iteration=cur_iter)

    #             # log epoch losses
    #             for k,v in losses_log.items():
    #                 epoch_loss[k].append(v)

    #         # calc epoch stats
    #         for k,v in list(epoch_loss.items()):
    #             epoch_loss["epoch_" + k] = np.mean(v)
    #             epoch_loss.pop(k)
    #         epoch_loss['lr'] = lr
    #         self.log_losses(epoch_loss, iteration=epoch)

    #         # epoch stats are already logged via self.log_losses
    #         print("=============================================================\n")
    #         # print(f"self.ckpt_dir: {self.ckpt_dir}")
    #         # if(self.ckpt_dir is None):
    #         gl_hparams.load_ckpt = self.ckpt_dir 
    #         print(f"self.ckpt_dir: {self.ckpt_dir}, gl_hparams.load_ckpt: {gl_hparams.load_ckpt}\n")
    #         # save model every epoch
    #         save_models(self.ckpt_dir, encoder, decoder, suffix=str(epoch+1) + "_epoch")

    #         # run validation and log losses
    #         self.log_losses(self.test(val_dataloader, encoder, decoder, data='val'), iteration=epoch)

    #     logger.info("finished training!")
    def train(self, train_dataloader, val_dataloader, encoder, decoder, optimizer, scheduler):
        epoch_it = trange(self.num_iters)

        # === 动态构建 Watermark AutoEncoder (only for transformer) ===
        wm_ae = None
        if self.model_type == 'transformer':
            if not hasattr(encoder, 'En_wm'):
                raise ValueError("Transformer mode requires encoder.En_wm")
            # Wrap AE using encoder's watermark encoder and a small MLP decoder that
            # maps the encoded watermark vector back to the original watermark size.
            class _WatermarkAE(nn.Module):
                def __init__(self, enc_wm, out_dim):
                    super().__init__()
                    self.enc_wm = enc_wm
                    feat_dim = getattr(enc_wm, 'feat_dim', 512)
                    hidden = max(64, feat_dim // 4)
                    self.dec = nn.Sequential(
                        nn.Linear(feat_dim, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, out_dim),
                        nn.Sigmoid()
                    )
                def forward(self, x):
                    z = self.enc_wm(x)
                    return self.dec(z)

            out_dim = getattr(self.config, 'watermark_size', None)
            if out_dim is None and hasattr(encoder, 'En_wm'):
                out_dim = getattr(encoder.En_wm, 'watermark_size', 128)
            wm_ae = _WatermarkAE(encoder.En_wm, out_dim).to(device)

        for epoch in epoch_it:
            lr = optimizer.param_groups[0]['lr']
            epoch_it.set_description(f"Epoch {epoch}, LR={lr}")
            epoch_loss = defaultdict(list)
            it = tqdm(train_dataloader)
            self.mode = 'train'

            # Set train mode
            encoder.train()
            decoder.train()
            if wm_ae is not None:
                wm_ae.train()

            for cur_iter, (carrier, msg) in enumerate(it):
                carrier, msg = carrier.to(device), msg.to(device)

                # Forward pass
                carrier_reconst, perturbation_S = forward_encoder(encoder, carrier, msg, self.model_type)

                # Decoder expects 4D input [B,C,H,W]; accept 3D external tensors by adding channel dim here
                carrier_for_decoder = carrier_reconst.unsqueeze(1) if carrier_reconst.dim() == 3 else carrier_reconst

                if self.model_type == 'transformer':
                    msg_reconst = decoder(carrier_for_decoder)
                    msg_ae_reconst = wm_ae(msg)
                    # Use the AE's reconstructed vector as the loss target (msg is spectrogram)
                    msg_target = msg_ae_reconst
                else:
                    msg_reconst = decoder(carrier_for_decoder)
                    msg_ae_reconst = None
                    msg_target = msg

                # Compute loss
                print("carrier shape and carrier_reconst shape:")
                print(carrier.shape)
                print(carrier_reconst.shape)
                loss, losses_log = training_step(
                    carrier, carrier_reconst, msg_target, msg_reconst,
                    msg_ae_reconst=msg_ae_reconst,
                    perturbation_S=perturbation_S,
                    lambda_carrier=self.config.lambda_carrier_loss,
                    lambda_msg=self.config.lambda_msg_loss,
                    lambda_ae=getattr(self.config, 'lambda_ae', 1.0),
                    lambda_pert=getattr(self.config, 'lambda_pert', 0.1),
                    epsilon=getattr(self.config, 'epsilon', 0.01),
                    loss_type=self.config.loss_type
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cur_iter % len(train_dataloader) == 0:
                    scheduler.step()

                # Logging...
                if cur_iter % self.print_every == 0:
                    log = f"[{cur_iter}/{len(train_dataloader)}]"
                    for k, v in losses_log.items():
                        log += f", {k}: {v:.4f}"
                    it.set_description(log)
                self.log_losses(losses_log, iteration=cur_iter)
                for k, v in losses_log.items():
                    epoch_loss[k].append(v)

            # ... [rest: logging, saving, validation] ...
    # calc epoch stats
        #         for k,v in list(epoch_loss.items()):
        #             epoch_loss["epoch_" + k] = np.mean(v)
        #             epoch_loss.pop(k)
        #         epoch_loss['lr'] = lr
        #         self.log_losses(epoch_loss, iteration=epoch)

        #         # epoch stats are already logged via self.log_losses
        #         print("=============================================================\n")
        #         # print(f"self.ckpt_dir: {self.ckpt_dir}")
        #         # if(self.ckpt_dir is None):
        #         gl_hparams.load_ckpt = self.ckpt_dir 
        #         print(f"self.ckpt_dir: {self.ckpt_dir}, gl_hparams.load_ckpt: {gl_hparams.load_ckpt}\n")
        #         # save model every epoch
        #         save_models(self.ckpt_dir, encoder, decoder, suffix=str(epoch+1) + "_epoch")

            # Validation
            val_metrics = self.test(val_dataloader, encoder, decoder, data='val')
            self.log_losses(val_metrics, iteration=epoch)


        # def test(self, test_dataloader, encoder, decoder, data='test'):
        #     # logger.debug("eval mode")
        #     self.mode = 'test'

        #     encoder.eval()
        #     decoder.eval()

        #     with torch.no_grad():
        #         avg_carrier_loss, avg_msg_loss = 0, 0
        #         carrier_snr_list = []
        #         msg_snr_list = []

        #         logger.info(f"phase: {'test' if data == 'test' else 'validation'}")
        #         # start of training loop
        #         logger.info(f"start {'testing' if data == 'test' else 'validation'}...")
        #         for carrier, msg in tqdm(test_dataloader):
        #             try:
        #                 assert carrier.shape == msg.shape == spect_audio_shape
        #             except AssertionError:
        #                 print(f"carrier.shape:{carrier.shape},msg.shape:{msg.shape},spect_audio_shape:{spect_audio_shape}")
        #             # feedforward and incur loss
        #             carrier, msg = carrier.to(device), msg.to(device)
        #             # carrier_reconst = encoder(carrier, msg)
        #             carrier_reconst = forward_encoder(encoder, carrier, msg, self.model_type)
        #             if self.model_type == 'transformer':
        #                 msg_reconst = decoder(carrier_reconst, target_size=msg.shape)
        #             else:
        #                 msg_reconst = decoder(carrier_reconst)
        #             # msg_reconst     = decoder(carrier_reconst)
        #             _, losses_log = training_step(carrier, carrier_reconst, msg, msg_reconst, self.config.lambda_carrier_loss, self.config.lambda_msg_loss, self.config.loss_type)
        #             avg_carrier_loss += losses_log['carrier_loss']
        #             avg_msg_loss += losses_log['msg_loss']

        #             # calculate SnR for msg
        #             msg_snr = snr(msg, msg_reconst)
        #             msg_snr_list.append(msg_snr)

        #             # calculate SnR for carrier
        #             carrier_snr = snr(carrier, carrier_reconst)
        #             carrier_snr_list.append(carrier_snr)

        #         logger.info(f"finished {'testing' if data == 'test' else 'validation'}!")
        #         logger.info(f"carrier loss: {avg_carrier_loss/len(test_dataloader)}")
        #         logger.info(f"carrier SnR: {np.mean(carrier_snr_list)}")
        #         logger.info(f"message loss: {avg_msg_loss/len(test_dataloader)}")
        #         logger.info(f"message SnR: {np.mean(msg_snr_list)}")

        #         # log validation/test metrics via Solver logging helper to ensure monotonic steps
        #         self.log_losses({ "carrier_snr": np.mean(carrier_snr_list), "msg_snr": np.mean(msg_snr_list) }, iteration=self.cur_iter)

        #     return {'val epoch carrier loss': avg_carrier_loss/len(test_dataloader),
        #             'val epoch msg loss': avg_msg_loss/len(test_dataloader),
        #             'val epoch carrier SnR': np.mean(carrier_snr_list),
        #             'val epoch msg SnR': np.mean(msg_snr_list)}
    def test(self, test_dataloader, encoder, decoder, data='test'):
        self.mode = 'test'
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            avg_carrier_loss, avg_msg_loss = 0, 0
            carrier_snr_list, msg_snr_list = [], []

            for carrier, msg in tqdm(test_dataloader):
                carrier, msg = carrier.to(device), msg.to(device)

                # Use same forward logic as training
                carrier_reconst, _ = forward_encoder(encoder, carrier, msg, self.model_type)

                # Decoder expects 4D input; add channel dim if needed
                carrier_for_decoder = carrier_reconst.unsqueeze(1) if carrier_reconst.dim() == 3 else carrier_reconst
                msg_reconst = decoder(carrier_for_decoder)
                # For transformer models the decoder often outputs a vector while msg is a spectrogram.
                # Build a small AE (like in train) to map original msg -> vector target matching decoder output.
                if self.model_type == 'transformer' and hasattr(encoder, 'En_wm'):
                    # determine decoder output dim
                    if msg_reconst.dim() == 2:
                        out_dim = msg_reconst.shape[1]
                    elif msg_reconst.dim() == 3:
                        out_dim = msg_reconst.shape[1]
                    elif msg_reconst.dim() == 4:
                        out_dim = msg_reconst.shape[1]
                    else:
                        out_dim = msg_reconst.shape[-1]

                    class _WatermarkAE_local(nn.Module):
                        def __init__(self, enc_wm, out_dim):
                            super().__init__()
                            self.enc_wm = enc_wm
                            feat_dim = getattr(enc_wm, 'feat_dim', None)
                            if feat_dim is None:
                                feat_dim = getattr(enc_wm, 'watermark_size', 512)
                            hidden = max(64, feat_dim // 4)
                            self.dec = nn.Sequential(
                                nn.Linear(feat_dim, hidden),
                                nn.ReLU(),
                                nn.Linear(hidden, out_dim),
                                nn.Sigmoid()
                            )
                        def forward(self, x):
                            z = self.enc_wm(x)
                            return self.dec(z)

                    wm_ae = _WatermarkAE_local(encoder.En_wm, out_dim).to(device)
                    msg_target = wm_ae(msg)
                else:
                    msg_target = msg

                # Use training_step to compute losses (it normalizes dims internally)
                _, losses_log = training_step(
                    carrier, carrier_reconst, msg_target, msg_reconst,
                    lambda_carrier=self.config.lambda_carrier_loss,
                    lambda_msg=self.config.lambda_msg_loss,
                    loss_type=self.config.loss_type
                )

                avg_carrier_loss += losses_log['carrier_loss']
                avg_msg_loss += losses_log['msg_loss']

                # SNR calculation: for transformer, compare vectors; otherwise compare spectrograms
                if self.model_type == 'transformer' and (hasattr(encoder, 'En_wm')):
                    # ensure vectors
                    orig_vec = msg_target.detach().cpu()
                    recon_vec = msg_reconst.detach().cpu()
                    # flatten to [B, L]
                    orig_v = orig_vec.view(orig_vec.shape[0], -1)
                    recon_v = recon_vec.view(recon_vec.shape[0], -1)
                    # compute simple vector SNR per-sample then mean
                    eps = 1e-8
                    rms1 = torch.sqrt(torch.mean(orig_v ** 2, dim=1) + eps)
                    rms2 = torch.sqrt(torch.mean((orig_v - recon_v) ** 2, dim=1) + eps)
                    msg_snr_vals = 10 * torch.log10((rms1 / (rms2 + eps)) ** 2)
                    msg_snr = float(msg_snr_vals.mean().item())
                else:
                    msg_ref = msg.unsqueeze(1) if msg.dim() == 3 else msg
                    msg_pred = msg_reconst.unsqueeze(1) if msg_reconst.dim() == 3 else msg_reconst
                    msg_snr = snr(msg_ref, msg_pred)

                carrier_ref = carrier.unsqueeze(1) if carrier.dim() == 3 else carrier
                carrier_pred = carrier_reconst.unsqueeze(1) if carrier_reconst.dim() == 3 else carrier_reconst
                carrier_snr = snr(carrier_ref, carrier_pred)

                msg_snr_list.append(msg_snr)
                carrier_snr_list.append(carrier_snr)

            # ... logging ...

            logger.info(f"finished {'testing' if data == 'test' else 'validation'}!")
            logger.info(f"carrier loss: {avg_carrier_loss/len(test_dataloader)}")
            logger.info(f"carrier SnR: {np.mean(carrier_snr_list)}")
            logger.info(f"message loss: {avg_msg_loss/len(test_dataloader)}")
            logger.info(f"message SnR: {np.mean(msg_snr_list)}")

            return {
                'val epoch carrier loss': avg_carrier_loss / len(test_dataloader),
                'val epoch msg loss': avg_msg_loss / len(test_dataloader),
                'val epoch carrier SnR': np.mean(carrier_snr_list),
                'val epoch msg SnR': np.mean(msg_snr_list)
            }