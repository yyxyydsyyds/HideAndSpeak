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
import warnings



if(gl_hparams==None): 
    gl_hparams = get_hparams()

spect_audio_shape = (gl_hparams.batch_size, 1, 129, 378)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def forward_encoder(encoder, carrier, msg, model_type):
    if model_type == 'unet':
        x = torch.cat([carrier, msg], dim=1)
        return encoder(x)
    else:  # original
        return encoder(carrier, msg)

# def unet_snr_per_channel(orig: torch.Tensor, recon: torch.Tensor) -> dict:
#     """
#     分通道计算SNR - 这才能体现三通道的意义
    
#     Args:
#         orig: 原始数据 (batch, 3, H, W)
#         recon: 重建数据 (batch, 3, H, W)
    
#     Returns:
#         dict: {
#             'magnitude_snr': float,  # 幅度通道SNR
#             'phase_snr': float,      # 相位通道SNR
#             'power_snr': float,      # 功率通道SNR
#             'mean_snr': float        # 三通道平均SNR
#         }
#     """
#     assert orig.shape == recon.shape, f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
#     assert orig.shape[1] == 3, f"Expected 3 channels, got {orig.shape[1]}"
    
#     orig, recon = orig.cpu(), recon.cpu()
    
#     channel_names = ['magnitude', 'phase', 'power']
#     results = {}
    
#     # 对每个通道分别计算SNR
#     for i, name in enumerate(channel_names):
#         # 提取单个通道 (batch, 1, H, W)
#         orig_channel = orig[:, i:i+1, :, :]
#         recon_channel = recon[:, i:i+1, :, :]
        
#         # 计算该通道的SNR（与Normal模式相同的计算方式）
#         N = orig_channel.shape[1] * orig_channel.shape[2] * orig_channel.shape[3]
#         rms_signal = ((torch.sum(orig_channel ** 2) / N) ** 0.5)
#         rms_noise = ((torch.sum((orig_channel - recon_channel) ** 2) / N) ** 0.5)
#         snr = 10 * torch.log10((rms_signal / rms_noise) ** 2)
        
#         results[f'{name}_snr'] = snr.item()
    
#     # 计算平均SNR（可选）
#     results['mean_snr'] = np.mean([results['magnitude_snr'], 
#                                    results['phase_snr'], 
#                                    results['power_snr']])
    
#     return results


# def unet_snr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
#     """
#     计算UNet模式下的信噪比
    
#     修正：正确处理多通道数据 (batch, 3, H, W)
    
#     Args:
#         orig: 原始数据 (batch, 3, 129, 378)
#         recon: 重建数据 (batch, 3, 129, 378)
    
#     Returns:
#         SNR值（dB）
#     """
#     assert orig.shape == recon.shape, f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
    
#     orig, recon = orig.cpu(), recon.cpu()
    
#     # 计算所有需要平均的元素数量：通道数 * 高度 * 宽度
#     # 对于 (batch, 3, 129, 378)，N = 3 * 129 * 378 = 146,286
#     N = orig.shape[1] * orig.shape[2] * orig.shape[3]  # C * H * W
    
#     # 对每个batch样本分别计算，然后平均
#     batch_size = orig.shape[0]
#     snr_list = []
    
#     for i in range(batch_size):
#         orig_sample = orig[i]  # (3, 129, 378)
#         recon_sample = recon[i]  # (3, 129, 378)
        
#         # 计算RMS
#         rms_signal = ((torch.sum(orig_sample ** 2) / N) ** 0.5)
#         rms_noise = ((torch.sum((orig_sample - recon_sample) ** 2) / N) ** 0.5)
        
#         # 计算SNR
#         snr_value = 10 * torch.log10((rms_signal / rms_noise) ** 2)
#         snr_list.append(snr_value)
    
#     # 返回batch的平均SNR
#     return torch.mean(torch.stack(snr_list))


# def unet_snr_v2(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
#     """
#     计算UNet模式下的信噪比（与Normal模式计算逻辑一致）
    
#     这个版本与solver.py中的snr函数保持相同的计算逻辑
    
#     Args:
#         orig: 原始数据 (batch, 1, 129, 378)
#         recon: 重建数据 (batch, 3, 129, 378)
    
#     Returns:
#         SNR值（dB）
#     """
#     assert orig.shape == recon.shape, f"Shape mismatch: orig {orig.shape} vs recon {recon.shape}"
    
#     orig, recon = orig.cpu(), recon.cpu()
    
#     # 计算总的元素数量（包括所有通道和空间维度，不包括batch）
#     # 对于 (batch, 3, 129, 378): N = 3 * 129 * 378 = 146,286
#     N = orig.shape[1] * orig.shape[2] * orig.shape[3]
    
#     # 对整个batch计算（与Normal模式一致）
#     rms_signal = ((torch.sum(orig ** 2) / N) ** 0.5)
#     rms_noise = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
    
#     snr = 10 * torch.log10((rms_signal / rms_noise) ** 2)
    
#     return snr


def snr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    try:
        assert orig.shape == recon.shape == spect_audio_shape
    except AssertionError:
        print(f"orig.shape:{orig.shape},recon.shape:{recon.shape},spect_audio_shape:{spect_audio_shape}")
    N = orig.shape[-1] * orig.shape[-2]
    orig, recon = orig.cpu(), recon.cpu()
    rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
    rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
    # print(rms1, rms2)
    snr = 10 * torch.log10((rms1 / rms2) ** 2)
    return snr

def training_step(carrier: torch.Tensor, carrier_reconst: torch.Tensor, msg: torch.Tensor, msg_reconst: torch.Tensor, lambda_carrier, lambda_msg, loss_type) -> Tuple[torch.Tensor, defaultdict]:
    try:
        assert carrier.shape == carrier_reconst.shape == msg.shape == msg_reconst.shape == spect_audio_shape
    except AssertionError:
        print(f"carrier.shape:{carrier.shape},carrier_reconst.shape:{carrier_reconst.shape},msg.shape:{msg.shape},msg_reconst.shape:{msg_reconst.shape},spect_audio_shape:{spect_audio_shape}")
    losses_log = defaultdict(int)
    carrier, msg = carrier.to(device), msg.to(device)
    loss = F.mse_loss if loss_type == 'mse' else F.l1_loss
    carrier_loss = loss(carrier_reconst, carrier)
    msg_loss = loss(msg_reconst, msg)
    losses_log['carrier_loss'] = carrier_loss.item()
    losses_log['msg_loss'] = msg_loss.item()
    loss = lambda_carrier * carrier_loss + lambda_msg * msg_loss

    return loss, losses_log

def save_models(ckpt_dir, encoder, decoder, suffix=''):
    logger.info(f"saving model to: {ckpt_dir}\n==> suffix: {suffix}")
    makedirs(join(ckpt_dir, suffix), exist_ok=True)
    torch.save(encoder.state_dict(), join(ckpt_dir, suffix, "encoder.ckpt"))
    torch.save(decoder.state_dict(), join(ckpt_dir, suffix, "decoder.ckpt"))

class Solver(object):
    def __init__(self, config):
        self.config = config

        # training config
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

        torch.autograd.set_detect_anomaly(True)

        
        # logging
        logger.add(join(self.run_dir, "stdout.log"))
        # no solver-level wandb initialization; Experiment handles external logging when use_wandb is set
        # global step counter for external loggers (ensures monotonic steps)
        self.global_step = 0

    def log_losses(self, losses, iteration=None):
        if iteration is None:
            iteration = self.cur_iter
        # Ensure strictly monotonic steps by incrementing the global step first
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        # advance global step so each call uses a larger step than previous
        try:
            self.global_step += 1
            step_for_logging = self.global_step
        except Exception:
            step_for_logging = iteration

        # Prefer Experiment-level logging which handles tensorboard and (optionally) wandb consistently
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


    def train(self, train_dataloader, val_dataloader, encoder, decoder, optimizer, scheduler):
        # start of training loop
        # logger.info("start training...")
        epoch_it = trange(self.num_iters)

        for epoch in epoch_it:
            lr = optimizer.param_groups[0]['lr']
            epoch_it.set_description(f"Epoch {epoch}, LR={lr}")
            epoch_loss = defaultdict(list)
            it = tqdm(train_dataloader)
            logger.debug("train mode")
            self.mode = 'train'
            
            encoder.train()
            decoder.train()

            # inner epoch loop
            for cur_iter, (carrier, msg) in enumerate(it):
                try:
                    assert carrier.shape == msg.shape == spect_audio_shape
                except AssertionError:
                    print(f"carrier.shape:{carrier.shape},msg.shape:{msg.shape},spect_audio_shape:{spect_audio_shape}")
                # feedforward and suffer loss
                carrier, msg = carrier.to(device), msg.to(device)
                # carrier_reconst = encoder(carrier, msg)
                carrier_reconst = forward_encoder(encoder, carrier, msg, self.model_type)
                msg_reconst     = decoder(carrier_reconst)
                loss, losses_log = training_step(carrier, carrier_reconst, msg, msg_reconst, self.config.lambda_carrier_loss, self.config.lambda_msg_loss, self.config.loss_type)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cur_iter % len(train_dataloader) == 0:
                    scheduler.step()

                # log stuff
                if cur_iter % self.print_every == 0:
                    log = f"[{cur_iter}/{len(train_dataloader)}]"
                    for loss_name, loss_value in losses_log.items():
                        log += f", {loss_name}: {loss_value:.4f}"
                    it.set_description(log)
                self.log_losses(losses_log, iteration=cur_iter)

                # log epoch losses
                for k,v in losses_log.items():
                    epoch_loss[k].append(v)

            # calc epoch stats
            for k,v in list(epoch_loss.items()):
                epoch_loss["epoch_" + k] = np.mean(v)
                epoch_loss.pop(k)
            epoch_loss['lr'] = lr
            self.log_losses(epoch_loss, iteration=epoch)

            # epoch stats are already logged via self.log_losses

            # save model every epoch
            save_models(self.ckpt_dir, encoder, decoder, suffix=str(epoch+1) + "_epoch")

            # run validation and log losses
            self.log_losses(self.test(val_dataloader, encoder, decoder, data='val'), iteration=epoch)

        logger.info("finished training!")

    def test(self, test_dataloader, encoder, decoder, data='test'):
        # logger.debug("eval mode")
        self.mode = 'test'

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            avg_carrier_loss, avg_msg_loss = 0, 0
            carrier_snr_list = []
            msg_snr_list = []

            logger.info(f"phase: {'test' if data == 'test' else 'validation'}")
            # start of training loop
            logger.info(f"start {'testing' if data == 'test' else 'validation'}...")
            for carrier, msg in tqdm(test_dataloader):
                try:
                    assert carrier.shape == msg.shape == spect_audio_shape
                except AssertionError:
                    print(f"carrier.shape:{carrier.shape},msg.shape:{msg.shape},spect_audio_shape:{spect_audio_shape}")
                # feedforward and incur loss
                carrier, msg = carrier.to(device), msg.to(device)
                carrier_reconst = encoder(carrier, msg)
                msg_reconst     = decoder(carrier_reconst)
                _, losses_log = training_step(carrier, carrier_reconst, msg, msg_reconst, self.config.lambda_carrier_loss, self.config.lambda_msg_loss, self.config.loss_type)
                avg_carrier_loss += losses_log['carrier_loss']
                avg_msg_loss += losses_log['msg_loss']

                # calculate SnR for msg
                msg_snr = snr(msg, msg_reconst)
                msg_snr_list.append(msg_snr)

                # calculate SnR for carrier
                carrier_snr = snr(carrier, carrier_reconst)
                carrier_snr_list.append(carrier_snr)

            logger.info(f"finished {'testing' if data == 'test' else 'validation'}!")
            logger.info(f"carrier loss: {avg_carrier_loss/len(test_dataloader)}")
            logger.info(f"carrier SnR: {np.mean(carrier_snr_list)}")
            logger.info(f"message loss: {avg_msg_loss/len(test_dataloader)}")
            logger.info(f"message SnR: {np.mean(msg_snr_list)}")

            # log validation/test metrics via Solver logging helper to ensure monotonic steps
            self.log_losses({ "carrier_snr": np.mean(carrier_snr_list), "msg_snr": np.mean(msg_snr_list) }, iteration=self.cur_iter)

        return {'val epoch carrier loss': avg_carrier_loss/len(test_dataloader),
                'val epoch msg loss': avg_msg_loss/len(test_dataloader),
                'val epoch carrier SnR': np.mean(carrier_snr_list),
                'val epoch msg SnR': np.mean(msg_snr_list)}
