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

if(gl_hparams==None): 
    gl_hparams = get_hparams()

spect_audio_shape = (gl_hparams.batch_size, 1, 129, 378)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def snr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    try:
        assert orig.shape == recon.shape == spect_audio_shape
    except AssertionError:
        print(f"orig.shape:{orig.shape},recon.shape:{recon.shape},spect_audio_shape:{spect_audio_shape}")
    N = orig.shape[-1] * orig.shape[-2]
    orig, recon = orig.cpu(), recon.cpu()
    rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
    rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
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

# # 定义判别器
# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super().__init__()
#         # 卷积层序列
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 64, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             # 更多层...
#             nn.Conv2d(512, 1, 4, 1, 0),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         return self.conv(x).view(-1, 1).squeeze(1)


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
        self.experiment    = Experiment(config.run_dir, use_comet=False, use_wandb=True)
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

        self.create_dirs()
        torch.manual_seed(10)

        self.stft = STFT(N_FFT, HOP_LENGTH)
        self.stft.num_samples = self.num_samples

        torch.autograd.set_detect_anomaly(True)

        
        # logging
        logger.add(join(self.run_dir, "stdout.log"))

    def log_losses(self, losses, iteration=None):
        if iteration is None:
            iteration = self.cur_iter

        self.experiment.log_metric(losses, step=iteration)

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
            if self.experiment.wandb_exp:
                import wandb
                wandb.log({
                    "lambda_carrier": self.config.lambda_carrier_loss,
                    "lambda_msg": self.config.lambda_msg_loss
                }, step=epoch)

            for k, v in list(epoch_loss.items()):
                epoch_loss["epoch_" + k] = np.mean(v)
                epoch_loss.pop(k)
            epoch_loss['lr'] = lr
            self.log_losses(epoch_loss, iteration=epoch)

            # 保存模型
            save_models(self.ckpt_dir, encoder, decoder, suffix=str(epoch+1) + "_epoch")

            # 验证集评估
            self.log_losses(self.test(val_dataloader, encoder, decoder, data='val'), iteration=epoch)


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
                carrier_reconst = encoder(carrier, msg)
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

        return {'val epoch carrier loss': avg_carrier_loss/len(test_dataloader),
                'val epoch msg loss': avg_msg_loss/len(test_dataloader),
                'val epoch carrier SnR': np.mean(carrier_snr_list),
                'val epoch msg SnR': np.mean(msg_snr_list)}
