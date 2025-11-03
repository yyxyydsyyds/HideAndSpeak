import argparse
from os.path import join

import lightning as L
import torch
import torch.nn as nn
import torch.nn.init as init
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from model import FullEncoder
from dataset import TimitDataset
from dataset import train_single_dataloader, val_single_dataloader, test_single_dataloader,unet_train_single_dataloader,unet_val_single_dataloader
from dataset import train_dataloader, val_dataloader, test_dataloader
from model import get_models, load_models


from dataset import TimitDataset, TimitSingleDataset
from hparams import AUDIO_LEN
from model import CarrierDecoder, Encoder, MsgDecoder
from config import get_hparams

from solver import Solver
from config import gl_hparams 
from model import Discriminator 
from config import device

from portable_udh import get_hiding_unet, get_reveal_net, count_params, UnetGenerator, RevealNet
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

from collections import defaultdict
from tqdm import tqdm, trange
import torch.nn.functional as F
import numpy as np

from solver import snr
from config import DEBUG, debug_print

try:
    from comet_ml import Experiment as CometExperiment
    EXTERNAL_LOGGING_AVAILABLE = True
except Exception as e:
    EXTERNAL_LOGGING_AVAILABLE = False

torch.manual_seed(0)



lambda_carrier_values = [0.1, 0.5, 1.0, 2.0, 5.0]  
lambda_msg_values = [0.1, 0.5, 1.0, 2.0, 5.0]

class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass



# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.0) 
        m.bias.data.fill_(0)

def unet_training_step(carrier: torch.Tensor, carrier_reconst: torch.Tensor, msg: torch.Tensor, msg_reconst: torch.Tensor, lambda_carrier, lambda_msg, loss_type):
    """训练步骤，计算损失"""
    losses_log = defaultdict(int)
    carrier, msg = carrier.to(device), msg.to(device)
    
    loss_fn = F.mse_loss if loss_type == 'mse' else F.l1_loss
    carrier_loss = loss_fn(carrier_reconst, carrier)
    msg_loss = loss_fn(msg_reconst, msg)
    
    losses_log['carrier_loss'] = carrier_loss.item()
    losses_log['msg_loss'] = msg_loss.item()
    
    total_loss = lambda_carrier * carrier_loss + lambda_msg * msg_loss
    return total_loss, losses_log

# def unet_snr(orig: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
#     """计算信噪比"""
#     N = orig.shape[-1] * orig.shape[-2]
#     orig, recon = orig.cpu(), recon.cpu()
#     rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
#     rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
#     snr = 10 * torch.log10((rms1 / rms2) ** 2)
#     return snr


def runUnet(hparams):
    logger.info(f"hparams.model_type:{hparams.model_type}")
    hparams.ngpu = torch.cuda.device_count()

    num_downs=5

    norm_layer_str = hparams.norm 
    Hnet = UnetGenerator(input_nc=1+1, output_nc=1, num_downs=num_downs, norm_layer=norm_layer_str, output_function=nn.Sigmoid)
    Rnet = RevealNet(input_nc=1, output_nc=1, nhf=64, norm_layer=norm_layer_str, output_function=nn.Sigmoid)

    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    if torch.cuda.is_available():
        # Hnet = torch.nn.DataParallel(Hnet).cuda()
        # Rnet = torch.nn.DataParallel(Rnet).cuda()
        Hnet = Hnet.cuda()
        Rnet = Rnet.cuda()

    if hparams.loss_type == 'abs':
        criterion = nn.L1Loss().cuda() if torch.cuda.is_available() else nn.L1Loss()
    if hparams.loss_type == 'mse':
        criterion = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

    # writer = SummaryWriter(log_dir='runs/' + hparams.run_dir) 
    params = list(Hnet.parameters())+list(Rnet.parameters())
    optimizer = optim.Adam(params, lr=hparams.lr, betas=(0.5, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)    

    train_loader = unet_train_single_dataloader(hparams.train_path, hparams.message_file, hparams.batch_size, num_workers=0)
    val_loader = unet_val_single_dataloader(hparams.val_path, hparams.message_file, hparams.batch_size, num_workers=0)

    # TensorBoard
    if not hparams.debug if hasattr(hparams, 'debug') else True:
        writer = SummaryWriter(log_dir='log/' + 'hideandspeak_unet_' )

    best_loss = float('inf')
    for epoch in range(hparams.num_iters):
        logger.info(f"Starting epoch {epoch}")
        
        # 训练模式
        Hnet.train()

        Rnet.train()
        
        epoch_losses = defaultdict(list)
        
        # 创建训练进度条
        train_desc = f"Epoch {epoch+1}/{hparams.num_iters} [TRAIN]"
        train_bar = tqdm(train_loader, desc=train_desc, leave=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Epoch {epoch+1}/{hparams.num_iters}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        for batch_idx, (carrier, msg) in enumerate(train_bar):
            try:
                carrier, msg = carrier.to(device), msg.to(device)
                #todo
                original_h, original_w = carrier.shape[2], carrier.shape[3]  # 保存原始尺寸


                # 前向传播
                H_input = torch.cat([carrier, msg], dim=1)  
                container =  Hnet(H_input) 
                
                # todo
                if container.shape[2] != original_h or container.shape[3] != original_w:
                    pad_h = original_h - container.shape[2]
                    pad_w = original_w - container.shape[3]
                    container = F.pad(container, (0, pad_w, 0, pad_h))
                
                container =  container + carrier

                revealed_msg = Rnet(container)
                
                # todo
                if revealed_msg.shape[2] != original_h or revealed_msg.shape[3] != original_w:
                    pad_h = original_h - revealed_msg.shape[2]
                    pad_w = original_w - revealed_msg.shape[3]
                    revealed_msg = F.pad(revealed_msg, (0, pad_w, 0, pad_h))

                # 计算损失
                total_loss, losses_log = unet_training_step(
                    carrier, container, msg, revealed_msg,
                    hparams.lambda_carrier_loss, hparams.lambda_msg_loss, 
                    hparams.loss_type
                )

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
               
                # 记录损失
                for k, v in losses_log.items():
                    epoch_losses[k].append(v)
                
                # 更新进度条
                train_bar.set_postfix({
                    'carrier_loss': f"{losses_log['carrier_loss']:.4f}",
                    'msg_loss': f"{losses_log['msg_loss']:.4f}"
                })
               
                # # 每10个batch打印一次
                # if batch_idx % 10 == 0:
                #     print(f"Batch {batch_idx}/{len(train_loader)} completed")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        # 验证
        Hnet.eval()
        Rnet.eval()
        
        val_losses = defaultdict(list)
        carrier_snr_list = []
        msg_snr_list = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
            for batch_idx, (carrier, msg) in enumerate(val_bar):
                try:
                    carrier, msg = carrier.to(device), msg.to(device)
                    #todo
                    original_h, original_w = carrier.shape[2], carrier.shape[3]  # 保存原始尺寸
                    
                    # 前向传播
                    H_input = torch.cat([carrier, msg], dim=1)
                    container = Hnet(H_input) 

                    # todo Pad容器回原始尺寸
                    if container.shape[2] != original_h or container.shape[3] != original_w:
                        pad_h = original_h - container.shape[2]
                        pad_w = original_w - container.shape[3]
                        container = F.pad(container, (0, pad_w, 0, pad_h))

                    container =  container + carrier
                    revealed_msg = Rnet(container)
                    
                    #todo Pad揭示的消息回原始尺寸
                    if revealed_msg.shape[2] != original_h or revealed_msg.shape[3] != original_w:
                        pad_h = original_h - revealed_msg.shape[2]
                        pad_w = original_w - revealed_msg.shape[3]
                        revealed_msg = F.pad(revealed_msg, (0, pad_w, 0, pad_h))

                    # 计算损失
                    _, losses_log = unet_training_step(
                        carrier, container, msg, revealed_msg,
                        hparams.lambda_carrier_loss, hparams.lambda_msg_loss,
                        hparams.loss_type
                    )
                    
                    # 记录损失
                    for k, v in losses_log.items():
                        val_losses[k].append(v)
                    
                    # 计算SNR
                    try:
                        carrier_snr = snr(carrier, container)
                        msg_snr = snr(msg, revealed_msg)
                        carrier_snr_list.append(carrier_snr.item())
                        msg_snr_list.append(msg_snr.item())
                    except:
                        pass  # 如果SNR计算失败，跳过

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # 计算平均损失
        avg_train_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        avg_val_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        # 学习率调度
        val_total_loss = avg_val_losses['carrier_loss'] + avg_val_losses['msg_loss']
        scheduler.step(val_total_loss)
        
        # 打印统计信息
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train - Carrier Loss: {avg_train_losses['carrier_loss']:.4f}, Msg Loss: {avg_train_losses['msg_loss']:.4f}")
        logger.info(f"Val - Carrier Loss: {avg_val_losses['carrier_loss']:.4f}, Msg Loss: {avg_val_losses['msg_loss']:.4f}")
        if carrier_snr_list:
            logger.info(f"Val - Carrier SNR: {np.mean(carrier_snr_list):.2f}dB, Msg SNR: {np.mean(msg_snr_list):.2f}dB")
        
        # TensorBoard记录
        if not hparams.debug if hasattr(hparams, 'debug') else True:
            writer.add_scalar('train/carrier_loss', avg_train_losses['carrier_loss'], epoch)
            writer.add_scalar('train/msg_loss', avg_train_losses['msg_loss'], epoch)
            writer.add_scalar('val/carrier_loss', avg_val_losses['carrier_loss'], epoch)
            writer.add_scalar('val/msg_loss', avg_val_losses['msg_loss'], epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            if carrier_snr_list:
                writer.add_scalar('val/carrier_snr', np.mean(carrier_snr_list), epoch)
                writer.add_scalar('val/msg_snr', np.mean(msg_snr_list), epoch)
        
        # 保存最佳模型（仅在用户显式启用保存时进行）
        save_every = getattr(hparams, 'save_model_every', None)
        if save_every and save_every > 0:
            if val_total_loss < best_loss:
                best_loss = val_total_loss
                torch.save({
                    'epoch': epoch,
                    'Hnet_state_dict': Hnet.state_dict(),
                    'Rnet_state_dict': Rnet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, f'best_unet_model_epoch_{epoch}.pth')
                logger.info(f"Saved best model at epoch {epoch}")

            # 定期保存检查点（按 save_model_every 指定的频率）
            if epoch % save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'Hnet_state_dict': Hnet.state_dict(),
                    'Rnet_state_dict': Rnet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, f'unet_checkpoint_epoch_{epoch}.pth')
    
    # 关闭TensorBoard
    if not hparams.debug if hasattr(hparams, 'debug') else True:
        writer.close()
    
    logger.info("UNet training completed!")

def run(hparams):
    torch.set_num_threads(1000)
    
    solver = Solver(hparams)
    encoder, decoder, optimizer, scheduler = get_models(hparams)
    # if torch.cuda.device_count() > 1:
    #     logger.info(f"Using {torch.cuda.device_count()} GPUs!")
    #     encoder = nn.DataParallel(encoder)
    #     decoder = nn.DataParallel(decoder)
        
    logger.info(f"hparams.mode:{hparams.mode}")
    if hparams.mode == 'train':
        if hparams.single is True:
            # if hparams.model_type =='unet':
            #     train_loader = unet_train_single_dataloader(hparams.train_path, hparams.message_file, hparams.batch_size, hparams.num_workers)
            #     val_loader   = unet_val_single_dataloader(hparams.val_path, hparams.message_file, hparams.batch_size, hparams.num_workers)

            # else:
            train_loader = train_single_dataloader(hparams.train_path, hparams.message_file, hparams.batch_size, hparams.num_workers)
            val_loader   = val_single_dataloader(hparams.val_path, hparams.message_file, hparams.batch_size, hparams.num_workers)
        else:
            train_loader = train_dataloader(hparams.train_path, hparams.batch_size, hparams.num_workers)
            val_loader   = val_dataloader(hparams.val_path, hparams.batch_size, hparams.num_workers)

        logger.info(f"loaded train ({len(train_loader)}), val ({len(val_loader)})")

        if hparams.model_type =='GAN':
            logger.info(f"hparams.model_type is {hparams.model_type}")
            discriminator = Discriminator().to(device)
            optimizer_G = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=hparams.lr)
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hparams.lr)
            solver.train_gan(train_loader, val_loader, encoder, decoder, discriminator, optimizer_G, optimizer_D, scheduler)
        # elif hparams.model_type =='unet':
        #     logger.info(f"hparams.model_type is {hparams.model_type}")
        #     solver.train(train_loader, val_loader, encoder, decoder, optimizer, scheduler)

        else:    
            solver.train(train_loader, val_loader, encoder, decoder, optimizer, scheduler)
    elif hparams.mode == 'test':
        if hparams.single is True:
            test_loader = test_single_dataloader(hparams.test_path, hparams.message_file, hparams.batch_size)
        else:
            test_loader = test_dataloader(hparams.test_path, hparams.batch_size)

        logger.info(f"loaded test ({len(test_loader)})")

        solver.test(test_loader, encoder, decoder)
    elif hparams.mode == 'sample':
        pass
        # solver.eval_mode()
        # solver.sample_examples()

freeze_num_list=[1, 5, 10, 20]

def run_all(hparams):

    if gl_hparams.model_type =='unet':
        runUnet(gl_hparams)
    else:
        run(gl_hparams)
    logger.info(f"Training complete!lr:{gl_hparams.lr}")
    torch.cuda.empty_cache()
    # Experiment will handle wandb lifecycle; nothing to do here
    logger.info("Training complete!")
    logger.info(f"lr: {gl_hparams.lr}, batch_size: {gl_hparams.batch_size}, mode: {gl_hparams.mode}, single: {gl_hparams.single}, lambda_carrier: {gl_hparams.lambda_carrier_loss}, lambda_msg: {gl_hparams.lambda_msg_loss}, freeze_num: {gl_hparams.freeze_num}, model_type: {gl_hparams.model_type}, lr: {gl_hparams.lr}1")

lambda_msg_values = {0.2,0.5,1,2,5,10}

def main():
    global gl_hparams
    if(gl_hparams==None): 
        gl_hparams = get_hparams()  

    
    # for gl_hparams.lambda_msg_loss in lambda_msg_values:
    #     gl_hparams.lambda_carrier_loss = 1
    run_all(gl_hparams)

if __name__ == '__main__':
    main()