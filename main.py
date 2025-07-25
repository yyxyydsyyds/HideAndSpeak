import argparse
from os.path import join

import lightning as L
import torch
import torch.nn as nn
from loguru import logger
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataset import TimitDataset, TimitSingleDataset
from hparams import AUDIO_LEN
from model import CarrierDecoder, Encoder, MsgDecoder
from config import get_hparams

from solver import Solver
from config import gl_hparams 
from model import Discriminator 

try:
    from comet_ml import Experiment as CometExperiment
    import wandb
    EXTERNAL_LOGGING_AVAILABLE = True
except Exception as e:
    EXTERNAL_LOGGING_AVAILABLE = False

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#gl_hparams=None

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


class FullEncoder(nn.Module):
    def __init__(self, block_type, enc_n_layers, dec_um_conv_dim, dec_c_n_layers) -> None:
        super().__init__()
        self.encoder_first = Encoder(block_type=block_type,
                                n_layers=enc_n_layers)

        self.encoder_second = CarrierDecoder(conv_dim=dec_um_conv_dim,
                                        block_type=block_type,
                                        n_layers=dec_c_n_layers)
        
    def forward(self, carrier, msg):
        msg_emb = self.encoder_first(msg)
        msg_merged = torch.cat((msg, msg_emb), dim=1)
        msg_u = self.encoder_second(msg_merged)
        carrier_reconst = carrier + msg_u

        return carrier_reconst
    

def train_dataloader(train_path, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    train_dataset = TimitDataset(train_path,
                                n_pairs     = 4608,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size  = batch_size,
                                  shuffle     = True,
                                  num_workers = num_workers)
    
    return train_dataloader

def val_dataloader(val_path, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    val_dataset = TimitDataset(val_path,
                               n_pairs     = 832,
                               trim_start  = trim_start,
                               num_samples = num_samples,
                               test        = True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size  = batch_size,
                                shuffle     = False,
                                num_workers = num_workers)
    
    return val_dataloader

def test_dataloader(test_path, batch_size):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    test_dataset = TimitDataset(test_path,
                                n_pairs     = 832,
                                trim_start  = trim_start,
                                num_samples = num_samples,
                                test        = True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size  = batch_size,
                                 shuffle     = False,
                                 num_workers = 0)
    
    return test_dataloader

def train_single_dataloader(train_path, message_file, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    train_dataset = TimitSingleDataset(train_path,
                                 message_file,
                                n_pairs     = 4608,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size  = batch_size,
                                  shuffle     = True,
                                  num_workers = num_workers)
    
    return train_dataloader

def val_single_dataloader(val_path, message_file, batch_size, num_workers):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    val_dataset = TimitSingleDataset(val_path,
                                     message_file,
                               n_pairs     = 832,
                               # n_pairs=32,
                               trim_start  = trim_start,
                               num_samples = num_samples)
    val_dataloader = DataLoader(val_dataset,
                                batch_size  = batch_size,
                                shuffle     = False,
                                num_workers = num_workers)
    
    return val_dataloader

def test_single_dataloader(test_path, message_file, batch_size):
    trim_start  = int(0.6*16000)
    num_samples = AUDIO_LEN * 16000
    test_dataset = TimitDataset(test_path,
                                message_file,
                                n_pairs     = 832,
                                # n_pairs=32,
                                trim_start  = trim_start,
                                num_samples = num_samples)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size  = batch_size,
                                 shuffle     = False,
                                 num_workers = 0)
    
    return test_dataloader


def load_models(encoder, decoder, ckpt_dir):
    encoder.load_state_dict(torch.load(join(ckpt_dir, "encoder.ckpt")))
    decoder.load_state_dict(torch.load(join(ckpt_dir, "decoder.ckpt")))
    logger.info("loader models")

def get_models(hparams):
        dec_m_conv_dim  = 1
        dec_um_conv_dim = 1 + 64

        encoder = FullEncoder(hparams.block_type, hparams.enc_n_layers, dec_um_conv_dim, hparams.dec_c_n_layers).to(device)

        decoder =  MsgDecoder(conv_dim=dec_m_conv_dim, block_type=hparams.block_type).to(device)
        
        params = list(encoder.parameters()) + list(decoder.parameters())
        
        optimizer = torch.optim.Adam(params, lr=hparams.lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        if hparams.load_ckpt:
            load_models(encoder, decoder, hparams.load_ckpt)

        logger.debug(encoder)
        logger.debug(decoder)

        return encoder, decoder, optimizer, scheduler

def run(hparams):
    torch.set_num_threads(1000)
    solver = Solver(hparams)
    
    encoder, decoder, optimizer, scheduler = get_models(hparams)
    logger.info(f"hparams.mode:{hparams.mode}")
    if hparams.mode == 'train':
        if hparams.single is True:
            train_loader = train_single_dataloader(hparams.train_path, hparams.message_file, hparams.batch_size, hparams.num_workers)
            val_loader   = val_single_dataloader(hparams.val_path, hparams.message_file, hparams.batch_size, hparams.num_workers)
        else:
            train_loader = train_dataloader(hparams.train_path, hparams.batch_size, hparams.num_workers)
            val_loader   = val_dataloader(hparams.val_path, hparams.batch_size, hparams.num_workers)

        logger.info(f"loaded train ({len(train_loader)}), val ({len(val_loader)})")

        if hparams.GAN_model is True:
            discriminator = Discriminator().to(device)
            optimizer_G = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=hparams.lr)
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=hparams.lr)
            solver.train_gan(train_loader, val_loader, encoder, decoder, discriminator, optimizer_G, optimizer_D, scheduler)
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

def main():
    global gl_hparams
    if(gl_hparams==None): 
        gl_hparams = get_hparams()  

    # gl_hparams.lambda_carrier_loss = 1
    for i in range(3):
        # gl_hparams.lambda_msg_loss = i
        # gl_hparams.lr = lr
        gl_hparams.freeze_num = freeze_num_list[i]
        
        run(gl_hparams)
        logger.info(f"Training complete!lr:{gl_hparams.lr}")
        torch.cuda.empty_cache()
        wandb.finish()
        logger.info("Training complete!")
    logger.info(f"lr: {gl_hparams.lr}, batch_size: {gl_hparams.batch_size}, mode: {gl_hparams.mode}, single: {gl_hparams.single}, lambda_carrier: {gl_hparams.lambda_carrier_loss}, lambda_msg: {gl_hparams.lambda_msg_loss}, freeze_num: {gl_hparams.freeze_num}")
    # run(gl_hparams)
    # logger.info("Training complete!")
    # logger.info(f"")

if __name__ == '__main__':
    main()