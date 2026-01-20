# ablate_stft.py
import argparse, math, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import get_hparams, device
from dataset import val_dataloader
from model import get_models
from solver import snr, training_step  # 直接复用现有指标
from loguru import logger

# --- 生成 R×C 网格 mask：1 表示保留，0 表示置零 ---
def grid_masks(F, T, rows, cols, to='tile'):
    masks = []
    print(f"to:{to}, rows:{rows}, cols:{cols}, F:{F}, T:{T}")
    if to == 'freq':  # 频带
        for r in range(rows):
            m = torch.ones(1, 1, F, T)
            f0 = int(r * F / rows)
            f1 = int((r + 1) * F / rows)
            m[:, :, f0:f1, :] = 0
            masks.append(m)
    elif to == 'time':  # 时间窗
        for c in range(cols):
            m = torch.ones(1, 1, F, T)
            t0 = int(c * T / cols)
            t1 = int((c + 1) * T / cols)
            m[:, :, :, t0:t1] = 0
            masks.append(m)
    else:  # 小格子
        for r in range(rows):
            for c in range(cols):
                m = torch.ones(1, 1, F, T)
                f0 = int(r * F / rows); f1 = int((r + 1) * F / rows)
                t0 = int(c * T / cols); t1 = int((c + 1) * T / cols)
                m[:, :, f0:f1, t0:t1] = 0
                masks.append(m)
    return masks

@torch.no_grad()
def ablate(hp, rows=8, cols=8, mode='tile', target='container', max_batches=10, ckpt_epoch=None, outdir='ablate_out'):
    os.makedirs(outdir, exist_ok=True)

    # 1) 拿模型
    from model import load_models
    encoder, decoder, _, _ = get_models(hp)
    print(f"ablate load_ckpt: {hp.load_ckpt}\n")
    ckpt_path = os.path.join(hp.load_ckpt, f"{ckpt_epoch}_epoch")
    print(f"ablate ckpt_path: {ckpt_path}\n")

    load_models(encoder, decoder, ckpt_path)
    encoder.eval(); decoder.eval()

    # 可选：如果你想加载某个 epoch 的权重（solver.save_models 每轮都存）
    # if hp.load_ckpt:
    #     logger.info(f'Loaded ckpt from {hp.load_ckpt}')
    # elif ckpt_epoch:
    # hp.load_ckpt = os.path.join(hp.load_ckpt, f'{ckpt_epoch}_epoch')
    # logger.info(f'Loading ckpt epoch {ckpt_epoch} -> {hp.load_ckpt}')


    # 2) 数据（用验证集跑就好）
    val_loader = val_dataloader(hp.val_path, hp.batch_size, hp.num_workers)

    # 3) 先做 baseline（不置零）
    base_msg_loss, base_car_loss, base_msg_snr, base_car_snr = [], [], [], []
    n_batches = 0
    for carrier, msg in val_loader:
        carrier = carrier.to(device); 
        msg = msg.to(device)
    # for batch in val_loader:
    #     # 兼容 dataset 返回 (carrier, msg) 或 (carrier, msg, carrier_audio, msg_audio)
    #     if isinstance(batch, (list, tuple)) and len(batch) == 4:
    #         carrier, msg, _, _ = batch
    #     else:
    #         carrier, msg = batch
    #     carrier = carrier.to(device); msg = msg.to(device)

        container = encoder(carrier, msg)
        msg_rec = decoder(container)
        loss, logs = training_step(
            carrier, container, msg, msg_rec,
            lambda_carrier=hp.lambda_carrier_loss,
            lambda_msg=hp.lambda_msg_loss,
            loss_type=hp.loss_type
        )
        base_msg_loss.append(logs['msg_loss'])
        base_car_loss.append(logs['carrier_loss'])
        base_msg_snr.append(float(snr(msg, msg_rec)))
        base_car_snr.append(float(snr(carrier, container)))
        n_batches += 1
        if n_batches >= max_batches: break

    b_msg_loss = float(np.mean(base_msg_loss))
    b_car_loss = float(np.mean(base_car_loss))
    b_msg_snr  = float(np.mean(base_msg_snr))
    b_car_snr  = float(np.mean(base_car_snr))
    print(f'Baseline  msg_loss={b_msg_loss:.4f}, car_loss={b_car_loss:.4f}, '
          f'msg_SNR={b_msg_snr:.2f}dB, car_SNR={b_car_snr:.2f}dB')

    # 4) 做 mask 并统计掉多少
    F, T = carrier.shape[-2], carrier.shape[-1]  # 129 x 378 通常
    masks = grid_masks(F, T, rows, cols, to=mode)
    n_masks = len(masks)

    drop_msg_loss = np.zeros(n_masks)
    drop_car_loss = np.zeros(n_masks)
    drop_msg_snr  = np.zeros(n_masks)
    drop_car_snr  = np.zeros(n_masks)

    for i, m in enumerate(masks):
        m = m.to(device)
        mm_msg, mm_car, sm_msg, sm_car = [], [], [], []
        nb = 0
        for carrier, msg in val_loader:
            carrier = carrier.to(device); msg = msg.to(device)
        # for batch in val_loader:
        # # 兼容 dataset 返回 (carrier, msg) 或 (carrier, msg, carrier_audio, msg_audio)
        #     if isinstance(batch, (list, tuple)) and len(batch) == 4:
        #         carrier, msg, _, _ = batch
        #     else:
        #         carrier, msg = batch
        #     carrier = carrier.to(device); msg = msg.to(device)
            if target == 'carrier_input':           # 在 encoder 前屏蔽载体
                c_in = carrier * m
                # if hp.model_type == 'unet':
                #     x = torch.cat([c_in, msg], dim=1)
                #     container = encoder(x); container = container + c_in
                # else:
                container = encoder(c_in, msg)
            elif target == 'msg_input':              # 在 encoder 前屏蔽消息
                m_in = msg * m
                if hp.model_type == 'unet':
                    x = torch.cat([carrier, m_in], dim=1)
                    container = encoder(x); container = container + carrier
                else:
                    container = encoder(carrier, m_in)
            else:                                    # target == 'container'：在 decoder 前屏蔽容器
                # if hp.model_type == 'unet':
                #     x = torch.cat([carrier, msg], dim=1)
                #     container = encoder(x); container = container + carrier
                # else:
                container = encoder(carrier, msg)
                container = container * m

            # print(f"hp.lambda_carrier_loss: {hp.lambda_carrier_loss}, hp.lambda_msg_loss: {hp.lambda_msg_loss}, hp.loss_type: {hp.loss_type}\n")
            msg_rec = decoder(container)
            _, logs = training_step(
                carrier, container, msg, msg_rec,
                lambda_carrier=hp.lambda_carrier_loss,
                lambda_msg=hp.lambda_msg_loss,
                loss_type=hp.loss_type
            )
                                    
            mm_msg.append(logs['msg_loss'])
            mm_car.append(logs['carrier_loss'])
            sm_msg.append(float(snr(msg, msg_rec)))
            sm_car.append(float(snr(carrier, container)))
            nb += 1
            if nb >= max_batches: break

        drop_msg_loss[i] = float(np.mean(mm_msg)) - b_msg_loss
        drop_car_loss[i] = float(np.mean(mm_car)) - b_car_loss
        drop_msg_snr[i]  = b_msg_snr - float(np.mean(sm_msg))
        drop_car_snr[i]  = b_car_snr - float(np.mean(sm_car))
        print(f'[{i+1}/{n_masks}] Masked {target} {mode} #{i}: '
              f'Δ msg_loss={drop_msg_loss[i]:.4f}, Δ car_loss={drop_car_loss[i]:.4f}, '
              f'Δ msg_SNR={drop_msg_snr[i]:.2f}dB, Δ car_SNR={drop_car_snr[i]:.2f}dB')
        

    # 5) 可视化（热力图）
    # def heatmap(arr, title, fname):
    #     if mode == 'tile':
    #         H = arr.reshape(rows, cols)
    #     elif mode == 'freq':
    #         H = arr.reshape(rows, 1)
    #     else:  # time
    #         H = arr.reshape(1, cols)
    #     plt.figure()
    #     plt.imshow(H, origin='lower', aspect='auto')
    #     plt.colorbar()
    #     plt.title(title)
    #     plt.xlabel('time bins'); plt.ylabel('freq bins')
    #     plt.tight_layout()
    #     path = os.path.join(outdir, fname)
    #     plt.savefig(path, dpi=180); plt.close()
    #     print('saved:', path)

    # heatmap(drop_msg_loss, f'Δ msg_loss ({target}, {mode})', f'heat_drop_msg_loss_{target}_{mode}.png')
    # heatmap(drop_car_loss, f'Δ carrier_loss ({target}, {mode})', f'heat_drop_car_loss_{target}_{mode}.png')
    # heatmap(drop_msg_snr,  f'Δ msg_SNR ({target}, {mode})', f'heat_drop_msg_snr_{target}_{mode}.png')
    # heatmap(drop_car_snr,  f'Δ carrier_SNR ({target}, {mode})', f'heat_drop_car_snr_{target}_{mode}.png')

# if __name__ == '__main__':
#     hp = get_hparams()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--rows', type=int, default=8)
#     parser.add_argument('--cols', type=int, default=8)
#     parser.add_argument('--ablate_mode', type=str, default='tile', choices=['tile','freq','time'])
#     parser.add_argument('--target', type=str, default='container',
#                         choices=['container','carrier_input','msg_input'])
#     parser.add_argument('--max_batches', type=int, default=10)
#     parser.add_argument('--ckpt_epoch', type=int, default=None)
#     parser.add_argument('--outdir', type=str, default='ablate_out')
#     args = parser.parse_args()
#     ablate(hp, rows=args.rows, cols=args.cols, mode=args.mode,
#            target=args.target, max_batches=args.max_batches,
#            ckpt_epoch=args.ckpt_epoch, outdir=args.outdir)
