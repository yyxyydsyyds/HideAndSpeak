import argparse
import torch
gl_hparams = None

DEBUG = False  # 改为 True 启用，False 禁用

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     device = torch.device("cuda")
# else:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hparams():
    global gl_hparams
    parser = argparse.ArgumentParser(description='Hide and Speak')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--num_iters', type=int, default=100, help='number of epochs')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'abs'], help='loss function used for training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'], help='`train` will initiate training, `test` should be used in conjunction with `load_ckpt` to run a test epoch, `sample` should be used in conjunction with `load_ckpt` to sample examples from dataset')
    parser.add_argument('--train_path', required=True, type=str, help='path to training set. should be a folder containing .wav files for training')
    parser.add_argument('--val_path', required=True, type=str, help='')
    parser.add_argument('--test_path', required=True, type=str, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_pairs', type=int, default=32, help='number of training examples generated from wav files')
    parser.add_argument('--dataset', type=str, default='timit', help='select dataset', choices=['timit', 'mini'])

    parser.add_argument('--block_type', type=str, default='normal', choices=['normal', 'skip', 'bn', 'in', 'relu'], help='type of block for encoder/decoder')
    parser.add_argument('--enc_n_layers', default=3, type=int, help='number of layers in encoder')
    parser.add_argument('--dec_c_n_layers', default=4, type=int, help='number of layers in decoder')
    parser.add_argument('--lambda_carrier_loss', type=float, default=3.0, help='coefficient for carrier loss term')
    parser.add_argument('--lambda_msg_loss', type=float, default=1.0, help='coefficient for message loss term')

    parser.add_argument('--num_workers', type=int, default=20, help='number of data loading workers')
    parser.add_argument('--load_ckpt', type=str, default=None, help='path to checkpoint (used for test epoch or for sampling)')
    parser.add_argument('--run_dir', type=str, default='.', help='output directory for logs, samples and checkpoints')
    parser.add_argument('--save_model_every', type=int, default=None, help='')
    parser.add_argument('--sample_every', type=int, default=None, help='')

    parser.add_argument('--single', type=bool, default=False)
    parser.add_argument('--message_file', type=str)

    parser.add_argument('--model_type', type=str, default='normal', choices=['normal','GAN','unet'], help='type of model')
    parser.add_argument('--freeze_num', type=int, default=1, help='num of epochs to freeze discriminator')
    parser.add_argument('--norm', default='instance', help='batch or instance')
    parser.add_argument('--use_wandb', action='store_true', help='enable wandb logging')



    gl_hparams = parser.parse_args()
    return gl_hparams

