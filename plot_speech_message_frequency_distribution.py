import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import config
from config import device
from model import get_models

from analyze_frequency_energy import build_test_loader, str2bool


BANDS = [
    ("0-2k", 0.0, 2000.0),
    ("2-4k", 2000.0, 4000.0),
    ("4-6k", 4000.0, 6000.0),
    ("6-8k", 6000.0, 8000.0),
]

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "message_representation_frequency_comparison"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare frequency distributions of msg, msg_emb, msg_merged, and msg_u."
    )
    parser.add_argument("--test_path", required=True, type=str, help="Path to the test wav folder.")
    parser.add_argument("--load_ckpt", required=True, type=str, help="Directory containing encoder.ckpt.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    parser.add_argument("--run_dir", default=".", type=str, help="Kept for compatibility with existing hparams.")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_pairs", default=832, type=int)
    parser.add_argument("--single", default=False, type=str2bool)
    parser.add_argument("--message_file", default=None, type=str)
    parser.add_argument("--dataset", default="timit", choices=["timit", "mini"])
    parser.add_argument("--model_type", default="normal", choices=["normal"])
    parser.add_argument("--block_type", default="normal", choices=["normal", "skip", "bn", "in", "relu"])
    parser.add_argument("--enc_n_layers", default=3, type=int)
    parser.add_argument("--dec_c_n_layers", default=4, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_iters", default=1, type=int)
    parser.add_argument("--loss_type", default="mse", choices=["mse", "abs"])
    parser.add_argument("--lambda_carrier_loss", default=3.0, type=float)
    parser.add_argument("--lambda_msg_loss", default=1.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--save_model_every", default=None)
    parser.add_argument("--sample_every", default=None)
    parser.add_argument("--msg_binary_watermark", action="store_true")
    parser.add_argument("--sr", default=16000, type=int)
    return parser.parse_args()


def load_encoder(args):
    build_args = argparse.Namespace(**vars(args))
    build_args.load_ckpt = None
    encoder, _, _, _ = get_models(build_args)

    ckpt_path = Path(args.load_ckpt) / "encoder.ckpt"
    state_dict = torch.load(str(ckpt_path), map_location=device)
    try:
        encoder.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            stripped[key.replace("module.", "", 1) if key.startswith("module.") else key] = value
        encoder.load_state_dict(stripped)

    encoder.eval()
    return encoder


def frequency_axis(n_freq_bins, sr):
    return np.linspace(0.0, sr / 2.0, n_freq_bins)


def add_power(power_sum, tensor):
    tensor = tensor.detach().double()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError(f"Expected [B, C, F, T] or [B, F, T], got shape {tuple(tensor.shape)}")

    power = tensor ** 2
    current = power.sum(dim=(0, 1, 3)).cpu().numpy()
    if power_sum is None:
        return current
    return power_sum + current


def band_percentages(freqs, power):
    band_energy = np.zeros(len(BANDS), dtype=np.float64)
    for idx, (_, low, high) in enumerate(BANDS):
        if idx == len(BANDS) - 1:
            mask = (freqs >= low) & (freqs <= high)
        else:
            mask = (freqs >= low) & (freqs < high)
        band_energy[idx] = float(power[mask].sum())

    total = float(power.sum())
    if total <= 0.0:
        raise RuntimeError("Total frequency energy is zero; cannot compute distribution.")
    return band_energy, 100.0 * band_energy / total


def normalize_db(power):
    power = np.asarray(power, dtype=np.float64)
    power = power / max(float(power.max()), 1e-12)
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def save_plot(path_png, path_pdf, freqs, powers, percentages):
    labels = [label for label, _, _ in BANDS]
    x = np.arange(len(labels))
    width = 0.18
    colors = {
        "msg": "#4C78A8",
        "msg_emb": "#F58518",
        "msg_merged": "#72B7B2",
        "msg_u": "#E45756",
    }
    display_names = {
        "msg": "msg",
        "msg_emb": "msg_emb",
        "msg_merged": "msg_merged",
        "msg_u": "msg_u",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for name, power in powers.items():
        axes[0].plot(
            freqs,
            normalize_db(power),
            label=display_names[name],
            color=colors[name],
            linewidth=1.8,
        )
    axes[0].set_xlim(0, freqs[-1])
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Normalized Power (dB)")
    axes[0].set_title("Average Spectrum")
    axes[0].grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    axes[0].legend(frameon=False)

    offsets = {
        "msg": -1.5 * width,
        "msg_emb": -0.5 * width,
        "msg_merged": 0.5 * width,
        "msg_u": 1.5 * width,
    }
    bars_list = []
    max_pct = 0.0
    for name, pct in percentages.items():
        max_pct = max(max_pct, float(pct.max()))
        bars = axes[1].bar(
            x + offsets[name],
            pct,
            width,
            label=display_names[name],
            color=colors[name],
        )
        bars_list.append(bars)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, max(108.0, max_pct * 1.18))
    axes[1].set_ylabel("Energy Percentage (%)")
    axes[1].set_title("Band Energy Distribution")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    axes[1].legend(frameon=False)

    for bars in bars_list:
        for bar in bars:
            value = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)


def write_csv(path, energies, percentages):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["frequency_band"]
        for name in energies:
            header.extend([f"{name}_energy", f"{name}_energy_percentage"])
        writer.writerow(header)

        for idx, (label, _, _) in enumerate(BANDS):
            row = [label]
            for name in energies:
                row.extend([f"{energies[name][idx]:.8f}", f"{percentages[name][idx]:.4f}"])
            writer.writerow(row)


def main():
    args = parse_args()
    if args.single and not args.message_file:
        raise ValueError("--message_file is required when --single True is used.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config.gl_hparams = args
    encoder = load_encoder(args)
    loader = build_test_loader(args)

    powers = {
        "msg": None,
        "msg_emb": None,
        "msg_merged": None,
        "msg_u": None,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Message representation frequency"):
            if len(batch) == 4:
                _, msg, _, _ = batch
            else:
                _, msg = batch

            msg = msg.to(device)
            msg_emb = encoder.encoder_first(msg)
            msg_merged = torch.cat((msg, msg_emb), dim=1)
            msg_u = encoder.encoder_second(msg_merged)

            powers["msg"] = add_power(powers["msg"], msg)
            powers["msg_emb"] = add_power(powers["msg_emb"], msg_emb)
            powers["msg_merged"] = add_power(powers["msg_merged"], msg_merged)
            powers["msg_u"] = add_power(powers["msg_u"], msg_u)

    freqs = frequency_axis(powers["msg"].shape[0], args.sr)
    for name, power in powers.items():
        current_freqs = frequency_axis(power.shape[0], args.sr)
        if not np.allclose(freqs, current_freqs):
            raise RuntimeError(f"Frequency bins for {name} do not match msg.")

    energies = {}
    percentages = {}
    for name, power in powers.items():
        energies[name], percentages[name] = band_percentages(freqs, power)

    png_path = output_dir / "message_representation_frequency_distribution.png"
    pdf_path = output_dir / "message_representation_frequency_distribution.pdf"
    csv_path = output_dir / "message_representation_frequency_distribution.csv"

    save_plot(png_path, pdf_path, freqs, powers, percentages)
    write_csv(csv_path, energies, percentages)

    print(f"Output dir: {output_dir}")
    print(f"Figure PNG: {png_path}")
    print(f"Figure PDF: {pdf_path}")
    print(f"CSV: {csv_path}")
    print("\nFrequency band energy distribution:")
    for idx, (label, _, _) in enumerate(BANDS):
        values = ", ".join(f"{name}={percentages[name][idx]:.2f}%" for name in percentages)
        print(f"  {label}: {values}")


if __name__ == "__main__":
    main()
