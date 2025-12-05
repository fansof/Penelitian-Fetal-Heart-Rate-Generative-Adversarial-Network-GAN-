"""
FHRGAN-Style Evaluation Script
PER-SAMPLE RE (histogram + optional KDE) + PER-SAMPLE FD
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import json

from LoadDataset import myDataset
from CTGGAN1000 import Generator1000 as CTGGANGenerator
from windowed_dataset import WindowedFromLongDataset
from scipy.stats import gaussian_kde


class FHRGANStyleAnalyzer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("=" * 60)
        print("FHRGAN-STYLE ANALYZER (PER-SAMPLE RE + FD)")
        print("=" * 60)

    def relative_entropy_histogram_per_sample(self, real_samples, generated_samples, num_bins=11):
        """
        Per-sample Relative Entropy using histogram distributions.
        RE(x,z) = Σ_i p_i (ln p_i - ln q_i)
        """
        real = np.asarray(real_samples)
        gen = np.asarray(generated_samples)

        if real.ndim == 3:
            real = real.squeeze(1)  # (N, L)
        if gen.ndim == 3:
            gen = gen.squeeze(1)

        batch_size = min(real.shape[0], gen.shape[0])
        real = real[:batch_size]
        gen = gen[:batch_size]

        re_scores = []
        eps = 1e-8

        for i in range(batch_size):
            # map [-1,1] → [50,200] BPM
            real_bpm = ((real[i] + 1.0) * 150.0 / 2.0) + 50.0
            gen_bpm  = ((gen[i]  + 1.0) * 150.0 / 2.0) + 50.0

            real_hist, _ = np.histogram(real_bpm, bins=num_bins, range=(50, 200), density=False)
            gen_hist,  _ = np.histogram(gen_bpm,  bins=num_bins, range=(50, 200), density=False)

            real_dist = real_hist.astype(np.float64)
            gen_dist  = gen_hist.astype(np.float64)

            # additive smoothing + renormalize
            real_dist += eps
            gen_dist  += eps
            real_dist /= real_dist.sum()
            gen_dist  /= gen_dist.sum()

            re_i = np.sum(real_dist * (np.log(real_dist) - np.log(gen_dist)))
            re_scores.append(re_i)

        if len(re_scores) == 0:
            return float("nan")

        return float(np.mean(re_scores))

    def relative_entropy_kde_bpm_per_sample(
        self,
        real_bpm_samples,
        gen_bpm_samples,
        xmin=50.0,
        xmax=200.0,
        num_points=512,
        bw_method="scott"
    ):
        """
        Continuous KL (relative entropy) per-sample using KDE on BPM.
        """
        real = np.asarray(real_bpm_samples)
        gen  = np.asarray(gen_bpm_samples)

        if real.ndim == 3:
            real = real.squeeze(1)  # (N, L)
        if gen.ndim == 3:
            gen = gen.squeeze(1)

        if real.ndim == 1:
            real = real[None, :]
        if gen.ndim == 1:
            gen = gen[None, :]

        batch_size = min(real.shape[0], gen.shape[0])
        real = real[:batch_size]
        gen  = gen[:batch_size]

        xs = np.linspace(xmin, xmax, num_points)
        re_scores = []

        for i in range(batch_size):
            r = real[i].reshape(-1)
            g = gen[i].reshape(-1)

            r = r[(r >= xmin) & (r <= xmax)]
            g = g[(g >= xmin) & (g <= xmax)]
            if r.size == 0 or g.size == 0:
                continue

            try:
                p_kde = gaussian_kde(r, bw_method=bw_method)
                q_kde = gaussian_kde(g, bw_method=bw_method)
            except Exception:
                continue

            p_vals = p_kde(xs)
            q_vals = q_kde(xs)

            mask = (p_vals > 0.0) & (q_vals > 0.0)
            if mask.sum() < 10:
                continue

            xs_m = xs[mask]
            p_m  = p_vals[mask]
            q_m  = q_vals[mask]

            p_area = np.trapz(p_m, xs_m)
            q_area = np.trapz(q_m, xs_m)
            if p_area <= 0.0 or q_area <= 0.0:
                continue

            p_m /= p_area
            q_m /= q_area

            integrand = p_m * (np.log(p_m) - np.log(q_m))
            re_i = float(np.trapz(integrand, xs_m))
            re_scores.append(re_i)

        if len(re_scores) == 0:
            return float("nan")

        return float(np.mean(re_scores))

    def frechet_distance_per_sample(self, real_samples, generated_samples):
        """
        Per-sample discrete Fréchet distance between normalized sequences.
        """
        real = np.asarray(real_samples)
        gen  = np.asarray(generated_samples)

        if real.ndim == 3:
            real = real.squeeze(1)
        if gen.ndim == 3:
            gen = gen.squeeze(1)

        batch_size = min(real.shape[0], gen.shape[0])
        real = real[:batch_size]
        gen  = gen[:batch_size]

        fd_scores = []

        def discrete_frechet_distance(seq1, seq2):
            n, m = len(seq1), len(seq2)

            max_length = 1000
            if n > max_length:
                step = n // max_length
                seq1 = seq1[::step]
                n = len(seq1)
            if m > max_length:
                step = m // max_length
                seq2 = seq2[::step]
                m = len(seq2)

            dist_matrix = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dist_matrix[i, j] = abs(seq1[i] - seq2[j])

            dp = np.full((n, m), np.inf)
            dp[0, 0] = dist_matrix[0, 0]

            for j in range(1, m):
                dp[0, j] = max(dp[0, j - 1], dist_matrix[0, j])

            for i in range(1, n):
                dp[i, 0] = max(dp[i - 1, 0], dist_matrix[i, 0])

            for i in range(1, n):
                for j in range(1, m):
                    dp[i, j] = max(
                        min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                        dist_matrix[i, j]
                    )
            return dp[n - 1, m - 1]

        for i in range(batch_size):
            fd = discrete_frechet_distance(real[i], gen[i])
            fd_scores.append(fd)

        if len(fd_scores) == 0:
            return float("nan")

        return float(np.mean(fd_scores))

    def analyze_data_characteristics(self, real_data, gen_data, data_name=""):
        print(f"\n{'-' * 40}")
        print(f"DATA ANALYSIS - {data_name}")
        print(f"{'-' * 40}")
        print(f"Shape: Real {real_data.shape}, Generated {gen_data.shape}")
        print(f"Range: Real [{np.min(real_data):.3f}, {np.max(real_data):.3f}]")
        print(f"       Gen  [{np.min(gen_data):.3f}, {np.max(gen_data):.3f}]")
        print(f"Stats: Real μ={np.mean(real_data):.3f}, σ={np.std(real_data):.3f}")
        print(f"       Gen  μ={np.mean(gen_data):.3f}, σ={np.std(gen_data):.3f}")

    def evaluate(self, real_norm, gen_norm, real_bpm, gen_bpm, data_name=""):
        self.analyze_data_characteristics(real_norm, gen_norm, "NORMALIZED [-1,1]")
        self.analyze_data_characteristics(real_bpm, gen_bpm, "BPM [50,200]")

        print(f"\n{'-' * 40}")
        print(f"FHRGAN-STYLE METRICS - {data_name}")
        print(f"{'-' * 40}")

        re_hist = self.relative_entropy_histogram_per_sample(real_norm, gen_norm, num_bins=11)
        print(f"RE (histogram per-sample): {re_hist:.6f}")

        fd = self.frechet_distance_per_sample(real_norm, gen_norm)
        print(f"FD (sequence-based per-sample): {fd:.6f}")

        re_kde = self.relative_entropy_kde_bpm_per_sample(real_bpm, gen_bpm, xmin=50.0, xmax=200.0)
        print(f"RE (KDE per-sample, BPM): {re_kde:.6f}")

        return {
            'RE_hist': re_hist,
            'FD': fd,
            'RE_kde_per_sample': re_kde
        }


def create_fhrgan_comparison_plots(real_bpm, gen_bpm, metrics, save_path="./results/"):
    os.makedirs(save_path, exist_ok=True)

    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 120

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.9, bottom=0.08)

    fig.suptitle('FHRGAN-Style Evaluation (CTGGAN) - Per-sample RE + FD',
                 fontsize=15, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 0])
    plot_length = min(1000, real_bpm.shape[-1])
    t = np.arange(plot_length)
    ax1.plot(t, real_bpm[0, 0, :plot_length], 'r-', alpha=0.8, label='Real', linewidth=1.5)
    ax1.plot(t, gen_bpm[0, 0, :plot_length], 'b-', alpha=0.8, label='Generated', linewidth=1.5)
    ax1.set_title('Sample FHR Signals', fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('BPM')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([60, 180])

    ax2 = fig.add_subplot(gs[0, 1])
    models = ['CTGGAN']
    re_values = [metrics['RE_hist']]
    bars = ax2.bar(models, re_values, color=['#2ecc71'], edgecolor='black')
    ax2.set_title('Relative Entropy (per-sample, hist)', fontweight='bold')
    ax2.set_ylabel('RE')
    ax2.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars, re_values):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{value:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    fd_values = [metrics['FD']]
    bars = ax3.bar(['CTGGAN'], fd_values, color=['#3498db'], edgecolor='black')
    ax3.set_title('Fréchet Distance (per-sample)', fontweight='bold')
    ax3.set_ylabel('FD')
    ax3.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars, fd_values):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{value:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4 = fig.add_subplot(gs[1, 0])
    real_sample = np.random.choice(real_bpm.reshape(-1),
                                   min(60000, real_bpm.size),
                                   replace=False)
    gen_sample = np.random.choice(gen_bpm.reshape(-1),
                                  min(60000, gen_bpm.size),
                                  replace=False)
    ax4.hist(real_sample, bins=50, alpha=0.6, label='Real', density=True,
             color='red', edgecolor='black', linewidth=0.5)
    ax4.hist(gen_sample, bins=50, alpha=0.6, label='Generated', density=True,
             color='blue', edgecolor='black', linewidth=0.5)
    ax4.set_title('BPM Distributions (Global)', fontweight='bold')
    ax4.set_xlabel('BPM')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xlim([60, 180])

    ax5 = fig.add_subplot(gs[1, 1])
    real_all = real_bpm.reshape(-1)
    gen_all  = gen_bpm.reshape(-1)
    num_bins = 11
    real_hist, bins = np.histogram(real_all, bins=num_bins, range=(50, 200), density=False)
    gen_hist,  _    = np.histogram(gen_all,  bins=num_bins, range=(50, 200), density=False)
    real_dist = real_hist / real_hist.sum()
    gen_dist  = gen_hist  / gen_hist.sum()
    centers   = (bins[:-1] + bins[1:]) / 2
    ax5.bar(centers, real_dist, width=(bins[1]-bins[0]),
            alpha=0.6, label='Real', color='red', edgecolor='black')
    ax5.bar(centers, gen_dist, width=(bins[1]-bins[0]),
            alpha=0.6, label='Generated', color='blue', edgecolor='black')
    ax5.set_title('Histogram Basis (All Sequences)', fontweight='bold')
    ax5.set_xlabel('BPM')
    ax5.set_ylabel('Probability')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_xlim([50, 200])

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_text = f"""FHRGAN-STYLE METRICS (Per-sample)

RE (hist, per-sample): {metrics['RE_hist']:.3f}
FD (per-sample):       {metrics['FD']:.3f}
RE (KDE, per-sample):  {metrics['RE_kde_per_sample']:.3f}

Data:
  Range BPM: [50, 200]
  Window length: {real_bpm.shape[-1]} steps
  Num windows: {real_bpm.shape[0]}
"""
    ax6.text(0.05, 0.95, summary_text,
             fontsize=9, va='top', family='monospace',
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#fffacd", edgecolor='black', alpha=0.9))

    out_path = os.path.join(save_path, "fhrgan_per_sample_metrics__CTGGANph715NORMAL.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved: {out_path}")
    plt.close()


def main():
    generator_path = "/home/fauzi/Documents/generateGAN_untukSKRIPSI/1a-GENERATE SAMPLE LAST/CTGGAN715/window/final_G.pth"
    # generator_path = "/home/fauzi/Documents/generateGAN_untukSKRIPSI/1a-GENERATE SAMPLE LAST/CTGGAN72/window/final_G.pth"
    data_path      = "/home/fauzi/Documents/generateGAN_untukSKRIPSI/1APREPROCESSEDFILE/PREPROCESSED_OKTOBER"
    ph_label       = "/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW/ph_labels.csv"

    sequence_length = 1000
    num_samples     = 240
    random_seed     = 81

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("\n" + "=" * 60)
    print("FHRGAN-STYLE EVALUATION (PER-SAMPLE)")
    print("=" * 60 + "\n")

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        print("\nLoading CTGGAN generator...")
        generator = CTGGANGenerator(num_classes=2).to(device)
        checkpoint = torch.load(generator_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        elif 'G_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['G_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        generator.eval()
        print("✓ CTGGAN Generator loaded successfully")

        print("\nLoading real data...")
        dataset = myDataset(
            data_path, ph_label, sequence_length=4000,
            max_normal=447, max_pathological=105,
            train_normal=70, train_pathological=70,
            eval_normal=30,  eval_pathological=0,
            mode='eval', random_seed=random_seed
        )

        dataset2   = WindowedFromLongDataset(dataset, win_len=1000)
        dataloader = DataLoader(dataset2, batch_size=120, shuffle=True)
        real_data, labels = next(iter(dataloader))
        real_data = real_data.to(device)
        labels    = labels.to(device)

        normal_count = int(torch.sum(labels == 1).item())
        patho_count  = int(torch.sum(labels == 0).item())

        print(f"✓ Loaded {len(real_data)} real samples")
        print(f"  → Normal: {normal_count}")
        print(f"  → Pathological: {patho_count}")

        print("\nGenerating synthetic data...")
        with torch.no_grad():
            gen_parts = []
            if normal_count > 0:
                noise_normal  = torch.randn(normal_count, 100, 1, device=device)
                labels_normal = torch.ones(normal_count, dtype=torch.long, device=device)
                gen_parts.append(generator(noise_normal, labels_normal))
            if patho_count > 0:
                noise_patho  = torch.randn(patho_count, 100, 1, device=device)
                labels_patho = torch.zeros(patho_count, dtype=torch.long, device=device)
                gen_parts.append(generator(noise_patho, labels_patho))
            gen_data = torch.cat(gen_parts, dim=0) if len(gen_parts) > 1 else gen_parts[0]

        print(f"✓ Generated {len(gen_data)} synthetic samples")

        print("\nConverting data...")
        real_norm = real_data.detach().cpu().numpy()
        gen_norm  = gen_data.detach().cpu().numpy()
        real_bpm  = ((real_norm + 1.0) * 150.0 / 2.0) + 50.0
        gen_bpm   = ((gen_norm  + 1.0) * 150.0 / 2.0) + 50.0

        print("✓ Data converted")
        print(f"  Real BPM: [{np.min(real_bpm):.1f}, {np.max(real_bpm):.1f}]")
        print(f"  Gen  BPM: [{np.min(gen_bpm):.1f}, {np.max(gen_bpm):.1f}]")

        analyzer = FHRGANStyleAnalyzer(device=device)
        metrics  = analyzer.evaluate(real_norm, gen_norm, real_bpm, gen_bpm, "CTGGAN")

        print(f"\nSaving results to JSON...")
        results = {
            'model': 'CTGGAN',
            'evaluation_method': 'per_sample_hist_fd',
            'data_mapping': '[-1,1] → [50,200] BPM',
            'num_samples': num_samples,
            'sequence_length': sequence_length,
            'metrics': {
                'RE_hist_per_sample': float(metrics['RE_hist']),
                'FD_per_sample':      float(metrics['FD']),
                'RE_kde_per_sample':  float(metrics['RE_kde_per_sample'])
            },
            'data_statistics': {
                'real_bpm': {
                    'mean': float(np.mean(real_bpm)),
                    'std':  float(np.std(real_bpm)),
                    'min':  float(np.min(real_bpm)),
                    'max':  float(np.max(real_bpm))
                },
                'generated_bpm': {
                    'mean': float(np.mean(gen_bpm)),
                    'std':  float(np.std(gen_bpm)),
                    'min':  float(np.min(gen_bpm)),
                    'max':  float(np.max(gen_bpm))
                }
            }
        }

        os.makedirs("./results/", exist_ok=True)
        with open("./results/fhrgan_per_sample_metrics_CTGGANph715NORMAL.json", "w") as f:
            json.dump(results, f, indent=2)
        print("✓ Results saved to ./results/fhrgan_per_sample_metrics_CTGGANph715NORMAL.json")

        create_fhrgan_comparison_plots(real_bpm, gen_bpm, metrics, "./results/")

        print("\n" + "=" * 60)
        print("✓✓✓ PER-SAMPLE EVALUATION COMPLETED ✓✓✓")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
