"""
Preprocessed FHR Signal Viewer for Jupyter Notebook
Simple viewer for already-preprocessed FHR signals (no preprocessing steps)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_preprocessed_fhr(csv_path):
    """
    Load preprocessed FHR signal from CSV.
    Matches the dataloader format: skiprows=2, column 1 = FHR
    """
    df = pd.read_csv(csv_path, skiprows=2, header=None, usecols=[1], names=["FHR"])
    return df["FHR"].values.astype(np.float32)


def denormalize(signal, min_bpm=50, max_bpm=200):
    """Convert normalized [-1, 1] signal back to BPM range."""
    return (signal + 1) * (max_bpm - min_bpm) / 2 + min_bpm


def finalize_length(signal, sequence_length, crop_strategy='last'):
    """
    Crop or pad signal to target length.
    Mirrors the dataloader's _finalize_length method.
    """
    n = len(signal)
    L = sequence_length

    # If too short, pad with last value
    if n < L:
        pad_val = signal[-1] if n > 0 else 0.0
        return np.pad(signal, (0, L - n), mode='constant', constant_values=pad_val)

    # If exact length
    if n == L:
        return signal

    # If too long, crop based on strategy
    if crop_strategy == 'from_first_valid':
        return signal[:L]
    elif crop_strategy == 'center':
        start = max(0, (n - L) // 2)
        return signal[start:start + L]
    else:  # 'last'
        return signal[-L:]


def view_signal(
    data_folder,
    file_name,
    fs=1,
    xunit="seconds",
    show_denormalized=True,
    min_bpm=50,
    max_bpm=200,
    sequence_length=None,
    crop_strategy='last',
    figsize=(16, 8)
):
    """
    Load and visualize a preprocessed FHR signal in Jupyter.

    Parameters:
    -----------
    data_folder : str
        Path to folder containing preprocessed CSV files
    file_name : str
        Name of CSV file to view (e.g., '1202.csv')
    fs : int
        Sampling frequency in Hz (default: 4)
    xunit : str
        'seconds' or 'samples' (default: 'seconds')
    show_denormalized : bool
        Show both normalized and BPM views (default: True)
    min_bpm : int
        Minimum BPM for denormalization (default: 50)
    max_bpm : int
        Maximum BPM for denormalization (default: 200)
    sequence_length : int or None
        If specified, crop/pad to this length (mimics dataloader behavior)
    crop_strategy : str
        'last', 'center', or 'from_first_valid' (default: 'last')
    figsize : tuple
        Figure size (default: (16, 8))

    Returns:
    --------
    signal : numpy array
        The loaded (and optionally cropped/padded) signal

    Example:
    --------
    >>> # Basic usage
    >>> signal = view_signal("/path/to/data", "1202.csv")

    >>> # With sequence_length (as dataloader sees it)
    >>> signal = view_signal("/path/to/data", "1202.csv", sequence_length=1000)

    >>> # Different crop strategy
    >>> signal = view_signal("/path/to/data", "1202.csv", 
    ...                      sequence_length=1000, crop_strategy='center')
    """
    csv_path = os.path.join(data_folder, file_name)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"📂 Loading: {file_name}")
    signal = load_preprocessed_fhr(csv_path)

    original_length = len(signal)
    print(f"📊 Original: {original_length} samples ({original_length/fs:.1f}s = {original_length/fs/60:.2f} min)")

    # Apply sequence length handling if specified
    if sequence_length and sequence_length > 0:
        signal = finalize_length(signal, sequence_length, crop_strategy)
        print(f"✂️  Applied sequence_length={sequence_length} with crop_strategy='{crop_strategy}'")
        print(f"📊 Final: {len(signal)} samples ({len(signal)/fs:.1f}s = {len(signal)/fs/60:.2f} min)")

    n_samples = len(signal)
    duration_sec = n_samples / fs
    duration_min = duration_sec / 60

    # Create time axis
    if xunit == "seconds":
        time_axis = np.arange(n_samples) / fs
        xlabel = "Time (s)"
    else:
        time_axis = np.arange(n_samples)
        xlabel = "Sample Index"

    # Determine if signal is normalized
    is_normalized = (signal.min() >= -1.5) and (signal.max() <= 1.5)

    if show_denormalized and is_normalized:
        # Show both normalized and denormalized
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle(f"Preprocessed FHR Signal — {file_name}", fontsize=16, fontweight='bold')

        # Plot 1: Normalized signal
        ax = axes[0]
        ax.plot(time_axis, signal, linewidth=1, color='steelblue')
        ax.set_ylabel('Normalized Value', fontsize=11)
        ax.set_title(f'Normalized Signal (range: [{signal.min():.3f}, {signal.max():.3f}])')
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(-1, color='red', linestyle='--', alpha=0.3)
        ax.axhline(1, color='red', linestyle='--', alpha=0.3)

        # Plot 2: Denormalized to BPM
        denorm_signal = denormalize(signal, min_bpm, max_bpm)
        ax = axes[1]
        ax.plot(time_axis, denorm_signal, linewidth=1, color='darkgreen')
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Heart Rate (BPM)', fontsize=11)
        ax.set_title(f'Denormalized Signal (range: [{denorm_signal.min():.1f}, {denorm_signal.max():.1f}] BPM)')
        ax.grid(alpha=0.3)
        ax.set_ylim(40, 210)
        ax.axhline(min_bpm, color='red', linestyle='--', alpha=0.3, label=f'Min={min_bpm} BPM')
        ax.axhline(max_bpm, color='red', linestyle='--', alpha=0.3, label=f'Max={max_bpm} BPM')
        ax.legend(loc='upper right')

        info_text = (
            f"Samples: {n_samples} | Duration: {duration_sec:.1f}s ({duration_min:.2f} min) | "
            f"Fs: {fs} Hz | Mean: {denorm_signal.mean():.1f} BPM | Std: {denorm_signal.std():.1f} BPM"
        )
        fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=10)

    else:
        # Single plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(f"Preprocessed FHR Signal — {file_name}", fontsize=16, fontweight='bold')

        ax.plot(time_axis, signal, linewidth=1, color='steelblue')
        ax.set_xlabel(xlabel, fontsize=11)

        if is_normalized:
            ax.set_ylabel('Normalized Value', fontsize=11)
            ax.set_ylim(-1.2, 1.2)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            info_text = f"Samples: {n_samples} | Duration: {duration_sec:.1f}s ({duration_min:.2f} min) | Fs: {fs} Hz | Range: [{signal.min():.3f}, {signal.max():.3f}]"
        else:
            ax.set_ylabel('Heart Rate (BPM)', fontsize=11)
            ax.set_ylim(40, 210)
            ax.axhline(50, color='red', linestyle='--', alpha=0.3)
            ax.axhline(200, color='red', linestyle='--', alpha=0.3)
            info_text = f"Samples: {n_samples} | Duration: {duration_sec:.1f}s ({duration_min:.2f} min) | Fs: {fs} Hz | Mean: {signal.mean():.1f} BPM | Std: {signal.std():.1f} BPM"

        ax.grid(alpha=0.3)
        fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    print(f"\n✅ Done! Signal shape: {signal.shape}")
    return signal

# def view_signal_windows(
#     data_folder,
#     file_name,
#     fs=1,
#     sequence_length=4000,
#     win_len=1000,
#     crop_strategy='last',
#     min_bpm=50,
#     max_bpm=200,
#     figsize=(16, 8)
# ):
#     """
#     View satu sinyal 4000 yang kemudian dipotong menjadi beberapa window 1000,
#     dengan cara yang sama seperti WindowedFromLongDataset.
#     """
#     csv_path = os.path.join(data_folder, file_name)
#     if not os.path.isfile(csv_path):
#         raise FileNotFoundError(f"File not found: {csv_path}")

#     print(f"📂 Loading: {file_name}")
#     signal = load_preprocessed_fhr(csv_path)

#     print(f"📊 Original: {len(signal)} samples ({len(signal)/fs:.1f}s = {len(signal)/fs/60:.2f} min)")

#     # Samakan dengan myDataset: finalize dulu ke 4000
#     signal = finalize_length(signal, sequence_length, crop_strategy)
#     print(f"✂️  Finalized to sequence_length={sequence_length} (like base_dataset)")
#     print(f"📊 Final: {len(signal)} samples ({len(signal)/fs:.1f}s = {len(signal)/fs/60:.2f} min)")

#     # Hitung jumlah window
#     assert sequence_length % win_len == 0, "sequence_length harus kelipatan win_len"
#     num_win = sequence_length // win_len
#     print(f"🪟 Will split into {num_win} windows of {win_len} samples each")

#     # Plot grid windows, misal 2x2 kalau 4 window
#     n_rows = int(np.ceil(num_win / 2))
#     n_cols = min(2, num_win)

#     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=False)
#     if num_win == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1:
#         axes = np.array([axes])
#     axes = axes.flatten()

#     for w in range(num_win):
#         start = w * win_len
#         end = start + win_len
#         win_sig = signal[start:end]

#         # denorm ke BPM biar kebayang
#         win_denorm = denormalize(win_sig, min_bpm, max_bpm)
#         t = np.arange(win_len) / fs

#         ax = axes[w]
#         ax.plot(t, win_sig, linewidth=0.8)
#         ax.set_title(f"Window {w} ({start}:{end})")
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("BPM (Normalized)")
#         ax.grid(alpha=0.3)
#         ax.set_ylim(-1, 1)
#         ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
#         # ax.axhline(min_bpm, color='red', linestyle='--', alpha=0.3)
#         # ax.axhline(max_bpm, color='red', linestyle='--', alpha=0.3)

#     # Sembunyikan subplot yang tidak terpakai
#     for k in range(num_win, len(axes)):
#         axes[k].set_visible(False)

#     fig.suptitle(f"Windows like WindowedFromLongDataset — {file_name}", fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.show()

    # return signal  # sinyal 4000 setelah finalize

def view_signal_windows(
    data_folder,
    file_name,
    fs=1,
    sequence_length=4000,
    win_len=1000,
    crop_strategy='last',
    label_class="Abnormal Asli",
    figsize=(16, 10)
):
    """
    View satu sinyal 4000 yang kemudian dipotong menjadi beberapa
    window 1000, dengan style hijau seperti plot contoh.
    """
    csv_path = os.path.join(data_folder, file_name)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"📂 Loading: {file_name}")
    signal = load_preprocessed_fhr(csv_path)
    print(f"📊 Original: {len(signal)} samples ({len(signal)/fs:.1f}s = {len(signal)/fs/60:.2f} min)")

    # Samakan dengan base_dataset: finalize dulu ke sequence_length
    signal = finalize_length(signal, sequence_length, crop_strategy)
    print(f"✂️  Finalized to sequence_length={sequence_length}")
    print(f"📊 Final: {len(signal)} samples ({len(signal)/fs:.1f}s = {len(signal)/fs/60:.2f} min)")

    assert sequence_length % win_len == 0, "sequence_length harus kelipatan win_len"
    num_win = sequence_length // win_len
    print(f"🪟 Will split into {num_win} windows of {win_len} samples each")

    # Layout grid (max 2 kolom seperti contoh)
    n_cols = min(2, num_win)
    n_rows = int(np.ceil(num_win / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=False,
        sharey=False
    )

    # Pastikan axes selalu array 1D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    # Warna background figure
    fig.patch.set_facecolor("white")

    base_id = os.path.splitext(file_name)[0]  # misal "21000" dari "21000.csv"

    for w in range(num_win):
        start = w * win_len
        end = start + win_len
        win_sig = signal[start:end]              # sudah dalam [-1, 1]
        t = np.arange(win_len)                   # "Time Steps" seperti contoh

        ax = axes[w]

        # Background hijau muda per-axes
        ax.set_facecolor("#ffebee")  # light green e8f5e9 light red = ffebee

        # Plot garis biru
        ax.plot(t, win_sig, linewidth=1.0, color="blue")

        # Title: Generated Signal <id+offset>
        signal_id = f"{base_id} (w{w})"
        # signal_id = f"{int(base_id) + w}" if base_id.isdigit() else f"{base_id}_{w}"
        ax.set_title(
            f"Real Signal {signal_id}\n{label_class}",
            fontsize=11,
            fontweight="bold"
        )

        # Kotak kecil mean ± std di pojok kiri atas
        mean_val = win_sig.mean()
        std_val = win_sig.std()
        text_str = f"Mean: {mean_val:.1f}±{std_val:.1f}"
        ax.text(
            0.02, 0.98,
            text_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none")
        )

        # Y-axis: tetap normalized [-1, 1]
        # ax.set_ylim(-1.0, 1.0)
        # ax.set_yticks(np.linspace(-1, 1, 5))
        ax.set_ylim(-1.0, 1.0)
        yticks = np.arange(-1.0, 1.01, 0.25)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.2f}" for y in yticks])
        ax.grid(alpha=0.3)

        # X/Y label hanya di pinggir bawah/kiri (seperti template)
        if w // n_cols == n_rows - 1:
            ax.set_xlabel("Time Steps", fontsize=9)
        if w % n_cols == 0:
            ax.set_ylabel("FHR (bpm)\n(Normalized)", fontsize=9)

    # Sembunyikan subplot yang tidak terpakai
    for k in range(num_win, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Windows like WindowedFromLongDataset — {file_name}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return signal

def view_multiple_signals(
    data_folder,
    file_names,
    fs=1,
    sequence_length=None,
    crop_strategy='last',
    max_per_row=2
):
    """
    View multiple signals in a grid layout.

    Parameters:
    -----------
    data_folder : str
        Path to folder containing preprocessed CSV files
    file_names : list of str
        List of CSV filenames to view
    fs : int
        Sampling frequency in Hz (default: 4)
    sequence_length : int or None
        If specified, crop/pad to this length
    crop_strategy : str
        'last', 'center', or 'from_first_valid'
    max_per_row : int
        Maximum number of plots per row (default: 2)

    Example:
    --------
    >>> files = ["1202.csv", "1203.csv", "1204.csv", "1205.csv"]
    >>> view_multiple_signals("/path/to/data", files, sequence_length=1000)
    """
    n_files = len(file_names)
    n_cols = min(max_per_row, n_files)
    n_rows = (n_files + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows))
    fig.suptitle(f"Preprocessed FHR Signals ({n_files} files)", fontsize=16, fontweight='bold')

    if n_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_files > 1 else [axes]

    signals = []

    for idx, file_name in enumerate(file_names):
        csv_path = os.path.join(data_folder, file_name)

        try:
            signal = load_preprocessed_fhr(csv_path)

            if sequence_length and sequence_length > 0:
                signal = finalize_length(signal, sequence_length, crop_strategy)

            signals.append(signal)

            # Denormalize for plotting
            denorm_signal = denormalize(signal)
            time_axis = np.arange(len(signal)) / fs

            ax = axes[idx]
            ax.plot(time_axis, denorm_signal, linewidth=0.8, color='steelblue')
            ax.set_title(f"{file_name}", fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("BPM", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_ylim(40, 210)
            ax.axhline(50, color='red', linestyle='--', alpha=0.2, linewidth=0.8)
            ax.axhline(200, color='red', linestyle='--', alpha=0.2, linewidth=0.8)

        except Exception as e:
            ax = axes[idx]
            ax.text(0.5, 0.5, f"Error loading\n{file_name}\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{file_name} (ERROR)", fontsize=10, color='red')
            signals.append(None)

    # Hide unused subplots
    for idx in range(n_files, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

    print(f"\n✅ Loaded {n_files} signals")
    return signals


# Quick access function with shorter name
def view(data_folder, file_name, **kwargs):
    """Shorthand for view_signal()"""
    return view_signal(data_folder, file_name, **kwargs)
