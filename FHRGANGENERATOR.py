import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from NewFHRGANmodel131018nov import Generator  # Generator Model



def generate_signals(generator, label, num_signals, latent_dim, num_classes, device, keep_normalized=False, batch_size=8):
    """
    Generate synthetic FHR signals in BATCHES to avoid OOM
    
    Args:
        generator: Trained generator model
        label: Class label (0=pathological, 1=normal)
        num_signals: Total number of signals to generate
        latent_dim: Latent dimension size
        num_classes: Number of classes
        device: torch device
        keep_normalized: If True, keep in [-1,1] range. If False, convert to BPM [50,200]
        batch_size: Number of signals to generate per batch (CRITICAL for memory)
    
    Returns:
        generated_signals: numpy array of shape (num_signals, sequence_length)
    """
    generator.eval()
    all_generated = []
    
    # Calculate number of batches needed
    num_batches = (num_signals + batch_size - 1) // batch_size
    
    print(f"Generating {num_signals} signals in {num_batches} batches of {batch_size}...")
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            # Calculate how many signals in this batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_signals)
            current_batch_size = end_idx - start_idx
            
            # Generate random noise for this batch
            z = torch.randn(current_batch_size, latent_dim, device=device)
            
            # Create label tensor and one-hot encode
            labels_tensor = torch.full((current_batch_size,), label, dtype=torch.long, device=device)
            labels_1h = F.one_hot(labels_tensor, num_classes).float()
            
            # Generate signals for this batch
            generated = generator(z, labels_1h)
            
            # Remove channel dimension if present: (B, 1, L) -> (B, L)
            if generated.dim() == 3:
                generated = generated.squeeze(1)
            
            # Move to CPU and convert to numpy
            generated_batch = generated.cpu().numpy()
            all_generated.append(generated_batch)
            
            # Clear GPU memory
            del z, labels_tensor, labels_1h, generated, generated_batch
            torch.cuda.empty_cache()
            
            print(f"  Batch {batch_idx + 1}/{num_batches} complete ({current_batch_size} signals)")
    
    # Concatenate all batches
    generated_signals = np.concatenate(all_generated, axis=0)
    
    # Print raw output info
    print(f"Raw generator output range: {generated_signals.min():.3f} to {generated_signals.max():.3f}")
    print(f"Raw generator output mean: {generated_signals.mean():.3f}")
    
    # Convert to BPM if requested
    if not keep_normalized:
        generated_signals = convert_normalized_to_fhr(generated_signals)
        print(f"Converted FHR range: {generated_signals.min():.3f} to {generated_signals.max():.3f}")
    else:
        print(f"Keeping normalized range [-1, 1]")
    
    return generated_signals



def convert_normalized_to_fhr(normalized_signal, min_bpm=50, max_bpm=200):
    """
    Convert normalized signal [-1, 1] back to FHR values [min_bpm, max_bpm]
    """
    signal_01 = (normalized_signal + 1) / 2
    fhr_signal = signal_01 * (max_bpm - min_bpm) + min_bpm
    return fhr_signal



def create_individual_csv(signal, output_filename, is_normalized=False, sampling_rate_hz=4):
    """
    Create a CSV file in the same format as real FHR data
    """
    interval_seconds = 1.0 / sampling_rate_hz
    
    elapsed_times = []
    for i in range(len(signal)):
        total_seconds = i * interval_seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        time_str = f"{minutes}:{seconds:06.3f}"
        elapsed_times.append(time_str)
    
    np.random.seed(42 + len(signal))
    uc_values = np.random.normal(10, 5, len(signal))
    uc_values = np.clip(uc_values, 0, 50)
    
    if is_normalized:
        fhr_column_name = 'FHR_normalized'
    else:
        fhr_column_name = 'FHR'
    
    data = {
        'Elapsed time': elapsed_times,
        fhr_column_name: signal,
        'UC': uc_values
    }
    df = pd.DataFrame(data)
    
    with open(output_filename, 'w') as f:
        if is_normalized:
            f.write("'Elapsed time','FHR_normalized','UC'\n")
            f.write("'hh:mm:ss.mmm','normalized','nd'\n")
        else:
            f.write("'Elapsed time','FHR','UC'\n")
            f.write("'hh:mm:ss.mmm','bpm','nd'\n")
        
        for i, row in df.iterrows():
            if is_normalized:
                f.write(f"'{row['Elapsed time']}',{row[fhr_column_name]:.6f},{row['UC']:.3f}\n")
            else:
                f.write(f"'{row['Elapsed time']}',{row[fhr_column_name]:.3f},{row['UC']:.3f}\n")



def plot_generated_signals(folder_name, signal_ids, is_normalized=False, save_plots=False):
    """
    Plot specific generated signals
    """
    print(f"\n📊 Plotting signals from {folder_name}...")
    
    n_plots = len(signal_ids)
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
    else:
        rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
    
    if is_normalized:
        fhr_col_name = "FHR_normalized"
        y_label = "FHR (normalized)"
        y_limits = [-1, 1]
    else:
        fhr_col_name = "FHR"
        y_label = "FHR (bpm)"
        y_limits = [50, 200]
    
    for i, signal_id in enumerate(signal_ids):
        if i >= len(axes):
            break
            
        csv_file = os.path.join(folder_name, f"{signal_id}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} not found")
            continue
        
        try:
            df = pd.read_csv(csv_file, skiprows=2, header=None, usecols=[1], names=[fhr_col_name])
            signal_values = df[fhr_col_name].values
            
            label_name = "Unknown"
            if "pathological" in folder_name.lower():
                label_name = "Pathological (Synthetic)"
                bg_color = '#fff0f0'
            elif "normal" in folder_name.lower():
                label_name = "Normal (Synthetic)"
                bg_color = '#f0fff0'
            else:
                bg_color = '#f0f0f0'
            
            title = f'Generated Signal {signal_id}\n{label_name}'
            
            axes[i].plot(range(len(signal_values)), signal_values, 'b-', linewidth=1.0)
            axes[i].set_title(title, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel(y_label)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(y_limits)
            axes[i].set_facecolor(bg_color)
            
            mean_val = np.mean(signal_values)
            std_val = np.std(signal_values)
            axes[i].text(0.02, 0.98, f'Mean: {mean_val:.3f}±{std_val:.3f}', 
                        transform=axes[i].transAxes, va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            print(f"✅ Plotted {signal_id}.csv")
            
        except Exception as e:
            print(f"Error plotting {signal_id}.csv: {e}")
    
    for j in range(len(signal_ids), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_plots:
        plot_filename = f"generated_signals_plot_{folder_name.replace('/', '_')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"💾 Plot saved as: {plot_filename}")
    
    plt.show()
    
    return fig



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # IMPORTANT: Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory before starting: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Model parameters
    latent_dim = 100
    num_classes = 2
    sequence_length = 1000
    
    # Load the trained generator
    generator = Generator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        sequence_length=sequence_length
    ).to(device)
    
    # Path to your saved model
    model_path = input("Enter path to generator checkpoint: ").strip()
    
    if not os.path.exists(model_path):
        raise Exception(f"Model file not found: {model_path}")
    
    # ═══════════════════════════════════════════════════════════════════
    # UPDATED LOADING LOGIC - Handles both old and new checkpoint formats
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n🔄 Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # NEW FORMAT: Full checkpoint with training state
        generator.load_state_dict(checkpoint['model_state_dict'])
        epoch_info = checkpoint.get('epoch', 'unknown')
        print(f"✅ Loaded NEW format checkpoint from epoch {epoch_info}")
        print(f"   Checkpoint size: ~2.4 GB (includes optimizer state)")
        
        # Optional: Show training history if available
        if 'history' in checkpoint:
            history = checkpoint['history']
            if history and len(history.get('g_loss', [])) > 0:
                print(f"   Final G loss: {history['g_loss'][-1]:.4f}")
                print(f"   Final D loss: {history['d_loss'][-1]:.4f}")
    
    elif isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        # ALTERNATIVE NEW FORMAT: Some scripts use 'generator_state_dict'
        generator.load_state_dict(checkpoint['generator_state_dict'])
        epoch_info = checkpoint.get('epoch', 'unknown')
        print(f"✅ Loaded NEW format checkpoint from epoch {epoch_info}")
    
    else:
        # OLD FORMAT: Direct state dict (weights only)
        generator.load_state_dict(checkpoint)
        print(f"✅ Loaded OLD format checkpoint (weights only)")
        print(f"   Checkpoint size: ~800 MB")
    
    generator.eval()
    print(f"✅ Model loaded from: {model_path}")
    print(f"   Device: {device}")
    # ═══════════════════════════════════════════════════════════════════
    
    # Output format selection
    output_format = input("\nOutput format - (1) BPM [50-200] or (2) Normalized [-1,1]? Enter 1 or 2: ").strip()
    keep_normalized = (output_format == '2')
    
    if keep_normalized:
        print("📊 Output will be NORMALIZED in range [-1, 1]")
    else:
        print("📊 Output will be in BPM range [50, 200]")
    
    # User input
    label = int(input("\nPlease enter the label (0=pathological, 1=normal): "))
    while label not in [0, 1]:
        label = int(input("Invalid input. Please enter 0 for pathological or 1 for normal: "))
    
    num_signals = int(input("Please enter the number of signals to generate: "))
    while num_signals <= 0:
        num_signals = int(input("Invalid input. Please enter a positive number: "))
    
    # CRITICAL: Ask for batch size
    default_batch_size = 8 if num_signals > 10 else num_signals
    batch_size_input = input(f"Batch size for generation (default: {default_batch_size}, use smaller if OOM): ").strip()
    if batch_size_input:
        batch_size = int(batch_size_input)
    else:
        batch_size = default_batch_size
    
    print(f"⚠️ Will generate in batches of {batch_size} to avoid OOM errors")
    
    # Folder setup
    label_name = "pathological" if label == 0 else "normal"
    format_suffix = "_normalized" if keep_normalized else "_bpm"
    default_folder = f"synthetic_{label_name}{format_suffix}"
    folder_name = input(f"Enter folder name (default: {default_folder}): ").strip()
    if not folder_name:
        folder_name = default_folder
    
    # ID range
    if label == 0:
        default_start = 11000
        print(f"Suggestion for pathological: 11000-11999")
    else:
        default_start = 21000
        print(f"Suggestion for normal: 21000-21999")
    
    start_id = input(f"Enter starting ID (default: {default_start}): ").strip()
    if not start_id:
        start_id = default_start
    else:
        start_id = int(start_id)
    
    print(f"\n📋 Configuration:")
    print(f"   Label: {label} ({label_name})")
    print(f"   Format: {'Normalized [-1,1]' if keep_normalized else 'BPM [50-200]'}")
    print(f"   Folder: {folder_name}")
    print(f"   IDs: {start_id} to {start_id + num_signals - 1}")
    print(f"   Batch size: {batch_size}")
    
    confirm = input("\nProceed with generation? (y/n): ").lower()
    if confirm != 'y':
        print("Generation cancelled.")
        exit()
    
    # Generate signals
    print(f"\n🔄 Generating {num_signals} signals with label {label}...")
    generated_signals = generate_signals(
        generator, label, num_signals, latent_dim, num_classes, device, 
        keep_normalized=keep_normalized,
        batch_size=batch_size  # CRITICAL: Pass batch size
    )
    print(f"Generated signals shape: {generated_signals.shape}")
    
    # Create output directory
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Created folder: {folder_name}")
    
    # Save each signal
    print(f"\n💾 Saving individual CSV files...")
    for i in range(num_signals):
        current_id = start_id + i
        signal = generated_signals[i]
        signal = np.round(signal, 6 if keep_normalized else 3)
        
        csv_filename = os.path.join(folder_name, f"{current_id}.csv")
        create_individual_csv(signal, csv_filename, is_normalized=keep_normalized)
        
        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{num_signals} files...")
    
    # Summary and info files
    summary_filename = os.path.join(folder_name, f"summary_label{label}_num{num_signals}.csv")
    np.savetxt(summary_filename, generated_signals, delimiter=",")
    
    info_filename = os.path.join(folder_name, "generation_info.txt")
    with open(info_filename, 'w') as f:
        f.write(f"Synthetic FHR Data Generation Info\n")
        f.write(f"===================================\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write(f"Label: {label} ({label_name})\n")
        f.write(f"Output format: {'Normalized [-1,1]' if keep_normalized else 'BPM [50-200]'}\n")
        f.write(f"Number of signals: {num_signals}\n")
        f.write(f"Batch size used: {batch_size}\n")
        f.write(f"ID range: {start_id} - {start_id + num_signals - 1}\n")
        f.write(f"Model: {model_path}\n")
    
    print(f"\n✅ Successfully generated {num_signals} CSV files in {folder_name}/")
    
    # Plotting
    plot_signals = input(f"\n📊 Plot some generated signals? (y/n): ").lower()
    if plot_signals == 'y':
        suggested_ids = [start_id + i for i in range(min(5, num_signals))]
        print(f"Suggested IDs: {suggested_ids}")
        
        ids_input = input(f"Enter IDs to plot (comma-separated, or Enter for suggested): ").strip()
        if ids_input:
            try:
                plot_ids = [int(x.strip()) for x in ids_input.split(',')]
            except:
                print("Invalid input, using suggested IDs")
                plot_ids = suggested_ids
        else:
            plot_ids = suggested_ids
        
        save_plot = input("Save plots as PNG? (y/n): ").lower() == 'y'
        plot_generated_signals(folder_name, plot_ids, is_normalized=keep_normalized, save_plots=save_plot)
    
    # Clear GPU memory at the end
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n🧹 GPU Memory after completion: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    print("\n🎉 Generation complete!")
