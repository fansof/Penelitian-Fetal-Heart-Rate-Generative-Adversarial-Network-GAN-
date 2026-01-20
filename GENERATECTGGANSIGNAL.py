
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from ctgganmodel import Generator1000 as Generator 

def generate_signals(generator, label, num_signals, input_dim):
    with torch.no_grad():
        test_noise = torch.randn(1, 100, 1).to(device)
        test_label = torch.tensor([1]).to(device)
        raw_output = generator(test_noise, test_label)
        print(f"Raw generator output range: {raw_output.min():.3f} to {raw_output.max():.3f}")
        print(f"Raw generator output mean: {raw_output.mean():.3f}")
        noise = torch.randn(num_signals, input_dim, 1).to(device)
        labels_tensor = torch.full((num_signals,), label, dtype=torch.long).to(device)
        generated_signals = generator(noise, labels_tensor).squeeze().cpu().numpy()
    return generated_signals

def convert_normalized_to_fhr(normalized_signal, min_bpm=50, max_bpm=200):
    """
    Convert normalized signal [-1, 1] back to FHR values [min_bpm, max_bpm]
    This reverses the MinMaxScaler normalization
    """
    # Convert from [-1, 1] to [0, 1]
    # signal_01 = (normalized_signal + 1) / 2
    
    # Convert from [0, 1] to [min_bpm, max_bpm]
    # fhr_signal = signal_01 * (max_bpm - min_bpm) + min_bpm
    fhr_signal = normalized_signal

    return fhr_signal

def create_individual_csv(fhr_signal, output_filename, sampling_rate_hz=4):
    """
    Create a CSV file in the same format as real FHR data (like 1001.csv)
    
    Args:
        fhr_signal: 1D array of FHR values
        output_filename: Output CSV filename
        sampling_rate_hz: Sampling rate (4Hz = 0.25s intervals)
    """
    # Calculate time intervals
    interval_seconds = 1.0 / sampling_rate_hz  # 0.25 seconds for 4Hz
    
    # Create time stamps
    elapsed_times = []
    for i in range(len(fhr_signal)):
        total_seconds = i * interval_seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        
        # Format as mm:ss.mmm
        time_str = f"{minutes}:{seconds:06.3f}"
        elapsed_times.append(time_str)
    
    # Generate dummy UC (Uterine Contraction) values
    # Real UC values are typically 0-100, we'll generate realistic random values
    np.random.seed(42)  # For reproducible UC values
    uc_values = np.random.normal(10, 5, len(fhr_signal))  # Mean=10, std=5
    uc_values = np.clip(uc_values, 0, 50)  # Clip to reasonable range
    
    # Create DataFrame
    data = {
        'Elapsed time': elapsed_times,
        'FHR': fhr_signal,
        'UC': uc_values
    }
    df = pd.DataFrame(data)
    
    # Create the CSV with proper headers (same as real data)
    with open(output_filename, 'w') as f:
        # Write headers exactly like real CSV
        f.write("'Elapsed time','FHR','UC'\n")
        f.write("'hh:mm:ss.mmm','bpm','nd'\n")
        
        # Write data
        for i, row in df.iterrows():
            f.write(f"'{row['Elapsed time']}',{row['FHR']:.3f},{row['UC']:.3f}\n")
    
    print(f"Created {output_filename} with {len(fhr_signal)} samples")

def plot_generated_signals(folder_name, signal_ids, save_plots=False):
    """
    Plot specific generated signals (no pH labels needed)
    
    Args:
        folder_name: Folder containing the generated CSV files
        signal_ids: List of signal IDs to plot (e.g., [11000, 11001, 11002])
        save_plots: Whether to save plots as PNG files
    """
    print(f"\n📊 Plotting signals from {folder_name}...")
    
    # Create subplots
    n_plots = len(signal_ids)
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
    else:
        rows = (n_plots + 1) // 2  # 2 plots per row
        fig, axes = plt.subplots(rows, 2, figsize=(15, 4 * rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
    
    for i, signal_id in enumerate(signal_ids):
        if i >= len(axes):
            break
            
        csv_file = os.path.join(folder_name, f"{signal_id}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: File {csv_file} not found, skipping...")
            axes[i].text(0.5, 0.5, f'File {signal_id}.csv\nNot Found', 
                        ha='center', va='center', transform=axes[i].transAxes)
            continue
        
        try:
            # Read the generated CSV (same format as real data)
            df = pd.read_csv(csv_file, skiprows=2, header=None, usecols=[1], names=["FHR"])
            fhr_values = df["FHR"].values
            
            # Determine label from folder or signal structure
            label_name = "Unknown"
            if "abnormal" in folder_name.lower():
                label_name = "Abnormal (Synthetic)"
                bg_color = '#fff0f0'  # Light red
            elif "normal" in folder_name.lower():
                label_name = "Normal (Synthetic)"
                bg_color = '#f0fff0'  # Light green
            else:
                bg_color = '#f0f0f0'  # Light gray
            
            title = f'Generated Signal {signal_id}\n{label_name}'
            
            # Plot
            axes[i].plot(range(len(fhr_values)), fhr_values, 'b-', linewidth=1.0)
            axes[i].set_title(title, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('FHR (bpm)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim([-1, 1])
            axes[i].set_facecolor(bg_color)
            
            # Add some statistics
            mean_fhr = np.mean(fhr_values)
            std_fhr = np.std(fhr_values)
            axes[i].text(0.02, 0.98, f'Mean: {mean_fhr:.1f}±{std_fhr:.1f}', 
                        transform=axes[i].transAxes, va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            print(f"✅ Plotted {signal_id}.csv - Length: {len(fhr_values)} samples, Mean FHR: {mean_fhr:.1f}")
            
        except Exception as e:
            print(f"Error plotting {signal_id}.csv: {e}")
            axes[i].text(0.5, 0.5, f'Error loading\n{signal_id}.csv', 
                        ha='center', va='center', transform=axes[i].transAxes)
    
    # Hide unused subplots
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

    # Model parameters (adjust these as per your model's architecture)
    input_dim = 100
    output_dim = 1

    # Load the best saved model
    generator = Generator(2).to(device)
    # checkpoint_dir = "./checkpoints2/"
    model_path = os.path.join("/path/to/final_G.pth")

    if not os.path.exists(model_path):
        raise Exception(f"Model file not found: {model_path}")

    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    # User input for label and number of signals
    label = int(input("Please enter the label (0=abnormal, 1=normal): "))
    while label not in [0, 1]:
        label = int(input("Invalid input. Please enter 0 for abnormal or 1 for normal: "))

    num_signals = int(input("Please enter the number of signals to generate: "))
    while num_signals <= 0:
        num_signals = int(input("Invalid input. Please enter a positive number: "))

    # User input for folder and naming
    label_name = "abnormal" if label == 0 else "normal"
    default_folder = f"synthetic_{label_name}"
    folder_name = input(f"Enter folder name (default: {default_folder}): ").strip()
    if not folder_name:
        folder_name = default_folder

    # User input for ID range
    if label == 0:
        default_start = 11000
        print(f"Suggestion for abnormal: 11000-11999")
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
    print(f"   Folder: {folder_name}")
    print(f"   IDs: {start_id} to {start_id + num_signals - 1}")
    print(f"   Number of files: {num_signals}")
    print(f"   Purpose: Testing only (no pH labels)")
    
    confirm = input("\nProceed with generation? (y/n): ").lower()
    if confirm != 'y':
        print("Generation cancelled.")
        exit()

    # Generate signals
    print(f"\n🔄 Generating {num_signals} signals with label {label}...")
    generated_signals = generate_signals(generator, label, num_signals, input_dim)
    print(f"Generated signals shape: {generated_signals.shape}")
    print(f"Generated signals range: {generated_signals.min():.3f} to {generated_signals.max():.3f}")
    print(f"Generated signals mean: {generated_signals.mean():.3f}")
    
    # Create output directory
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📁 Created folder: {folder_name}")
    
    # Convert each signal and save as individual CSV
    generated_files = []
    
    for i in range(num_signals):
        # Calculate current ID
        current_id = start_id + i
        
        # Get the normalized signal for this sample
        normalized_signal = generated_signals[i] if num_signals > 1 else generated_signals
        
        # Convert back to FHR values
        fhr_signal = convert_normalized_to_fhr(normalized_signal)
        print(f"Converted FHR range: {fhr_signal.min():.3f} to {fhr_signal.max():.3f}")
        print(f"First 10 FHR values: {fhr_signal[:10]}")
        
        # Round to reasonable precision
        fhr_signal = np.round(fhr_signal, 3)
        
        # Create individual CSV filename with custom ID
        csv_filename = os.path.join(folder_name, f"{current_id}.csv")
        generated_files.append(csv_filename)
        
        # Create the CSV file
        create_individual_csv(fhr_signal, csv_filename)
    
    # Create summary CSV with all signals (for reference)
    summary_filename = os.path.join(folder_name, f"summary_label{label}_num{num_signals}.csv")
    np.savetxt(summary_filename, generated_signals, delimiter=",")
    
    # Create a simple info file about the generation
    info_filename = os.path.join(folder_name, "generation_info.txt")
    with open(info_filename, 'w') as f:
        f.write(f"Synthetic FHR Data Generation Info\n")
        f.write(f"===================================\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")
        f.write(f"Label: {label} ({label_name})\n")
        f.write(f"Number of signals: {num_signals}\n")
        f.write(f"ID range: {start_id} - {start_id + num_signals - 1}\n")
        f.write(f"Purpose: Testing only (no pH labels)\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"FHR range: 50-200 bpm\n")
        f.write(f"Sequence length: 3600 samples\n")
        f.write(f"Sampling rate: 4Hz (0.25s intervals)\n")
    
    print(f"\n✅ Successfully generated {num_signals} individual CSV files!")
    print(f"📁 Files saved in: {folder_name}/")
    print(f"📊 Generated files:")
    for i, file_path in enumerate(generated_files[:5]):  # Show first 5 files
        print(f"   - {os.path.basename(file_path)}")
    if len(generated_files) > 5:
        print(f"   - ... and {len(generated_files) - 5} more files")
    
    print(f"📋 Summary file: {summary_filename}")
    print(f"ℹ️  Info file: {info_filename}")
    
    print(f"\n🎯 Ready for CNN testing!")
    print(f"💡 Usage for testing:")
    print(f"   # Load your trained model")
    print(f"   # Make predictions on synthetic files:")
    print(f"   for csv_file in os.listdir('{folder_name}'):")
    print(f"       if csv_file.endswith('.csv') and 'summary' not in csv_file:")
    print(f"           # Process {folder_name}/{{csv_file}}")  # Fixed: Use {{csv_file}} to escape the braces
    
    # Ask if user wants to plot some signals
    plot_signals = input(f"\n📊 Plot some generated signals? (y/n): ").lower()
    if plot_signals == 'y':
        # Suggest some IDs to plot
        suggested_ids = [start_id + i for i in range(min(5, num_signals))]
        print(f"Suggested IDs to plot: {suggested_ids}")
        
        ids_input = input(f"Enter signal IDs to plot (comma-separated, or press Enter for suggested): ").strip()
        if ids_input:
            try:
                plot_ids = [int(x.strip()) for x in ids_input.split(',')]
            except:
                print("Invalid input, using suggested IDs")
                plot_ids = suggested_ids
        else:
            plot_ids = suggested_ids
        
        save_plot = input("Save plots as PNG? (y/n): ").lower() == 'y'
        
        # Plot the signals
        plot_generated_signals(folder_name, plot_ids, save_plots=save_plot)
