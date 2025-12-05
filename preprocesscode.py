import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline

def preprocess_and_save_full_sequences(
    data_folder='/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW',
    output_folder='./PreprocessedOKTOBER',
    ph_file='/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW/ph_labels.csv',
    min_bpm=50,
    max_bpm=200,
    fs=4,
    zero_sec_thresh=15
):
    """
    Preprocess all FHR sequences following your exact pipeline:
    1) Fill zero-gaps ≤15s with spline; delete zero-gaps >15s
    2) Spike fix: if |Δ| >25 bpm, interpolate to stable section
    3) Replace <50 or >200 bpm by cubic spline interpolation
    4) Normalize to [-1, 1] using tanh-range mapping
    
    Save FULL LENGTH preprocessed signals (no sequence_length truncation).
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    zero_len_thresh = int(fs * zero_sec_thresh)  # 60 samples at 4Hz
    
    # Load pH labels
    ph_df = pd.read_csv(ph_file)
    ph_df["Label"] = (ph_df["pH"] >= 7.15).astype(int)
    ph_dict = dict(zip(ph_df["Record"].astype(str), ph_df["Label"]))
    
    # Get all CSV files
    all_files = sorted([
        f for f in os.listdir(data_folder)
        if f.endswith('.csv') and f != os.path.basename(ph_file)
    ])
    
    print(f"Preprocessing {len(all_files)} files...")
    successful = 0
    failed = 0
    
    for file_name in tqdm(all_files):
        try:
            # Load raw signal (skip 2 header rows, FHR at column 1)
            fpath = os.path.join(data_folder, file_name)
            df = pd.read_csv(fpath, skiprows=2, header=None, usecols=[1], names=["FHR"])
            raw = df["FHR"].values.astype(np.float32)
            
            if len(raw) == 0:
                print(f"Skipping {file_name}: empty signal")
                failed += 1
                continue
            
            # === PREPROCESSING PIPELINE ===
            
            # Step 1: Fix zeros
            x = _fix_zeros(raw, zero_len_thresh, min_bpm, max_bpm)
            if x is None or len(x) == 0:
                print(f"Skipping {file_name}: failed zero fixing")
                failed += 1
                continue
            
            # Step 2: Fix spikes
            x = _fix_spikes(x, min_bpm, max_bpm)
            if x is None or len(x) == 0:
                print(f"Skipping {file_name}: failed spike fixing")
                failed += 1
                continue
            
            # Step 3: Replace out-of-range values
            x = _replace_out_of_range(x, min_bpm, max_bpm)
            if x is None or len(x) == 0:
                print(f"Skipping {file_name}: failed range fixing")
                failed += 1
                continue
            
            # Step 4: Normalize to [-1, 1] (tanh-range)
            normalized = _normalize_tanh(x, min_bpm, max_bpm)
            
            # Save as CSV
            output_path = Path(output_folder) / file_name
            pd.DataFrame({'FHR_preprocessed': normalized}).to_csv(output_path, index=False)
            successful += 1
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            failed += 1
            continue
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {output_folder}")

# === Helper Functions (matching your code exactly) ===

def _fix_zeros(self, x, keep_mask):
        """
        Fix zero-gaps: short gaps (≤15s) are filled by cubic spline, long gaps are deleted.

        Args:
            x: FHR signal array
            keep_mask: Boolean mask of indices to keep

        Returns:
            Tuple of (fixed signal, updated keep_mask) or (None, None) if salvage fails
        """
        zero_mask = (x == 0)
        if not np.any(zero_mask):
            return x, keep_mask

        # Find zero runs
        dif = np.diff(zero_mask.astype(np.int8))
        starts = np.where(dif == 1)[0] + 1
        ends = np.where(dif == -1)[0] + 1

        # Handle edge cases
        if zero_mask[0]:
            starts = np.insert(starts, 0, 0)
        if zero_mask[-1]:
            ends = np.append(ends, len(zero_mask))

        x = x.copy()

        for s, e in zip(starts, ends):
            run_len = e - s

            if run_len <= self.zero_len_thresh:
                # Short gap: fill by cubic spline
                left = s - 1
                right = e

                # Find valid left and right anchors
                while left >= 0 and (x[left] == 0 or np.isnan(x[left])):
                    left -= 1
                while right < len(x) and (x[right] == 0 or np.isnan(x[right])):
                    right += 1

                if left >= 0 and right < len(x):
                    # Interpolate using cubic spline
                    xi = np.array([left, right], dtype=float)
                    yi = np.array([x[left], x[right]], dtype=float)
                    cs = CubicSpline(xi, yi, bc_type='clamped')
                    xs = np.arange(left, right + 1)
                    vals = cs(xs)
                    vals = np.clip(vals, self.min_bpm, self.max_bpm)
                    x[left:right + 1] = vals
                else:
                    # Cannot bridge: use local median
                    local = x[max(0, s-50):min(len(x), e+50)]
                    med = np.median(local[local > 0]) if np.any(local > 0) else 140.0
                    x[s:e] = med
            else:
                # Long gap: mark for deletion
                keep_mask[s:e] = False

        # Apply deletion mask
        x = x[keep_mask]

        return (x, keep_mask) if len(x) > 0 else (None, None)


def _fix_spikes(self, x):
        # We will iteratively smooth spikes by interpolation over [i0, i_stable]
        x = x.copy()
        i = 1
        while i < len(x):
            delta = abs(x[i] - x[i-1])
            if delta > 25:
                # find first stable section: 5 consecutive diffs < 10
                j = i + 1
                consec = 0
                while j < len(x):
                    d = abs(x[j] - x[j-1])
                    if d < 10:
                        consec += 1
                        if consec >= 5:
                            # stable section starts at j-4 (first of the 5)
                            i_stable = j - 4
                            break
                    else:
                        consec = 0
                    j += 1
                else:
                    # No stable section found; bridge to last sample
                    i_stable = len(x) - 1

                i0 = max(0, i-1)
                # interpolate from x[i0] at i0 to x[i_stable] at i_stable
                xi = np.array([i0, i_stable], dtype=float)
                yi = np.array([x[i0], x[i_stable]], dtype=float)
                if i_stable - i0 >= 2:
                    cs = CubicSpline(xi, yi, bc_type='clamped')
                    xs = np.arange(i0, i_stable + 1)
                    vals = np.clip(cs(xs), self.min_bpm, self.max_bpm)
                    x[i0:i_stable + 1] = vals
                    i = i_stable + 1
                    continue
                else:
                    # If too close, just move on
                    i += 1
                    continue
            i += 1
        return x

def _replace_out_of_range(self, x):
    x = x.copy()
    bad = (x < self.min_bpm) | (x > self.max_bpm) | np.isnan(x)
    if not np.any(bad):
        return x

    # Interpolate all valid points using cubic spline over their indices
    idx_all = np.arange(len(x))
    idx_good = idx_all[~bad]
    if len(idx_good) < 2:
        return None
    cs = CubicSpline(idx_good.astype(float), x[~bad].astype(float), bc_type='clamped')
    x[bad] = np.clip(cs(idx_all[bad].astype(float)), self.min_bpm, self.max_bpm)
    return x

def _normalize_tanh(x, min_bpm, max_bpm):
    """Normalize to [-1, 1] range (tanh-style)."""
    return np.clip(2 * (x - min_bpm) / (max_bpm - min_bpm) - 1, -1.0, 1.0)

# === RUN PREPROCESSING ===
preprocess_and_save_full_sequences(
    data_folder='/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW/',
    output_folder='./PREPOCESSED_OKTOBER',
    ph_file='/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW/ph_labels.csv'
)
