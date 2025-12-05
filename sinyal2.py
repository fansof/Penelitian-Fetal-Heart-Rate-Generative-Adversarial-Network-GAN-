import wfdb  # WaveForm-Database package
import pandas as pd
import numpy as np
import glob
import os

# Get list of all .dat files in the folder
dat_files = glob.glob("home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/inidata/*.dat")

# Write the list of files to a CSV
df = pd.DataFrame(dat_files)
df.to_csv("files_list.csv", index=False, header=None)

# Read back the file list
files = pd.read_csv("files_list.csv", header=None)

for i in range(len(files)):
    record_path = files.iloc[i, 0]
    recordname = os.path.basename(record_path)  # Get filename
    recordname_new = os.path.splitext(recordname)[0]  # Remove .dat extension
    
    print(f"Processing: {recordname_new}")

    try:
        record, _ = wfdb.rdsamp(f"home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/inidata/{recordname_new}")
        record_array = np.asarray(record)

        # Save to CSV
        output_path = f"{recordname_new}.csv"
        np.savetxt(output_path, record_array, delimiter=",")

        print(f"File {i+1}/{len(files)} done: {output_path}")

    except Exception as e:
        print(f"Error processing {recordname_new}: {e}")

print("\nAll files done!")
