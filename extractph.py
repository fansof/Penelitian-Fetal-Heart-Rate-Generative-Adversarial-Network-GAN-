import os
import re
import pandas as pd

# Path to your dataset folder
folder_path = r"D:\SKRIPSI\ctu-chb-intrapartum-cardiotocography-database-1.0.0"

# Dictionary to store extracted pH values
ph_values = {}

# Loop through all .hea files
for file in os.listdir(folder_path):
    if file.endswith(".hea"):
        file_path = os.path.join(folder_path, file)
        
        # Read the .hea file
        with open(file_path, "r") as f:
            content = f.read()

        # Extract pH value using regex
        match = re.search(r"#pH\s+([\d.]+)", content)
        if match:
            ph_value = float(match.group(1))
            record_name = file.replace(".hea", "")
            ph_values[record_name] = ph_value

# Convert to DataFrame
df = pd.DataFrame(list(ph_values.items()), columns=["Record", "pH"])

# Sort DataFrame numerically by Record
df["Record"] = df["Record"].astype(int)  # Convert to integer for correct sorting
df = df.sort_values("Record")

# Save to CSV
df.to_csv(os.path.join(folder_path, "ph_labels.csv"), index=False)

print(f"Extracted pH values for {len(ph_values)} records. Saved to ph_labels.csv.")
