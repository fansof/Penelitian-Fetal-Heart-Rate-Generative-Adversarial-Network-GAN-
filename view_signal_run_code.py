# Cell 1: Import the viewer
from view_signal_notebook import view_signal, view_multiple_signals, view_signal_windows

# Cell 2: Set your data folder
DATA_FOLDER = "/home/fauzi/Documents/generateGAN_untukSKRIPSI/1APREPROCESSEDFILE/PREPROCESSED_OKTOBERv2"
# DATA_FOLDER =  "/home/fauzi/Documents/SKRIPSI AAMIIN/wfdbpy/fhrdataNEW"
# Cell 3: View a single signal (basic)
signal = view_signal(DATA_FOLDER, "2040.csv")

# Cell 4: View with sequence_length=1000 (exactly as dataloader sees it)
signal = view_signal(DATA_FOLDER, "1108.csv", sequence_length=1000)

# Cell 5: View multiple signals in a grid
files = ["1470.csv", "2013.csv", "1158.csv", "1044.csv"]
signals = view_multiple_signals(DATA_FOLDER, files, sequence_length=4000)

#plot 1188 untuk normal ph 715
#plot 2001 untuk abnormal ph 715

signal_4000 = view_signal_windows(
    DATA_FOLDER,
    "1466.csv",
    fs=4,
    sequence_length=4000,   # sama seperti base_dataset
    win_len=1000,           # sama seperti WindowedFromLongDataset
    crop_strategy='last'    # samakan dengan yang dipakai di training
)