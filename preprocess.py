# %% imports
# standard libs
from pathlib import Path
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import argparse

# local
import preprocess_functions

# %% main
# user inputs
parser = argparse.ArgumentParser(description='extract IS-RC data and preprocess')
parser.add_argument('-data_loc'  , type=str)
parser.add_argument('-target_loc', type=str, default = "preprocessed//")
args = parser.parse_args()

max_epochs = 1792

# get all edf files
file_names = natsorted(glob(f"{args.data_loc}//*.edf"))
no_files = len(file_names)

# create output folder
Path(args.target_loc).mkdir(parents=True, exist_ok=True)

# loop over all files
for i,file in enumerate(tqdm(file_names)):
    # get the label files
    STA_file = file.replace('edf','STA')
    scoring, lag = preprocess_functions.extract_labels(STA_file,max_epochs,i)
    
    #extract edf data
    PSG = preprocess_functions.extract_edf(file,max_epochs, lag)
    
    #scale
    PSG = preprocess_functions.scale(PSG)
    
    # cast data to correct type
    PSG = np.float32(PSG)
    scoring = np.int64(scoring)
    
    # save the data
    save_name = f"{args.target_loc}//IS-RC_{i}.npz"
    np.savez(save_name, PSG = PSG, scoring = scoring)
    
    
    