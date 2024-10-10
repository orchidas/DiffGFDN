import pickle
from pathlib import Path

import h5py
from loguru import logger

# This script converts the .mat file to pickle format which can be read much faster by Python
logger.info("Reading mat file")

# Load the MATLAB v7.3 .mat file using h5py
file_path = Path("../resources/Georg_3room_FDTD/srirs.mat").resolve()

with h5py.File(file_path, 'r') as mat_file:
    # Get the dataset
    srir_mat = mat_file['srirDataset']
    sample_rate = srir_mat['fs'][:]
    source_position = srir_mat['srcPos'][:]
    receiver_position = srir_mat['rcvPos'][:]
    # these are second order ambisonic signals
    # I am guessing the first channel contains the W component
    srirs = srir_mat['srirs'][:]

# load the common slopes from the other mat files
file_path = Path(
    "../resources/Georg_3room_FDTD/Common_Slope_Analysis_Results/")
filename = 'cs_analysis_results_omni'
freqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
common_t60 = []
amplitudes = []

for i in range(len(freqs)):
    full_path = file_path / f'{filename}_{freqs[i]}.mat'
    with h5py.File(full_path.resolve(), 'r') as mat_file:
        data = mat_file['analysisResults']
        common_t60.append(data['commonDecayTimes'][:])
        amplitudes.append(data['aVals'][:])

# Convert the list to a NumPy array if needed
data_dict = {
    'fs': sample_rate,
    'srcPos': source_position,
    'rcvPos': receiver_position,
    'srirs': srirs,
    'band_centre_hz': freqs,
    'common_decay_times': common_t60,
    'amplitudes': amplitudes,
}

# Specify the output pickle file path
pickle_file_path = Path("../resources/Georg_3room_FDTD/srirs.pkl").resolve()

# Write the data to a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(data_dict, pickle_file)

logger.info("Saved pickle file")
