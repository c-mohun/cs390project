from nilearn import plotting, image, input_data, datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
from nilearn.connectome import ConnectivityMeasure

# Download and fetch the atlas
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_filename = atlas['maps']

# File paths (update these based on your directory structure)
func_file = 'sub-20900/ses-v1/func/sub-20900_ses-v1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
confounds_file = 'sub-20900/ses-v1/func/sub-20900_ses-v1_task-rest_run-1_desc-confounds_timeseries.tsv'

# Load and clean confounds
confounds = pd.read_csv(confounds_file, sep='\t')
confound_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']  # or use 'a_comp_cor_XX'

# Define masker for your network (e.g., DMN mask or atlas-based)
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

# Extract time series
timeseries = masker.fit_transform(func_file, confounds=confounds[confound_vars])

# Compute correlation matrix
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([timeseries])[0]

# **Normalize correlation matrix to range [0,1]**
scaler = MinMaxScaler(feature_range=(0,1))
correlation_matrix_scaled = scaler.fit_transform(correlation_matrix)

# Plot scaled connectivity matrix
plotting.plot_matrix(correlation_matrix_scaled, figure=(10, 8), labels=None, 
                      colorbar=True, vmax=0.2, vmin=0, cmap='coolwarm', title='Functional Connectivity (Scaled 0-1)')

# Plot connectome
coords = plotting.find_parcellation_cut_coords(atlas_filename)
plotting.plot_connectome(correlation_matrix, coords, edge_threshold='80%')

plt.show()
