from nilearn import plotting, image, input_data, datasets
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# -------------------------------
# 1. Load Atlas
# -------------------------------
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_filename = atlas['maps']
print("✅ Atlas loaded")

# -------------------------------
# 2. File Paths
# -------------------------------
func_file = 'sub-20900/ses-v1/func/sub-20900_ses-v1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
confounds_file = 'sub-20900/ses-v1/func/sub-20900_ses-v1_task-rest_run-1_desc-confounds_timeseries.tsv'

# -------------------------------
# 3. Load Confounds
# -------------------------------
confounds = pd.read_csv(confounds_file, sep='\t')
confound_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
print("✅ Confounds loaded")

# -------------------------------
# 4. Extract Time Series
# -------------------------------
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
timeseries = masker.fit_transform(func_file, confounds=confounds[confound_vars])
print("✅ Timeseries extracted")

# -------------------------------
# 5. Compute Functional Connectivity
# -------------------------------
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
print("✅ Correlation matrix computed")

# -------------------------------
# 6. Plot Heatmap & Connectome
# -------------------------------
plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=None, colorbar=True, vmax=1, vmin=-1, title='Functional Connectivity')
coords = plotting.find_parcellation_cut_coords(atlas_filename)
plotting.plot_connectome(correlation_matrix, coords, edge_threshold='80%')
plt.show()
print("✅ Functional connectivity plotted")

# -------------------------------
# 7. Extract Graph Features
# -------------------------------
G = nx.from_numpy_array(np.abs(correlation_matrix))  # build graph from absolute correlations
strengths = dict(G.degree(weight='weight'))          # node strength = sum of edge weights
features = list(strengths.values())                  # 100 features (one per node)
X = np.array(features).reshape(1, -1)
print("✅ Graph features extracted")

# -------------------------------
# 8. Regression Model
# -------------------------------
shaps_score = 5  # replace this with the real SHAPS score for this subject
model = LinearRegression()
model.fit(X, [shaps_score])
predicted = model.predict(X)[0]

print("✅ Linear regression model trained")
print(f"Predicted SHAPS score: {predicted:.2f}")
print(f"Actual SHAPS score: {shaps_score}")
