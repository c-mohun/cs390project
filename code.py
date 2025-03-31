import os
import numpy as np
import pandas as pd
import networkx as nx
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import datasets, input_data, plotting
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
from scipy.ndimage import center_of_mass


subject_ids = [
    "sub-20900", "sub-21669", "sub-21748", "sub-22695", "sub-22699",
    "sub-23199", "sub-23457", "sub-23490", "sub-23502", "sub-23513",
]

shaps_scores = [1, 5, 15, 9, 12, 1, 20, 27, 25, 23]
subject_to_score = dict(zip(subject_ids, shaps_scores))

HV_MVV = [1, 1, 0, 1, 0, 1, 1, 0, 0, 0]

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_filename = atlas['maps']
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

X_all = []
y_all = []

for subject_id in subject_ids:
    try:
        base_path = f"{subject_id}/ses-v1/func/"
        func_file = os.path.join(base_path, f"{subject_id}_ses-v1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
        confounds_file = os.path.join(base_path, f"{subject_id}_ses-v1_task-rest_run-1_desc-confounds_timeseries.tsv")

        confounds = pd.read_csv(confounds_file, sep='\t')
        confound_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        confound_data = confounds[confound_vars]

        timeseries = masker.fit_transform(func_file, confounds=confound_data)

        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([timeseries])[0]

        G = nx.from_numpy_array(np.abs(correlation_matrix))
        strengths = dict(G.degree(weight='weight'))
        features = list(strengths.values())

        X_all.append(features)
        y_all.append(subject_to_score[subject_id])

        print(f"Processed {subject_id}")

    except Exception as e:
        print(f"Error with {subject_id}: {e}")

X_all_array = np.array(X_all)
y_array = np.array(y_all)

correlations = np.corrcoef(X_all_array.T, y_array)[-1, :-1]

sorted_idx = np.argsort(correlations)

labels_list = atlas['labels'][1:]  # Exclude background label
labels_list = [label.decode("utf-8") if isinstance(label, bytes) else label for label in labels_list]


print("\nTop 5 negatively correlated regions (stronger connectivity = lower SHAPS):")
for i in sorted_idx[:5]:
    print(f"{labels_list[i]}: r = {correlations[i]:.2f}")

print("\nTop 5 positively correlated regions (stronger connectivity = higher SHAPS):")
for i in sorted_idx[-5:][::-1]:
    print(f"{labels_list[i]}: r = {correlations[i]:.2f}")

labels_img = nib.load(atlas['maps'])
labels_data = labels_img.get_fdata()
affine = labels_img.affine

unique_labels = np.unique(labels_data)[1:]  # skip background (0)
coords = []

for label in unique_labels:
    mask = labels_data == label
    com = center_of_mass(mask)
    real_coords = nib.affines.apply_affine(affine, com)
    coords.append(real_coords)


plotting.plot_connectome(
    adjacency_matrix=np.zeros((100, 100)),  # No edges
    node_coords=coords,
    node_color=correlations,
    node_size=50,
    edge_threshold=None,
    title="Correlation of Node Strength with SHAPS Score"
)
plotting.show()


import matplotlib.pyplot as plt

sorted_corr_idx = np.argsort(correlations)
top_neg_idx = sorted_corr_idx[:5]
top_pos_idx = sorted_corr_idx[-5:][::-1]
top_indices = np.concatenate([top_neg_idx, top_pos_idx])
top_labels = [labels_list[i] for i in top_indices]
top_values = [correlations[i] for i in top_indices]

plt.figure(figsize=(12, 6))
colors = ['steelblue'] * 5 + ['goldenrod'] * 5
sns.barplot(x=top_labels, y=top_values, palette=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Correlation with SHAPS Score")
plt.title("Top Brain Regions Correlated with SHAPS")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()


#blue means - stringer region = lower shapaps, yellow - stronger region = higher shaps 

print("\n--- Visualizing Group-Level Connectomes ---")

correlation_measure = ConnectivityMeasure(kind='correlation')

hv_indices = [i for i, label in enumerate(HV_MVV) if label == 1]
mvv_indices = [i for i, label in enumerate(HV_MVV) if label == 0]

hv_matrices = []
mvv_matrices = []

for i in hv_indices:
    ts = masker.transform(f"{subject_ids[i]}/ses-v1/func/{subject_ids[i]}_ses-v1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    hv_matrices.append(correlation_measure.fit_transform([ts])[0])

for i in mvv_indices:
    ts = masker.transform(f"{subject_ids[i]}/ses-v1/func/{subject_ids[i]}_ses-v1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    mvv_matrices.append(correlation_measure.fit_transform([ts])[0])


avg_hv_matrix = np.mean(hv_matrices, axis=0)
avg_mvv_matrix = np.mean(mvv_matrices, axis=0)
diff_matrix = avg_hv_matrix - avg_mvv_matrix


plotting.plot_connectome(
    adjacency_matrix=diff_matrix,
    node_coords=coords,
    node_size=30,
    edge_threshold='97%',
    title="HV - MVV: Connectivity Difference"
)
plotting.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

print("\n--- Running HV/MVV Classification with Feature Selection + LOOCV ---")

selected_indices = [38, 22, 44, 37, 56, 59, 2, 4, 8, 14]  
X_selected = X_all_array[:, selected_indices]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

loo = LeaveOneOut()
y_true, y_pred = [], []

print("\nSubject-wise Predictions:")
for i, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = np.array(HV_MVV)[train_idx], np.array(HV_MVV)[test_idx]

    clf = LogisticRegression(penalty='l2', solver='liblinear')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)[0]

    y_pred.append(prediction)
    y_true.append(y_test[0])

    subject = subject_ids[test_idx[0]]
    print(f"{subject}: Predicted = {prediction}, Actual = {y_test[0]}")

print(f"\nAccuracy (LOOCV): {accuracy_score(y_true, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["MVV (0)", "HV (1)"]))