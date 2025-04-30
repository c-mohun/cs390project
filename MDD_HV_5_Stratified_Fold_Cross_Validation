from nilearn import input_data, datasets
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from nilearn.connectome import ConnectivityMeasure

# === Setup ===
subject_ids = [
    "sub-20900", "sub-21669", "sub-21748", "sub-22695",
    "sub-23199", "sub-23457", "sub-23490", "sub-23502", "sub-23513", "sub-23399", "sub-23017", "sub-21670",
    "sub-23457", "sub-22698", "sub-21111", "sub-21723", "sub-21250", "sub-22477"
]
labels = [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  # 1 = MDD, 0 = HV
shaps_scores = [1, 5, 15, 9, 12, 1, 20, 27, 25, 23, 11, 20, 20, 14, 12, 1, 1, 15]  # SHAPS scores aligned with subject_ids

confound_vars = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_filename = atlas['maps']
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
correlation_measure = ConnectivityMeasure(kind='correlation')

X = []
y = []

# === Feature Extraction ===
for i, subject in enumerate(subject_ids):
    try:
        func_file = f'{subject}/ses-v1/func/{subject}_ses-v1_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
        confounds_file = f'{subject}/ses-v1/func/{subject}_ses-v1_task-rest_run-1_desc-confounds_timeseries.tsv'
        
        confounds = pd.read_csv(confounds_file, sep='\t')
        timeseries = masker.fit_transform(func_file, confounds=confounds[confound_vars])
        correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
        
        G = nx.from_numpy_array(np.abs(correlation_matrix))
        strengths = dict(G.degree(weight='weight'))
        features = list(strengths.values())

        if len(features) == 100:
            features_with_shaps = features + [shaps_scores[i]]  # Append SHAPS score as 101st feature
            X.append(features_with_shaps)
            y.append(labels[i])
            print(f"Processed {subject}")
        else:
            print(f"Skipped {subject} due to incomplete features")

    except Exception as e:
        print(f"Error with {subject}: {e}")

X = np.array(X)
y = np.array(y)

pipeline = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', max_iter=1000))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv)

print("Cross-validation complete")
print(f"Accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean():.2f}")

y_pred = cross_val_predict(pipeline, X, y, cv=cv)

print("\nPredictions per subject:")
for i, subject in enumerate(subject_ids[:len(y_pred)]):
    actual = "MDD" if y[i] == 1 else "HV"
    predicted = "MDD" if y_pred[i] == 1 else "HV"
    print(f"{subject}: Actual={actual}, Predicted={predicted}")

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["HV", "MDD"]))
