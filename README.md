Code used in Clementine's and Vivienne's cs390 Project 
fmriprep_singleSubj.sh - the code used for frmiPrep. 
connectivity_matrix - code to create connectivity Matirx. 
LinearRegression_Wout_LOOCV - linear regression model to predict MDD or HV based on brain connectivity. Code disregarded due to the small subject sample size and replaced with LOOCV instead. 
MDD_HV_NodeStrength_Comparison_LOOCV.py python code for MDD vs HV node strength comparison. Model trained on Leave one out cross validation. Code also disregarded due to the small subject sample size and replaced with 5-fold Stratified Cross Validation instead. MDD_HV_5_Stratified_Fold_Cross_Validation.py final pyhton code used to predict MDD vs HV with 5-fold Stratified Cross Validation and SHAPS Score appended to subject's feature vector. 
