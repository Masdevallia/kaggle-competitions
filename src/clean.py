
import pandas as pd
import numpy as np


# Data examination:
'''
print(training_df.head())
print(training_df.shape)
print(test_df.shape)
print(sample_submission.shape)
print(training_df.columns)
print(X.dtypes)
print(X.isnull().sum())
corr_matrix = X.select_dtypes(['number']).corr() 
print(corr_matrix[corr_matrix > 0.9])
'''


# Data cleaning:
def cleanDF(X):
    # droping highly correlated features:
    X.drop(columns=['x','y','z'], inplace=True)
    # cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
    diamonds_cut = {'Ideal':5,'Premium':4,'Very Good':3,'Good':2,'Fair':1}
    for key, value in diamonds_cut.items():
        X.cut = np.where(X.cut == key, value, X.cut)
    X.cut = pd.to_numeric(X.cut)
    # color: diamond colour, from J (worst) to D (best)
    diamonds_color = {'D': 6,'E': 5,'F': 4,'G': 3,'H': 2,'I': 1,'J': 0}
    for key, value in diamonds_color.items():
        X.color = np.where(X.color == key, value, X.color)
    X.color = pd.to_numeric(X.color)
    # clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
    diamonds_clarity = {'IF': 7,'VVS1': 6,'VVS2': 5,'VS1': 4,'VS2': 3,'SI1': 2,'SI2': 1,'I1': 0}
    for key, value in diamonds_clarity.items():
        X.clarity = np.where(X.clarity == key, value, X.clarity)
    X.clarity = pd.to_numeric(X.clarity)
    return X


# Feature selection:
# Finally I ended up not implementing feature selection because model metrics were better using all features.
'''
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif

selectFeaturesPerc = SelectPercentile(f_classif, percentile=10)
X_train_new = selectFeaturesPerc.fit_transform(X_train_scaled, y_train)
X_test_new = selectFeaturesPerc.transform(X_test_scaled)

selectBestFeatures = SelectKBest(f_classif, k=5)
X_train_new = selectBestFeatures.fit_transform(X_train_scaled, y_train)
X_test_new = selectBestFeatures.transform(X_test_scaled)
'''


# List of all the possible scores to evaluate the GridSearchCV:
'''
from sklearn.metrics import SCORERS
sorted(SCORERS.keys())
'''