
# Import packages:
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.clean import cleanDF
from src.charts import featureImportance, caratChart

# Import data:
sample_submission = pd.read_csv('./input/diamonds/sample_submission.csv')
test_df = pd.read_csv('./input/diamonds/test.csv')
training_df = pd.read_csv('./input/diamonds/data.csv')

X = training_df.drop(columns=['price'])
y = training_df['price']

# Data wrangling:
Xclean = cleanDF(X)
rs = RobustScaler() # Scale features using statistics that are robust to outliers
X_scaled = rs.fit_transform(Xclean)

# Split data:
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,y,train_size=0.2)

# Model testing:
models = {'LinearRegression': LinearRegression(),
        'SVR': SVR(gamma='auto'),
        'KNeighborsRegressor': KNeighborsRegressor(7),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=500),
        'DecisionTreeRegressor': DecisionTreeRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor()}

metrics = {}
for modelName, model in models.items():
    clf = model.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    metrics[modelName] = {'RMSE': round((mean_squared_error(y_test,y_pred))**0.5,2),'r2': round(r2_score(y_test,y_pred),2)}

#{'DecisionTreeRegressor': {'RMSE': 785.24, 'r2': 0.96},
# 'GradientBoostingRegressor': {'RMSE': 654.08, 'r2': 0.97},
# 'KNeighborsRegressor': {'RMSE': 959.8, 'r2': 0.94},
# 'LinearRegression': {'RMSE': 1232.13, 'r2': 0.9},
# 'RandomForestRegressor': {'RMSE': 590.53, 'r2': 0.98},
# 'SVR': {'RMSE': 3888.72, 'r2': 0.05}}

# RandomForestRegressor GridSearchCV:
parameters = { 
    'bootstrap': [True, False],
    'n_estimators': [100, 500, 750, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 8, 10, 12],
    'max_depth' : [10, 50, 75, 100, 150]}
rfr = RandomForestRegressor() 
clf = GridSearchCV(rfr, parameters, cv=5, scoring='r2', verbose=5, n_jobs= -1)
clf.fit(X_scaled, y)
# print(clf.best_estimator_)
# print(clf.best_score_)
# print(clf.best_params_)

# Predictions:
test_df.drop(columns=['id'], inplace=True)
test_dfclean = cleanDF(test_df)
test_df_scaled = rs.transform(test_dfclean)
finalpred = clf.predict(test_df_scaled) # Call predict on the estimator with the best found parameters
sample_submission.price = finalpred
sample_submission.to_csv('./submissions/submission-diamonds-1.csv', index=False)

# Charts:
featureImportance(X_scaled, y, Xclean)
caratChart(Xclean, y)