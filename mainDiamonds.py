
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
import h2o
from h2o.automl import H2OAutoML
from src.clean import cleanDF
from src.charts import featureImportance, caratChart

# Import data:
sample_submission = pd.read_csv('./input/diamonds/sample_submission.csv')
test_df = pd.read_csv('./input/diamonds/test.csv')
training_df = pd.read_csv('./input/diamonds/data.csv')

# .......................................................................................................

# Data wrangling:
X = training_df.drop(columns=['price'])
y = training_df['price']

Xclean = cleanDF(X)
# Xclean = Xclean[['carat','clarity','color']]

# Scale features using statistics that are robust to outliers:
rs = RobustScaler() 
X_scaled = rs.fit_transform(Xclean)

# Split data:
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,y,train_size=0.2)

# .......................................................................................................

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

# With all features:
#{'DecisionTreeRegressor': {'RMSE': 785.24, 'r2': 0.96},
# 'GradientBoostingRegressor': {'RMSE': 654.08, 'r2': 0.97},
# 'KNeighborsRegressor': {'RMSE': 959.8, 'r2': 0.94},
# 'LinearRegression': {'RMSE': 1232.13, 'r2': 0.9},
# 'RandomForestRegressor': {'RMSE': 590.53, 'r2': 0.98},
# 'SVR': {'RMSE': 3888.72, 'r2': 0.05}}

# Only with 'carat','clarity' and 'color':
#{'LinearRegression': {'RMSE': 1256.69, 'r2': 0.9},
# 'SVR': {'RMSE': 3738.45, 'r2': 0.14},
# 'KNeighborsRegressor': {'RMSE': 684.68, 'r2': 0.97},
# 'RandomForestRegressor': {'RMSE': 657.38, 'r2': 0.97},
# 'DecisionTreeRegressor': {'RMSE': 742.01, 'r2': 0.97},
# 'GradientBoostingRegressor': {'RMSE': 667.18, 'r2': 0.97}}

# .......................................................................................................

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
sample_submission.to_csv('./submissions/submission-diamonds-8.csv', index=False)

# .......................................................................................................

# H2O AutoML:
h2o.init(nthreads = -1, max_mem_size = 6)

# Preparing train data for H2O:
DFclean = Xclean.copy()
DFclean['price'] = y
rs = RobustScaler()
numpy_clean_scaled = rs.fit_transform(DFclean[Xclean.columns])
DFclean_scaled = pd.DataFrame(numpy_clean_scaled, columns= Xclean.columns)
DFclean_scaled['price'] = DFclean['price']
# DFclean_scaled['carat2'] = [e**2 for e in DFclean_scaled.carat]
DFclean_scaled['carat3'] = [e**3 for e in DFclean_scaled.carat]
hf = h2o.H2OFrame(DFclean_scaled)

y_columns = 'price'
# x_columns = list(Xclean.columns)
x_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'carat2']

# Fitting models (AutoML):
aml_ti = H2OAutoML(max_runtime_secs= 300, seed= 1, nfolds=5,sort_metric='RMSE') # max_models= 50
aml_ti.train(x = x_columns, y = y_columns, training_frame = hf)
lb_ti = aml_ti.leaderboard
# print(aml_ti.leader)

# Preparing test data for H2O:
test_df.drop(columns=['id'], inplace=True)
test_dfclean = cleanDF(test_df)
test_numpy_scaled = rs.transform(test_dfclean)
test_df_scaled = pd.DataFrame(test_numpy_scaled, columns=test_dfclean.columns)
# test_df_scaled['carat2'] = [e**2 for e in test_df_scaled.carat]
test_df_scaled['carat3'] = [e**3 for e in test_df_scaled.carat]
test_df_scaled_h2o = h2o.H2OFrame(test_df_scaled)

# Call predict on the estimator with the best found parameters:
finalpred = aml_ti.leader.predict(test_df_scaled_h2o)
sample_submission.price = finalpred.as_data_frame()
sample_submission.to_csv('./submissions/submission-diamonds-10.csv', index=False)

h2o.cluster().shutdown()

# .......................................................................................................

# Charts:
featureImportance(X_scaled, y, Xclean)
caratChart(Xclean, y)
