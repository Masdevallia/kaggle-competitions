
# Import packages:
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Import data:
sample_submission = pd.read_csv('./input/avila/sample_submission.csv')
test_df = pd.read_csv('./input/avila/test_dataset.csv')
training_df = pd.read_csv('./input/avila/training_dataset.csv')

# .......................................................................................................

# Data wrangling:
X = training_df.drop(columns=['id','scribe'])
y = training_df['scribe']

rs = RobustScaler() # Scale features using statistics that are robust to outliers
X_scaled = rs.fit_transform(X)

# Split data:
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X,y,train_size=0.2) # not scaled
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,y,train_size=0.2) # scaled

# .......................................................................................................

# Model testing:
models = {'LogisticRegression': LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=500),
          'SVC': SVC(gamma='auto'),
          'KNeighborsClassifier': KNeighborsClassifier(3),
          'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
          'DecisionTreeClassifier': DecisionTreeClassifier(),
          'GradientBoostingClassifier': GradientBoostingClassifier()}

# Scaled:
metrics = {}
for modelName, model in models.items():
    clf = model.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    metrics[modelName] = {'accuracy': round(accuracy_score(y_test, y_pred),2)}
#{'LogisticRegression': {'accuracy': 0.57},
# 'SVC': {'accuracy': 0.74},
# 'KNeighborsClassifier': {'accuracy': 0.76},
# 'RandomForestClassifier': {'accuracy': 0.89},
# 'DecisionTreeClassifier': {'accuracy': 0.84},
# 'GradientBoostingClassifier': {'accuracy': 0.87}}

# Not scaled
metrics_ns = {}
for modelName, model in models.items():
    clf = model.fit(X_train_ns, y_train_ns)
    y_pred = clf.predict(X_test_ns)
    metrics_ns[modelName] = {'accuracy': round(accuracy_score(y_test_ns, y_pred),2)}
#{'LogisticRegression': {'accuracy': 0.56},
# 'SVC': {'accuracy': 0.67},
# 'KNeighborsClassifier': {'accuracy': 0.65},
# 'RandomForestClassifier': {'accuracy': 0.89},
# 'DecisionTreeClassifier': {'accuracy': 0.86},
# 'GradientBoostingClassifier': {'accuracy': 0.9}}

# .......................................................................................................

# RandomForestClassifier GridSearchCV:
parameters = { 
    'n_estimators': [200,500,700,900,1100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [10,12,14,16,18,20],
    'criterion' :['gini', 'entropy']}
rfc = RandomForestClassifier() 
clf = GridSearchCV(rfc, parameters, cv=5, scoring='accuracy', verbose=5, n_jobs= -1)
clf.fit(X_scaled, y)
# print(clf.best_estimator_)
# print(clf.best_score_) 
# print(clf.best_params_)

# GradientBoostingClassifier GridSearchCV:
parameters = { 
    'n_estimators': [400, 450, 500, 550, 600],
    'learning_rate' :[0.1, 0.2, 0.3, 0.4, 0.5]}
gbc = GradientBoostingClassifier() 
clf = GridSearchCV(gbc, parameters, cv=5, scoring='accuracy', verbose=5, n_jobs= -1)
clf.fit(X, y)
# print(clf.best_estimator_)
# print(clf.best_score_) 
# print(clf.best_params_)

# .......................................................................................................

# Predictions:
Xtesting = test_df.drop(columns=['id'])
Xtesting_scaled = rs.transform(Xtesting)
finalpred = clf.predict(Xtesting) # Call predict on the estimator with the best found parameters
# finalpred = clf.predict(Xtesting_scaled)
sample_submission.scribe = finalpred
sample_submission.to_csv('./submissions/submission-avila-8.csv', index=False)
