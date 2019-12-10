# <p align="center">Kaggle competitions: Diamonds and Avila Bible</p>

## <p align="center">Ironhack's Data Analytics Bootcamp Project V: Supervised Machine Learning | Regression and Classification</p>

The goal of this project was to participate in two Kaggle competitions to put into practice what we have learned about supervised machine learning methods.

* [Diamonds competition](https://www.kaggle.com/c/diamonds-datamad1019/overview) (regression problem).
* [Avila Bible competition](https://www.kaggle.com/c/avila-bible-datamad1019/overview) (classification problem).

### Diamonds competition:

In this competition we had to predict diamonds price based on their characteristics (color, clarity, quality of cut, carat weight, etc.).

#### STEPS:

* Data preparation, feature selection and extraction:
    * Convert categorical variables (cut, color, clarity) to numbers with ordinal encoding.
    * Drop highly correlated features: x, y, z.
    * Scale features using statistics that are robust to outliers: RobustScaler().
    * Add polynomial terms: carat quadratic and cubic term. To test for quadratic/cubic effects and to try to improve models performance, carat was squared/cubed and entered as another predictor in some models. Carat was squared/cubed after standardization to reduce the correlation between the linear and quadratic/cubic terms.
* Train-test split to divide the data in X_ train, X_test, y_train, y_test.
* Model building, training and evaluation (R2 and RMSE): Linear Regression, SVR, K Neighbors Regressor, Random Forest Regressor, Decision Tree Regressor and Gradient Boosting Regressor.
* H2OAutoML: Automated machine learning: Automating the process of applying machine learning.
* Hyperparameter tuning: GridSearchCV: Tune model parameters for improved performance: number of training steps, learning rate, initialization values and distribution, etc.
* Call predict on the estimator with the best found parameters: Train selected model with all Kaggle train dataset and predict test.

<p align="center"><img  src="https://github.com/Masdevallia/kaggle-competitions/blob/master/images/Masdevallia.png" width="700"></p>

### Avila Bible competition:

In this competition we had to predict who was the scribe of a verse from the Avila bible according to the text format.

#### STEPS:

* Data examination.
* Train-test split to divide the data in X_ train, X_test, y_train, y_test.
* Model building, training and evaluation (Accuracy and Confusion matrix): Logistic Regression, SVC, K Neighbors Classifier, Random Forest Classifier, Decision Tree Classifier and Gradient Boosting Classifier.
* Hyperparameter tuning: GridSearchCV: Tune model parameters for improved performance: number of training steps, learning rate, initialization values and distribution, etc.
* Call predict on the estimator with the best found parameters: Train model with all Kaggle train dataset and predict test.
