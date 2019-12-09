
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# %matplotlib inline


def featureImportance(X_scaled, y, Xclean):   
    clf = RandomForestRegressor(bootstrap=True,n_estimators=1000,max_features='auto', min_samples_leaf=2,
                                min_samples_split=8,max_depth=50)
    clf.fit(X_scaled, y)
    importances = clf.feature_importances_
    df = pd.DataFrame(Xclean.columns, columns=['Features'])
    df['Importance'] = importances
    df.sort_values('Importance', ascending = False, inplace= True)
    colors = ['yellowgreen','royalblue','tomato','gold','purple','pink']
    sns.set_palette(sns.color_palette(colors))
    sns.barplot(df.Features, df.Importance, alpha=0.7)
    plt.title("Feature importance:")
    plt.xlim([-1, Xclean.shape[1]])
    plt.savefig('./images/featureImportance.png', dpi=300, bbox_inches='tight')


def caratChart(Xclean, y):
    prices = ['Affordable', 'Intermediate', 'Expensive']
    bins = pd.cut(y, 3, labels = prices)
    Xclean['price'] = bins
    colors = ['yellowgreen','royalblue','tomato'] 
    for i, price in enumerate(prices):
        subset = Xclean[Xclean['price'] == price]
        sns.distplot(subset['carat'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1},
                    label = price, color=colors[i])
    plt.legend(prop={'size': 10}, title = 'Prices')
    plt.title('Density plot')
    plt.xlabel('Carat')
    plt.ylabel('Density')
    plt.savefig('./images/caratChart.png', dpi=300, bbox_inches='tight')

