import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.cluster import KMeans

import os

# Any results you write to the current directory are saved as output.
yale = pd.read_csv('../input/yalev1.csv')
yale = yale.loc[:31,:]
yale.isnull().sum()
#label encoder 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(yale['County'].unique())
yale['County'] = le.transform(yale['County'])

#Label propagation
from sklearn.semi_supervised import LabelSpreading
def getLabelPropa(yale):
    n = len(yale)
    yale['labels'] = yale['Rank']
    yale['labels'].loc[yale['Town'].isin(['Greenwich','Westport','Fairfield','Trumbull','Ridgefield'])] = 1
    #print(yale['labels'])
    label = yale['labels']
    yale = yale.select_dtypes(include = ['float64','int64'])
    label_prop_model = LabelSpreading(alpha = 0.1, kernel = 'rbf', n_neighbors = 3, max_iter = 300,gamma =2)
    yale = yale.drop(['labels'],axis = 1)
    yale = preprocessing.normalize(yale,axis = 0,norm='max')
    label_prop_model.fit(yale, label)
    label = label_prop_model.predict(yale)
    ##print(label_prop_model.predict(yale))
    #print(label_prop_model)
    #print(label_prop_model.predict_proba(yale))
    return label
label = getLabelPropa(yale)
yale['label_lp'] = label
yale[yale['label_lp'] == 1]


def get_label_km(yale):
    yale_train = yale.loc[yale['Town'].isin(['Greenwich','Westport','Fairfield','Trumbull','Ridgefield'])]
    yale_trial = yale.drop(['Rank'],axis = 1)
    normalized_yale_train = preprocessing.normalize(yale_train.select_dtypes(include = ['float64','int64']),axis = 0,norm='max')
    print(normalized_yale_train.var(axis = 0))
    yale_cov = (preprocessing.normalize(yale_trial.select_dtypes(include = ['float64','int64']),axis = 0,norm='max'))
    kmeans = KMeans(n_clusters = 2, random_state=0).fit(yale_cov)
    yale['labels'] = kmeans.labels_
    #print(yale.loc[yale['Town'].isin(['Greenwich','Westport','Fairfield','Trumbull','Ridgefield'])])
    #print(yale[yale['labels'] == 0])
    return yale['labels']
label = get_label_km(yale)
yale['label_km'] = label
yale.loc[yale['Town'].isin(['Greenwich','Westport','Fairfield','Trumbull','Ridgefield'])]
yale[yale['label_km'] == 0]


yale_train_cov = yale[yale['label_lp'] == 1].select_dtypes(include = ['float64','int64'])
import  matplotlib.pyplot as plt
import seaborn as sns
def get_cor(train_SJ):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(train_SJ.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
get_cor(yale_train_cov)

def param(x):
    mean = np.mean(x,axis = 0)
    std = np.std(x,axis = 0)
    return mean,std
def preprocess(x,mean,std):
    m,n=x.shape
    x_normal = np.zeros((m,n))
    for i in range(x.shape[1]):
        if(std[i] == 0):
            std[i] =1
        x_normal[:,i] = (x[:,i] - mean[i])/std[i]
    #b = np.ones((len(x),1))
   # x_new= np.column_stack([b,x_normal])
    x_new = x_normal
    return x_new

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def rmsle_cv(model,train_data,y):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data)
    rmse= np.sqrt(-cross_val_score(model, train_data, y, scoring="neg_mean_squared_error", cv = kf))
    return(np.mean(rmse))
from sklearn.linear_model import LassoCV, Lasso, ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
#Lasso = make_pipeline(RobustScaler(),LassoCV(alphas = [1, 0.1, 0.001, 0.0005], random_state = 1))

#lasso
train_x = yale[yale['label_lp'] == 1]
#print(train_x.info)
test_x = train_x[train_x.Town == 'Fairfield']
train_x = train_x[train_x.Town != 'Fairfield']
test_y = test_x['PPE']
train_y = train_x['PPE']
train_x = train_x.drop(['PPE','Town','labels','label_lp','label_km','Average_Age','Population'],axis = 1)
test_x = test_x.drop(['PPE','Town','labels','label_lp','label_km','Average_Age','Population'],axis = 1)
Lasso = Lasso(alpha = 0.001,tol=0.005).fit(train_x,train_y)
print(Lasso.coef_)
mean,std = param(train_x)
train_x = preprocess(train_x.values,mean,std)
test_x = preprocess(test_x.values,mean,std)
print(rmsle_cv(Lasso,train_x,train_y))
Lasso.fit(train_x,train_y)
print(Lasso.predict(test_x))
print(test_y)
