
# coding: utf-8

# # Peformance evaluation

# ### Dataset
# 
# As an example dataset, we will use the [pima indian diabetes dataset](pima-indians-diabetes.csv), whose description is [here](http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names), which records measurements about several hundred patients and an indication of whether or not they tested positive for diabetes (the class label).  The classification is therefore to predict whether a patient will test positive for diabetes, based on the various measurements.

# In[1]:

###Answer to 1a) here
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

##load in the data
pima=pd.read_csv('pima-indians-diabetes.csv',encoding = 'ISO-8859-1')

##get just the features
data=pima[['numpregnant','plasma','blood pressure','sf-thickness','serum-insulin','BMI','pedigree-function','age']]

##get just the class labels
classlabel=pima['has_diabetes']


# In[2]:

#conda install -c conda-forge py-xgboost
from xgboost import XGBClassifier
xgboostmodel = XGBClassifier(n_estimators=100,eta=0.3,min_child_weight=1,max_depth=3)


xgboostmodel.fit(data,classlabel)


# In[ ]:




# ## Practical Exercises
# 1) Consider the pima diabetes dataset contained in the folder
# 
# a) Report the AUC of an XGBoost classifier on the dataset using 10 fold cross validation.
# 
# b) What is the effect on AUC of varying the following parameters for XGBoost? 
# 
#     • max depth
#     • learning rate 
#     • n estimators 
#     • gamma 
#     • min child weight 
#     • reg lambda

# In[3]:

print("10-fold cross validation AUC= ",cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean())


# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 
# 
# #### Booster Parameters
# 1) max_depth [default=6]
# 
#     - The maximum depth of a tree, same as GBM.
#     - Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#     - Should be tuned using CV.
#     - Typical values: 3-10

# In[4]:

max_depths = range(1,15,1)
accuracies=[]
for m_depth in max_depths:

    xgboostmodel = XGBClassifier(n_estimators=100,eta=0.3,min_child_weight=1,max_depth=m_depth)

    #print("10-fold cross validation AUC= ",cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean())
        
    accuracies.append(cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean())

plt.plot(max_depths,accuracies)
plt.xlabel('Max-Depth', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.show()


# #### Booster Parameters
# 2) eta [default=0.3]
# 
#     - Analogous to learning rate in GBM
#     - Makes the model more robust by shrinking the weights on each step
#     - Typical final values to be used: 0.01-0.2

# In[8]:

import numpy as np
l_rates = np.arange(0.01,0.2,0.01)
accuracies=[]
for l_r in l_rates:

    xgboostmodel = XGBClassifier(n_estimators=100,eta=l_r,min_child_weight=1,max_depth=7)

    #print("10-fold cross validation AUC= ",cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean())
        
    accuracies.append(cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean())

plt.plot(l_rates,accuracies)
plt.xlabel('Learning Rate', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.show()


# #### Booster Parameters
# 3) n_estimators (covered before)

# In[9]:

import matplotlib.pyplot as plt
import numpy as np

n_estimator = range(50, 500, 50)

auc=[]
for nt in n_estimator:
    
    xgboostmodel = XGBClassifier(n_estimators=nt,eta=0.3,min_child_weight=3,max_depth=2)
    
    cv_score = cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean()
    auc.append(cv_score)

plt.plot(n_estimator,auc)
plt.xlabel('n_estimators', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.show()


# #### Booster Parameters
# 4) min_child_weight [default=1]
#     - Defines the minimum sum of weights of all observations required in a child.
#     - This refers to min “sum of weights” of observations.
#     - Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#     - Too high values can lead to under-fitting hence, it should be tuned using CV.
#     
# ** stop trying to split once your sample size in a node goes below a given threshold.
# 

# In[10]:

Min_child_weight = range(1,15)

auc=[]

for mcw in Min_child_weight:
    
    xgboostmodel = XGBClassifier(n_estimators=75,eta=0.3,min_child_weight=mcw,max_depth=2, gamma=1.5)
    
    cv_score = cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean()
    auc.append(cv_score)

plt.plot(Min_child_weight,auc)
plt.xlabel('Min_child_weight', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.show()


# #### Booster Parameters
# 5) gamma [default=0]
#     - A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
#     - Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
# 
# ** The complexity cost by introducing additional leaf

# In[12]:

import matplotlib.pyplot as plt
import numpy as np

gamma = np.arange(0, 1, 0.01)

auc=[]
for gm in gamma:
    
    xgboostmodel = XGBClassifier(n_estimators=75,eta=0.3,min_child_weight=3,max_depth=2, gamma=gm)
    
    cv_score = cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean()
    auc.append(cv_score)

plt.plot(gamma,auc)
plt.xlabel('gamma', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.show()


# #### Booster Parameters
# 6) lambda [default=1]
#     - L2 regularization term on weights (analogous to Ridge regression)
#     - This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.

# In[13]:

reg_lambda = [1e-5,1e-3, 1e-4 ,1e-2, 0.1, 0.2, 0.5, 0.9, 1,5,10,20,50, 100]

auc=[]

for regl in reg_lambda:
    
    xgboostmodel = XGBClassifier(n_estimators=75,eta=0.3,min_child_weight=5,max_depth=2, gamma=1.5, reg_lambda = regl)
    
    cv_score = cross_val_score(xgboostmodel, data,classlabel,cv=10,scoring='roc_auc').mean()
    auc.append(cv_score)

plt.plot(reg_lambda,auc)
plt.xlabel('reg_lambda', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.show()


# ### Grid Search for Parameter Tuning

# In[16]:

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

xgboostmodel = XGBClassifier(n_estimators=nt,eta=0.3,min_child_weight=3,max_depth=2)

n_estimators = [50,100,200,300]
max_depth = [2,4,6,8,10]
tuned_parameters = dict(max_depth =max_depth , n_estimators = n_estimators)

kfold = StratifiedKFold(n_splits=10, random_state=7, shuffle=True)

clf = GridSearchCV(xgboostmodel, tuned_parameters, cv=kfold, scoring='roc_auc')

clf.fit(data,classlabel)

print("Best parameters set found on development set:")
print(clf.best_params_)


# #### Other Booster Parameters
# 7) max_leaf_nodes
#     - The maximum number of terminal nodes or leaves in a tree.
#     - Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
#     - If this is defined, GBM will ignore max_depth.

# In[ ]:



