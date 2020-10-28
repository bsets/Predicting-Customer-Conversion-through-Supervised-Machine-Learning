#Import Libraries
import numpy as np
from numpy import array
from numpy import argmax

import pandas as pd
from csv import reader

import sklearn
import sklearn.datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from collections import Counter

import xgboost
from xgboost import XGBClassifier


import imblearn
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy=0.5)

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

#Read a CSV file and create a list containing the loaded dataset
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# load and prepare data
filename = 'input_training_data.csv'
training_data=load_csv(filename)
filename2= 'input_test_data.csv'
test_data=load_csv(filename2)


#list of attributes that must be one hot encoded

ohe_attributes_list=[['visitor_type',16],['month_train',11],['operating_systems',12],
                     ['browser_train',13],['region_train',14],['traffic_type',15],
                     ['weekend_train',17]
                     ]

#print(ohe_attributes_list[3][1])

for i in range(len(ohe_attributes_list)):
    a_train=[]
    for j in range (1,len(training_data)):
        a_train.append(training_data[j][ohe_attributes_list[i][1]])
    a_train_array = array(a_train)
    a_integer_encoded = label_encoder.fit_transform(a_train_array)
    a_integer_encoded = a_integer_encoded.reshape(len(a_integer_encoded), 1)
    a_onehot_encoded = onehot_encoder.fit_transform(a_integer_encoded)
    a_onehot_encoded_list = a_onehot_encoded.tolist()

    for k in range(1,len(training_data)):
        training_data[k][ohe_attributes_list[i][1]]=a_onehot_encoded_list[k-1]
    
#Convert String to Float: All columns that are numerical - both discrete and continuous

for i in range(1,11):
    str_column_to_float(training_data[1:], i)
    
for i in range(18,19):
    str_column_to_float(training_data[1:], i)
    
#Normalization of Columns
    
for j in range(1,11):
    column=[]
    for i in range (1,len(training_data)):
        column.append(training_data[i][j])
    min_val=min(column)
    max_val=max(column)
    for k in range (1,len(training_data)):
        training_data[k][j]=((training_data[k][j]-min_val)/(max_val-min_val))

# Combine non-list attributes of training_data into a list 
        
for i in range(1,len(training_data)):
    a=[]
    for j in range (1,11):
        a.append(training_data[i][j])
    del (training_data[i][1:11])
    training_data[i].insert(1,a)
    
# Combine all lists in a training_data sample into a single list 

b=[ [] for _ in range(len(training_data)-1) ]
for i in range(1,len(training_data)):
    for j in range(len(training_data[1])):
        if (isinstance(training_data[i][j], list)==0):
            b[i-1].append(training_data[i][j])
        elif (isinstance(training_data[1][j], list)==1):
            for k in range (len(training_data[i][j])):
                b[i-1].append(training_data[i][j][k])
            

training_data=b;

y_train=[]
x_train=[]
for i in range(len(training_data)):
    y_train.append(training_data[i][74])
    x_train.append(training_data[i][0:74])
   



print(Counter(y_train))

temp_list=[]
temp_list.append('ID')
for i in range(1,74):
    j='var'+str(i)
    temp_list.append(j)

df1=pd.DataFrame(x_train,columns=temp_list)

#print(df1.columns)

for col in df1:
    if (col=='ID'):
        pass
    else:
        df1[col] = df1[col].astype(float) 

#print(df1.dtypes)

df2=pd.DataFrame(y_train,columns=['Revenue']) 
#print(df2.dtypes)

X=df1.drop('ID', axis=1)
y=df2.Revenue

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state = 0,stratify=y,shuffle=True)
print(Counter(y_train))
print(Counter(y_validation))


#Hyperparameter Optimization

space={ 'max_depth': hp.quniform('max_depth', 5,15,1),
        'learning_rate' : hp.quniform('learning_rate', 0.01, 0.25, 0.01),
        'reg_alpha' : hp.quniform('reg_alpha',1,10,1),
        'reg_lambda' : hp.quniform('reg_lambda',1,10,1),
        'colsample_bytree' :hp.quniform('colsample_bytree',0.1,1,0.1),
        'min_child_weight' : hp.quniform('min_child_weight',0,10,1),
        'subsample' : hp.quniform('subsample',0.1,1,0.1),
        'max_delta_step' : hp.quniform('max_delta_step',1,10,1),
        'n_estimators': hp.quniform('n_estimators',100,300,100),
        'scale_pos_weight':hp.quniform('scale_pos_weight',1,10,1),
        'random_state': 0,
        'gamma':hp.quniform('gamma',1,10,1)
    }

# =============================================================================
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.1, max_delta_step=6, max_depth=8,
#               min_child_weight=0, missing=nan, monotone_constraints='()',
#               n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
#               reg_alpha=2, reg_lambda=2, scale_pos_weight=5.4, subsample=0.5,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# =============================================================================

def objective(space):
    clf=XGBClassifier(
                    n_estimators =int(space['n_estimators']), max_depth =int(space['max_depth']), gamma = int(space['gamma']),
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=space['colsample_bytree'],reg_lambda = int(space['reg_lambda']),
                    subsample = space['subsample'],max_delta_step = int(space['max_delta_step']),
                    learning_rate = space['learning_rate'],scale_pos_weight = int(space['scale_pos_weight']),
                    random_state = space['random_state'])
    
    evaluation = [( X_train, y_train), ( X_validation, y_validation)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_validation)
    accuracy = accuracy_score(y_validation, pred)
    print ("SCORE:", accuracy)
    prob_y_validation = clf.predict_proba(X_validation)
    prob_y_validation_1 = [p[1] for p in prob_y_validation]
    return {'loss': 1-roc_auc_score(y_validation,prob_y_validation_1), 'status': STATUS_OK }
    #return {'loss': -accuracy, 'status': STATUS_OK }
#    return {'roc_auc_score': roc_auc_score(y_validation,prob_y_validation_1), 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 1000,
                        trials = trials)
 

print(best_hyperparams)



# =============================================================================
# #Itertion 2:
# clf_5 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=1, gamma=1, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.01, max_delta_step=6, max_depth=10,
#               min_child_weight=0, monotone_constraints='()',
#               n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
#               reg_alpha=2, reg_lambda=5, scale_pos_weight=5.4, subsample=0.5,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# =============================================================================

for col in X:
    if (col=='ID'):
        pass
    else:
        df1[col] = df1[col].astype(float) 

print(X.dtypes)
print(y.dtypes)

df2=pd.DataFrame(y_train,columns=['Revenue']) 
print(df2.dtypes)

#Itertion 5:
clf_5 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=9, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=7, max_depth=10,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=1, reg_lambda=10, scale_pos_weight=6, subsample=0.8,
              tree_method='exact', validate_parameters=1, verbosity=None)

clf_5.fit(X, y)
 
# Predict on training set
pred_y_5 = clf_5.predict(X)

print( np.unique( pred_y_5 ) )

 

print( accuracy_score(y, pred_y_5) )

 

prob_y_5 = clf_5.predict_proba(X)

prob_y_5 = [p[1] for p in prob_y_5]
print( roc_auc_score(y, prob_y_5) )


# Test Data

#one_hot encoding of visitor_type attribute in test dataset

#print(ohe_attributes_list[3][1])

for i in range(len(ohe_attributes_list)):
    a_test=[]
    for j in range (1,len(test_data)):
        a_test.append(test_data[j][ohe_attributes_list[i][1]])
    a_test_array = array(a_test)
    a_integer_encoded = label_encoder.fit_transform(a_test_array)
    a_integer_encoded = a_integer_encoded.reshape(len(a_integer_encoded), 1)
    a_onehot_encoded = onehot_encoder.fit_transform(a_integer_encoded)
    a_onehot_encoded_list = a_onehot_encoded.tolist()

    for k in range(1,len(test_data)):
        test_data[k][ohe_attributes_list[i][1]]=a_onehot_encoded_list[k-1]

#Convert String to Float: All columns that are numerical - both discrete and continuous

for i in range(1,11):
    str_column_to_float(test_data[1:], i)
    
#Normalization of Columns
    
for j in range(1,11):
    column=[]
    for i in range (1,len(test_data)):
        column.append(test_data[i][j])
    min_val=min(column)
    max_val=max(column)
    for k in range (1,len(test_data)):
        test_data[k][j]=((test_data[k][j]-min_val)/(max_val-min_val))

# Combine non-list attributes of test_data into a list 
        
for i in range(1,len(test_data)):
    a=[]
    for j in range (1,11):
        a.append(test_data[i][j])
    del (test_data[i][1:11])
    test_data[i].insert(1,a)
    
# Combine all lists in a test_data sample into a single list 

b=[ [] for _ in range(len(test_data)-1) ]
for i in range(1,len(test_data)):
    for j in range(len(test_data[1])):
        if (isinstance(test_data[i][j], list)==0):
            b[i-1].append(test_data[i][j])
        elif (isinstance(test_data[1][j], list)==1):
            for k in range (len(test_data[i][j])):
                b[i-1].append(test_data[i][j][k])
            
x_test=b;

df3=pd.DataFrame(x_test,columns=temp_list) 
 
X_test = df3.drop('ID', axis=1)    

# Predict on training set
pred_y_test = clf_5.predict(X_test)

print( np.unique( pred_y_test ) )

prob_y_test = clf_5.predict_proba(X_test)
prob_y_test_final = [p[1] for p in prob_y_test]

import csv
with open('Probability_Prediction_XG_Boost_hp_tuning_14.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(prob_y_test_final)