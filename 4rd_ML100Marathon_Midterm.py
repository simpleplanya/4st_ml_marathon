# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:46:05 2020

@author: Rocky
"""

import pandas as pd 
from sklearn.model_selection import cross_val_score
#from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

train_data_source = pd.read_csv('C:\\Users\\Rocky\\Desktop\\train_data.csv')
#train_data_source.fillna(0,inplace=True)
train_Y = train_data_source['poi']

# 觀察欄位與target的相關係數
#corr_res_target = train_data_source.corr()['poi']
#corr_res_total_stock_value = train_data_source.corr()['total_stock_value']
#corr_res_exercised_stock_options = train_data_source.corr()['exercised_stock_options']

#corr_res_target.abs().sort_values()
#corr_res_total_stock_value.abs().sort_values()
#loan_advances只有兩筆不是nan
#fea_col= ['exercised_stock_options','restricted_stock','total_payments','other','salary','bonus','long_term_incentive']



train_data_source['to_poi_ratio'] = train_data_source['from_poi_to_this_person'] / train_data_source['to_messages']
train_data_source['from_poi_ratio'] = train_data_source['from_this_person_to_poi'] / train_data_source['from_messages']
train_data_source['shared_poi_ratio'] = train_data_source['shared_receipt_with_poi'] / train_data_source['to_messages']
train_data_source['bonus_to_salary'] = train_data_source['bonus'] / train_data_source['salary']
train_data_source['bonus_to_total'] = train_data_source['bonus'] / train_data_source['total_payments']
non_fea = ['name','email_address','poi']
fea_col = list(train_data_source.columns)
fea_col.remove('name')
fea_col.remove('email_address')
fea_col.remove('poi')
fea_col.remove('loan_advances')

#fea_col.remove('restricted_stock_deferred')
train_X = train_data_source[fea_col]
salary_outlier=600000
train_X.loc[train_X['salary'] >salary_outlier,'salary'] = train_X['salary'].median()
train_X['salary'].fillna(train_X['salary'].median(),inplace=True)

bonus_outlier=500000
train_X.loc[train_X['bonus'] >salary_outlier,'bonus'] = train_X['bonus'].median()
train_X['bonus'].fillna(train_X['bonus'].median(),inplace=True)
long_term_incentive_outlier = 3000000
train_X.loc[train_X['long_term_incentive'] >long_term_incentive_outlier,'long_term_incentive'] = train_X['long_term_incentive'].median()
train_X['long_term_incentive'].fillna(train_X['long_term_incentive'].median(),inplace=True)


#import matplotlib.pyplot as plt
#import seaborn as sns 
#sns.set(style='whitegrid',context='notebook')
#sns.pairplot(train_X)
#plt.show()



#train_X.fillna(train_X.median(),inplace=True)
train_X.fillna(0,inplace=True)

#train_X = train_data_source
#train_X = train_X.drop(columns=['name','poi','email_address'])


estimator = RandomForestClassifier()
#estimator = DecisionTreeClassifier(random_state=0)
#estimator = XGBClassifier()
score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(score)


estimator.fit(train_X,train_Y)
test_data_source = pd.read_csv('C:\\Users\\Rocky\\Desktop\\test_features.csv')
test_data_source['salary'].fillna(test_data_source['salary'].median(),inplace=True)
test_data_source['bonus'].fillna(test_data_source['bonus'].median(),inplace=True)
test_data_source['long_term_incentive'].fillna(test_data_source['long_term_incentive'].median(),inplace=True)

#test_data_source.fillna(test_data_source.median(),inplace=True)

test_data_source['to_poi_ratio'] = test_data_source['from_poi_to_this_person'] / test_data_source['to_messages']
test_data_source['from_poi_ratio'] = test_data_source['from_this_person_to_poi'] / test_data_source['from_messages']
test_data_source['shared_poi_ratio'] = test_data_source['shared_receipt_with_poi'] / test_data_source['to_messages']

test_data_source.fillna(0,inplace=True)

test_data = test_data_source[fea_col]

predict_res = estimator.predict_proba(test_data)
native_res = predict_res[:,1]
res_df =  pd.DataFrame()
res_df['name'] = test_data_source['name']
res_df['poi'] = native_res
res_df.to_csv('test_padding_median.csv',index=False)







