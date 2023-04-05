#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from preprocess_2nd import preprocessing_2
train_data = pd.read_excel('./drive-download-20221227T023406Z-001/df_total_raw_train.xlsx')
test_data = pd.read_excel('./drive-download-20221227T023406Z-001/df_total_raw_test.xlsx')



train_data_X, train_data_y, test_data_X, test_data_y = preprocessing_2(train_data, test_data)

# In[77]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

forest = RandomForestClassifier()
forest.fit(train_data_X, train_data_y)


# In[62]:


importance = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

print("Feature ranking")

for f in range(train_data_X.shape[1]) :
    print("{}. feature {} ({:.3f})".format(f+1, train_data_X.columns[indices][f], importance[indices[f]]))
    
plt.figure()
plt.title("Feature importance")
plt.bar(range(train_data_X.shape[1]), importance[indices], color='r', align='center')
plt.xticks(range(train_data_X.shape[1]), train_data_X.columns[indices], rotation=45)
plt.xlim([-1, train_data_X.shape[1]])
plt.show()


# In[78]:


from sklearn.metrics import accuracy_score

pred = forest.predict(test_data_X)
accuracy = accuracy_score(test_data_y, pred)
print('{:4f}'.format(accuracy))


# In[79]:


# feature selection
from sklearn.feature_selection import RFECV, RFE

forest2 = RandomForestClassifier()
rfecv = RFECV(estimator=forest2, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv.fit(train_data_X, train_data_y)


# In[80]:


print(rfecv.support_.sum())

print('optimial number of features : %d' % rfecv.n_features_)

plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score (accuracy)")
plt.plot(
range(1, len(rfecv.grid_scores_)*1 + 1, 1), rfecv.grid_scores_,
)


# In[81]:


mask = rfecv.get_support()
features = np.array(train_data_X.columns)
best_features = features[mask]

print('all_features : ', train_data_X.shape[1])
print(features)
print()
print('Selected best : ', best_features.shape[0])
print(features[mask])


# In[83]:


# SELECTTION rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T1_lp1_MIN', 'T1_lp1_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX',
 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
 'T5_Exp in_MIN', 'T5_Exp in_MAX', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX','T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX',
 'center_MIN', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MAX', 'cond air out temp_MIN']]
sfor_x_com= test_data_X.loc[:,['T1_lp1_MIN', 'T1_lp1_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX',
 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
 'T5_Exp in_MIN', 'T5_Exp in_MAX', 'T6_Exp', 'T1_lp1_MIN', 'T1_lp1_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX',
 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
 'T5_Exp in_MIN', 'T5_Exp in_MAX', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX','T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX',
 'center_MIN', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MAX', 'cond air out temp_MIN','out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX','T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX',
 'center_MIN', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MAX', 'cond air out temp_MIN']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[84]:


#회사에서 언급한 것들 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[85]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX',
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX'
                              , 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[86]:


#=입출구 차 , 고내온도, 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['inside temp_MIN', 'inside temp_MAX',
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['inside temp_MIN', 'inside temp_MAX'
                              , 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[87]:


#회사에서 언급한 것들 학습 후 결과(센서 1번) rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'inside temp_MIN', 'inside temp_MAX',
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX',]]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                               'inside temp_MIN', 'inside temp_MAX',
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX', ]]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[88]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과(센서 2번) rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX',
                                'T8_2-T7_MIN', 'T8_2-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX',
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX',
                               'T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[89]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[90]:


sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX',
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX'
                              , 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
forest3 = RandomForestClassifier()
rfecv = RFECV(estimator=forest3, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv.fit(sfor_X_com, train_data_y)


# In[91]:


print(rfecv.support_.sum())

print('optimial number of features : %d' % rfecv.n_features_)

plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score (accuracy)")
plt.plot(
range(1, len(rfecv.grid_scores_)*1 + 1, 1), rfecv.grid_scores_,
)


# In[93]:


mask = rfecv.get_support()
features = np.array(sfor_X_com.columns)
best_features = features[mask]

print('all_features : ', sfor_X_com.shape[1])
print(features)
print()
print('Selected best : ', sfor_X_com.shape[0])
print(features[mask])


# In[94]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX',
 'inside temp_MIN', 'inside temp_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX',
 'inside temp_MIN', 'inside temp_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[95]:


forest4 = RandomForestClassifier()
forest4 .fit(sfor_X_com, train_data_y)


# In[96]:


importance = forest4.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest4.estimators_], axis=0)
indices = np.argsort(importance)[::-1]

print("Feature ranking")

for f in range(sfor_X_com.shape[1]) :
    print("{}. feature {} ({:.3f})".format(f+1, sfor_X_com.columns[indices][f], importance[indices[f]]))
    
plt.figure()
plt.title("Feature importance")
plt.bar(range(sfor_X_com.shape[1]), importance[indices], color='r', align='center')
plt.xticks(range(sfor_X_com.shape[1]), sfor_X_com.columns[indices], rotation=45)
plt.xlim([-1, sfor_X_com.shape[1]])
plt.show()





