#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

train_data = pd.read_excel('./drive-download-20221227T023406Z-001/df_total_raw_train.xlsx')
test_data = pd.read_excel('./drive-download-20221227T023406Z-001/df_total_raw_test.xlsx')


# In[2]:


train_label = train_data.loc[:,'Heat_ON']
train_set = list(set(train_label))


# In[3]:


from tqdm import tqdm

train_datas = train_data.iloc[:]
print( train_datas.iloc[1].index)

for each_index in ['T1_lp1_MIN', 'T1_lp1_MAX', 'T2_hp1_MIN', 'T2_hp1_MAX', 'T3_hp2_MIN',
       'T3_hp2_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX', 'T2_comp out_MIN',
       'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
       'T4_cond out_MIN', 'T4_cond out_MAX', 'T5_Exp in_MIN', 'T5_Exp in_MAX',
       'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T7_evap in_MIN', 'T7_evap in_MAX',
       'T8_evap out1_MIN', 'T8_evap out1_MAX', 'T8_evap out2_MIN',
       'T8_evap out2_MAX', 'T10_sol out_MIN', 'T10_sol out_MAX',
       'inside temp_MIN', 'inside temp_MAX', 'center_MIN', 'center_MAX',
       'outside temp_MIN', 'outside temp_MAX', 'eva air in temp_MIN',
       'eva air in temp_MAX', 'eva air out temp_MIN', 'eva air out temp_MAX',
       'cond air in temp_MIN', 'cond air in temp_MAX', 'cond air out temp_MIN',
       'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'] :
    print(each_index)
    data_over = train_data[(train_data[each_index] == '+OVER')].index
    train_data.drop(data_over, inplace=True)
    data_over2 = train_data[(train_data[each_index] == '-OVER')].index
    train_data.drop(data_over2, inplace=True)

for each_index2 in ['T1_lp1_MIN', 'T1_lp1_MAX', 'T2_hp1_MIN', 'T2_hp1_MAX', 'T3_hp2_MIN',
       'T3_hp2_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX', 'T2_comp out_MIN',
       'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
       'T4_cond out_MIN', 'T4_cond out_MAX', 'T5_Exp in_MIN', 'T5_Exp in_MAX',
       'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T7_evap in_MIN', 'T7_evap in_MAX',
       'T8_evap out1_MIN', 'T8_evap out1_MAX', 'T8_evap out2_MIN',
       'T8_evap out2_MAX', 'T10_sol out_MIN', 'T10_sol out_MAX',
       'inside temp_MIN', 'inside temp_MAX', 'center_MIN', 'center_MAX',
       'outside temp_MIN', 'outside temp_MAX', 'eva air in temp_MIN',
       'eva air in temp_MAX', 'eva air out temp_MIN', 'eva air out temp_MAX',
       'cond air in temp_MIN', 'cond air in temp_MAX', 'cond air out temp_MIN',
       'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'] :
    
    data_over3 = test_data[(test_data[each_index2] == '+OVER')].index
    test_data.drop(data_over3, inplace=True)
    data_over4 = test_data[(test_data[each_index2] == '-OVER')].index
    test_data.drop(data_over4, inplace=True)

# train_over = train_data[(train_data['eva air out temp_MAX']=='+OVER') | (train_data['eva air out temp_MAX']=='-OVER')].index
# train_data.drop(train_over, inplace=True)
# train_over2 = train_data[(train_data['eva air out temp_MIN']=='+OVER') | (train_data['eva air out temp_MIN']=='-OVER')].index
# train_data.drop(train_over2, inplace=True)

# test_over = test_data[(test_data['eva air out temp_MAX']=='+OVER') | (test_data['eva air out temp_MAX']=='-OVER')].index
# test_data.drop(test_over, inplace=True)
# test_over2 = test_data[(test_data['eva air out temp_MIN']=='+OVER') | (test_data['eva air out temp_MIN']=='-OVER')].index
# test_data.drop(test_over2, inplace=True)
# test_over3 = test_data[(test_data['outside temp_MAX']=='+OVER')].index
# test_data.drop(test_over3, inplace=True)
# print('Off' in list(train_datas.iloc[1]))

# for a in tqdm(range(len(train_datas))) :
#     data = list(train_datas.iloc[a])
    
#     if '+OVER' in data or '-OVER' in data :

        


# In[15]:



for i in train_set :
    count = list(train_label).count(i)
    print(i)
    print(count)


# In[33]:


train_label = train_data.loc[:,'Heat_ON']
train_set = list(set(train_label))
for i in train_set :
    count = list(train_label).count(i)
    print(i)
    print(count)


# In[4]:


train_label_60 = train_data.loc[(train_data['Heat_ON'] == 'Low(60%)')]
train_label_70 = train_data.loc[(train_data['Heat_ON'] == 'Low(70%)')]
train_label_80 = train_data.loc[(train_data['Heat_ON'] == 'Low(80%)')]
train_label_90 = train_data.loc[(train_data['Heat_ON'] == 'Low(90%)')]
train_label_on = train_data.loc[(train_data['Heat_ON'] == 'On')]
train_label_off = train_data.loc[(train_data['Heat_ON'] == 'Off')]


# In[5]:


print(len(train_label_60))
print(len(train_label_70))
print(len(train_label_80))
print(len(train_label_90))
print(len(train_label_on))
print(len(train_label_off))


# In[6]:


train_lablel_60 = train_label_60.sample(n=404)
train_lablel_70 = train_label_70.sample(n=404)
train_lablel_80 = train_label_80.sample(n=404)
train_lablel_on = train_label_on.sample(n=404)
train_lablel_off = train_label_off.sample(n=404)


# In[49]:


print(len(train_lablel_60))
print(len(train_lablel_70))
print(len(train_lablel_80))
print(len(train_label_90))
print(len(train_lablel_on))
print(len(train_lablel_off))


# In[7]:


sampled_train_data = pd.concat([train_lablel_60, train_lablel_70, train_lablel_80, train_label_90, train_lablel_on, train_lablel_off])


# In[8]:


sampled_train_label_60 = sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'Low(60%)')]
sampled_train_label_70 = sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'Low(70%)')]
sampled_train_label_80 = sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'Low(80%)')]
sampled_train_label_90 = sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'Low(90%)')]
sampled_train_label_on = sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'On')]
sampled_train_label_off = sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'Off')]


# In[9]:


print(len(sampled_train_label_60))
print(len(sampled_train_label_70))
print(len(sampled_train_label_80))
print(len(sampled_train_label_90))
print(len(sampled_train_label_on))
print(len(sampled_train_label_off))


# In[10]:


print(len(sampled_train_data.loc[(sampled_train_data['Heat_ON'] == 'Low(60%)')]))


# In[11]:


sampled_train_data.to_csv('./test.csv')


# In[12]:


train_data_X = sampled_train_data.iloc[:, :-1]
train_data_y = sampled_train_data.iloc[:, -1]

test_data_X = test_data.iloc[:, :-1]
test_data_y = test_data.iloc[:, -1]


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
 'T5_Exp in_MIN', 'T5_Exp in_MAX', 'T6_Exp'T1_lp1_MIN', 'T1_lp1_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX',
 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
 'T5_Exp in_MIN', 'T5_Exp in_MAX', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX','T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX',
 'center_MIN', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MAX', 'cond air out temp_MIN' out_MAX', 'T8_evap out1_MIN',
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


# In[13]:


# feature selection
from sklearn.svm import SVC

svc = SVC(kernel='linear')
svc.fit(train_data_X, train_data_y)

from sklearn.metrics import accuracy_score

pred = svc.predict(test_data_X)
accuracy = accuracy_score(test_data_y, pred)
print('{:4f}'.format(accuracy))


# In[97]:


# feature selection
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, RFE

svc = SVC(kernel='linear')
rfecv2 = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv2.fit(train_data_X, train_data_y)


# In[98]:


import matplotlib.pyplot as plt
print(rfecv2.support_.sum())

print('optimial number of features : %d' % rfecv2.n_features_)

plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score (accuracy)")
plt.plot(
range(1, len(rfecv2.grid_scores_)*1 + 1, 1), rfecv2.grid_scores_,
)


# In[99]:


import numpy as np
mask2 = rfecv2.get_support()
features = np.array(train_data_X.columns)
best_features2 = features[mask2]

print('all_features : ', train_data_X.shape[1])
print(features)
print()
print('Selected best : ', best_features2.shape[0])
print(features[mask2])


# In[124]:


x = ['T1_lp1_MIN', 'T2_hp1_MIN', 'T3_hp2_MIN', 'T3_hp2_MAX', 'T1_comp in_MIN',
 'T1_comp in_MAX', 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN',
 'T3_cond in_MAX', 'T4_cond out_MIN', 'T4_cond out_MAX','T5_Exp in_MIN',
 'T5_Exp in_MAX', 'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX',
 'T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX', 'center_MIN',
 'center_MAX', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MIN', 'eva air out temp_MAX', 'cond air in temp_MIN',
 'cond air out temp_MIN', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN',
 'T8_2-T7_MAX']
x2 = ['T1_lp1_MIN', 'T1_lp1_MAX', 'T2_hp1_MIN', 'T2_hp1_MAX', 'T3_hp2_MIN',
 'T3_hp2_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX', 'T2_comp out_MIN',
 'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX', 'T4_cond out_MIN',
 'T4_cond out_MAX', 'T5_Exp in_MIN', 'T5_Exp in_MAX', 'T6_Exp out_MIN',
 'T6_Exp out_MAX', 'T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX',
 'T10_sol out_MIN', 'T10_sol out_MAX', 'inside temp_MIN', 'inside temp_MAX',
 'center_MIN', 'center_MAX', 'outside temp_MIN', 'outside temp_MAX',
 'eva air in temp_MIN', 'eva air in temp_MAX', 'eva air out temp_MIN',
 'eva air out temp_MAX' ,'cond air in temp_MIN', 'cond air in temp_MAX',
 'cond air out temp_MIN', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN',
 'T8_2-T7_MAX']

for aa in x2 :
    if aa not in x :
        print(aa)


# In[101]:


#선택된 피처로 학습 후 결과

ssvm_train_X = train_data_X.loc[:,['T1_lp1_MIN', 'T2_hp1_MIN', 'T3_hp2_MIN', 'T3_hp2_MAX', 'T1_comp in_MIN',
 'T1_comp in_MAX', 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN',
 'T3_cond in_MAX', 'T4_cond out_MIN', 'T4_cond out_MAX','T5_Exp in_MIN',
 'T5_Exp in_MAX', 'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX',
 'T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX', 'center_MIN',
 'center_MAX', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MIN', 'eva air out temp_MAX', 'cond air in temp_MIN',
 'cond air out temp_MIN', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN',
 'T8_2-T7_MAX']]
ssvm_test_X = test_data_X.loc[:,['T1_lp1_MIN', 'T2_hp1_MIN', 'T3_hp2_MIN', 'T3_hp2_MAX', 'T1_comp in_MIN',
 'T1_comp in_MAX', 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN',
 'T3_cond in_MAX', 'T4_cond out_MIN', 'T4_cond out_MAX','T5_Exp in_MIN',
 'T5_Exp in_MAX', 'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX',
 'T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX', 'center_MIN',
 'center_MAX', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MIN', 'eva air out temp_MAX', 'cond air in temp_MIN',
 'cond air out temp_MIN', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN',
 'T8_2-T7_MAX']]
selected_svc = SVC(kernel='linear')
selected_svc.fit(ssvm_train_X, train_data_y)
pred_for_ssvc = selected_svc.predict(ssvm_test_X )
saccuracy2 = accuracy_score(test_data_y, pred_for_ssvc)
print('{:4f}'.format(saccuracy2))


# In[77]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

forest = RandomForestClassifier()
forest.fit(train_data_X, train_data_y)


# In[102]:


selected_svc = SVC(kernel='linear')
selected_svc.fit(train_data_X, train_data_y)
pred_for_ssvc = selected_svc.predict(test_data_X)
saccuracy2 = accuracy_score(test_data_y, pred_for_ssvc)
print('{:4f}'.format(saccuracy2))


# In[128]:


#선택된 피처로 학습 후 결과

ssvm_train_X = train_data_X.loc[:,['T1_lp1_MIN', 'T2_hp1_MIN', 'T3_hp2_MIN', 'T3_hp2_MAX', 'T1_comp in_MIN',
 'T1_comp in_MAX', 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN',
 'T3_cond in_MAX', 'T4_cond out_MIN', 'T4_cond out_MAX','T5_Exp in_MIN',
 'T5_Exp in_MAX', 'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX',
 'T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX', 'center_MIN',
 'center_MAX', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MIN', 'eva air out temp_MAX', 'cond air in temp_MIN',
 'cond air out temp_MIN', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN',
 'T8_2-T7_MAX']]
ssvm_test_X = test_data_X.loc[:,['T1_lp1_MIN', 'T2_hp1_MIN', 'T3_hp2_MIN', 'T3_hp2_MAX', 'T1_comp in_MIN',
 'T1_comp in_MAX', 'T2_comp out_MIN', 'T2_comp out_MAX', 'T3_cond in_MIN',
 'T3_cond in_MAX', 'T4_cond out_MIN', 'T4_cond out_MAX','T5_Exp in_MIN',
 'T5_Exp in_MAX', 'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T8_evap out1_MIN',
 'T8_evap out1_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX',
 'T10_sol out_MIN', 'inside temp_MIN', 'inside temp_MAX', 'center_MIN',
 'center_MAX', 'eva air in temp_MIN', 'eva air in temp_MAX',
 'eva air out temp_MIN', 'eva air out temp_MAX', 'cond air in temp_MIN',
 'cond air out temp_MIN', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN',
 'T8_2-T7_MAX']]
selected_svc = SVC(kernel='rbf')
selected_svc.fit(ssvm_train_X, train_data_y)
pred_for_ssvc = selected_svc.predict(ssvm_test_X )
saccuracy2 = accuracy_score(test_data_y, pred_for_ssvc)
print('{:4f}'.format(saccuracy2))


# In[129]:


ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
selected_svc3 = SVC(kernel='linear')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[127]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_svc4 = SVC(kernel='linear')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[107]:


#회사에서 언급한 것들로 학습 후 결과 선형커널 사용, 차 사용 고내온도 사용
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                                 , 'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='linear')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[108]:


#회사에서 언급한 것들로 학습 후 결과 선형 커널 사용, 차, 고내온도만 사용
ssvm_X_com2 = train_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX','inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX','inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='linear')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[109]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 + 고내온도 , 센서 사용 1번
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                  'T8_1-T7_MIN', 'T8_1-T7_MAX'
                                 , 'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX'
                               ,'inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='linear')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[111]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용,차 + 고내온도 , 센서 사용 2번
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                                 , 'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='linear')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[115]:


from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, RFE

ssvm_train_X = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN', 'inside temp_MAX']]
ssvm_test_X = test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN', 'inside temp_MAX']]

svc = SVC(kernel='linear')
rfecv2 = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv2.fit(ssvm_train_X, train_data_y)


# In[116]:


import matplotlib.pyplot as plt
print(rfecv2.support_.sum())

print('optimial number of features : %d' % rfecv2.n_features_)

plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score (accuracy)")
plt.plot(
range(1, len(rfecv2.grid_scores_)*1 + 1, 1), rfecv2.grid_scores_,
)


# In[118]:


import numpy as np
mask2 = rfecv2.get_support()
features = np.array(ssvm_train_X.columns)
best_features2 = features[mask2]

print('all_features : ', ssvm_train_X.shape[1])
print(features)
print()
print('Selected best : ', best_features2.shape[0])
print(features[mask2])


# In[119]:


ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN']]
selected_svc4 = SVC(kernel='linear')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[ ]:




