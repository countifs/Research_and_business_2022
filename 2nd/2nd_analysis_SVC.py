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




