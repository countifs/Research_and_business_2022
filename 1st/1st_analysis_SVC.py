#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from preprocess_1st import preporcessing
train_data = pd.read_excel('-')
test_data = pd.read_excel('-')


# In[2]:

train_data_X, train_data_y, test_data_X,  test_data_y = preporcessing(train_data, test_data) #데이터 전처리



# feature selection : SVM

# In[6]:


# feature selection
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import accuracy_score
svc = SVC(kernel='linear')
rfecv2 = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv2.fit(train_data_X, train_data_y)


# In[11]:


import matplotlib.pyplot as plt
print(rfecv2.support_.sum())

print('optimial number of features : %d' % rfecv2.n_features_)

plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score (accuracy)")
plt.plot(
range(1, len(rfecv2.grid_scores_)*1 + 1, 1), rfecv2.grid_scores_,
)


# In[13]:


import numpy as np
mask2 = rfecv2.get_support()
features = np.array(train_data_X.columns)
best_features2 = features[mask2]

print('all_features : ', train_data_X.shape[1])
print(features)
print()
print('Selected best : ', best_features2.shape[0])
print(features[mask2])


# In[24]:


#선택된 피처로 학습 후 결과

ssvm_train_X = train_data_X.loc[:,['center_MAX', 'eva air in temp_MIN']]
ssvm_test_X = test_data_X.loc[:,['center_MAX', 'eva air in temp_MIN']]
selected_svc = SVC(kernel='linear')
selected_svc.fit(ssvm_train_X, train_data_y)
pred_for_ssvc = selected_svc.predict(ssvm_test_X )
saccuracy2 = accuracy_score(test_data_y, pred_for_ssvc)
print('{:4f}'.format(saccuracy2))


# In[26]:


#선택된 피처로 학습 후 결과 rbf 커널 사용
ssvm_train_X = train_data_X.loc[:,['center_MAX', 'eva air in temp_MIN']]
ssvm_test_X = test_data_X.loc[:,['center_MAX', 'eva air in temp_MIN']]
selected_svc2 = SVC(kernel='rbf')
selected_svc2.fit(ssvm_train_X, train_data_y)
pred_for_ssvc2 = selected_svc2.predict(ssvm_test_X )
saccuracy3 = accuracy_score(test_data_y, pred_for_ssvc2)
print('{:4f}'.format(saccuracy3))


# In[29]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[31]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[66]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                                 , 'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'
                               ,'inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[43]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX','inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX','inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[44]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[34]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[35]:


#회사에서 언급한 것들 (min) 학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T8_evap out1_MIN', 
                                 'T8_evap out2_MIN', 'inside temp_MIN' ]]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T8_evap out1_MIN',  
                                 'T8_evap out2_MIN',  'inside temp_MIN']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[36]:


#회사에서 언급한 것들 (max)학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MAX', 'T8_evap out1_MAX', 
                                  'T8_evap out2_MAX', 'inside temp_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MAX', 'T8_evap out1_MAX', 
                                  'T8_evap out2_MAX', 'inside temp_MAX']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[37]:


#회사에서 언급한 것들(출구 센서 1번만 사용) 학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                  'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'inside temp_MIN', 'inside temp_MAX']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[67]:


#회사에서 언급한 것들(출구 센서 1번만 사용) 학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX','inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX', 'T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 'T8_1-T7_MIN', 'T8_1-T7_MAX','inside temp_MIN', 'inside temp_MAX']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[38]:


#회사에서 언급한 것들(출구센서 2번만 사용) 학습 후 결과 rdf 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX',  
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX',
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX', 'inside temp_MIN', 'inside temp_MAX']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[39]:


#회사에서 언급한 것들(출구 센서 1번만 사용, min) 학습 후 결과 rbf 커널 사용
ssvm_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T8_evap out1_MIN',  
                                  'inside temp_MIN' ]]
ssvm_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T8_evap out1_MIN', 
                                 'inside temp_MIN']]
selected_svc3 = SVC(kernel='rbf')
selected_svc3.fit(ssvm_X_com, train_data_y)
pred_for_ssvc3 = selected_svc3.predict(ssvm_x_com)
saccuracy4 = accuracy_score(test_data_y, pred_for_ssvc3)
print('{:4f}'.format(saccuracy4))


# In[58]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[59]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 ]]
ssvm_x_com2= test_data_X.loc[:,['T8_evap out1_MIN', 'T8_evap out1_MAX', 
                                 ]]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[60]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,[
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
ssvm_x_com2= test_data_X.loc[:,[
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[61]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX',]]
ssvm_x_com2= test_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[62]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['T8_2-T7_MIN', 'T8_2-T7_MAX']]
ssvm_x_com2= test_data_X.loc[:,['T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# In[63]:


#회사에서 언급한 것들로 학습 후 결과 rbf 커널 사용, 차 사용
ssvm_X_com2 = train_data_X.loc[:,['inside temp_MIN', 'inside temp_MAX']]
ssvm_x_com2= test_data_X.loc[:,['inside temp_MIN', 'inside temp_MAX']]
selected_svc4 = SVC(kernel='rbf')
selected_svc4.fit(ssvm_X_com2, train_data_y)
pred_for_ssvc4 = selected_svc4.predict(ssvm_x_com2)
saccuracy5 = accuracy_score(test_data_y, pred_for_ssvc4)
print('{:4f}'.format(saccuracy5))


# svm 은 센서를 사용할 때, 회사에서 언급한 센서 (입출구 센서, 고내온도, 입출구 차이)를 사용할 때, min 값을 사용하면 성능이 올라감
# 
# 만약 min, max를 다 같이 사용할 경우, 출구 센서 1번만 사용하는 것이 더 유리함
# 
# 출구 센서 1번만 사용하면서 min 값만 사용할 경우, 성능이 더 상승하는 것을 확인함
# 

# In[ ]:




