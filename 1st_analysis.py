#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

train_data = pd.read_excel('./drive-download-20221215T064857Z-001/df_total_train.xlsx')
test_data = pd.read_excel('./drive-download-20221215T064857Z-001/df_total_test.xlsx')


# In[2]:


#eva air in temp_MAX	eva air out temp_MIN 에 over 결측치 존재. 결측치가 존재하는 행만 삭제
train_over = train_data[(train_data['eva air out temp_MAX']=='+OVER') | (train_data['eva air out temp_MAX']=='-OVER')].index
train_data.drop(train_over, inplace=True)
train_over2 = train_data[(train_data['eva air out temp_MIN']=='+OVER') | (train_data['eva air out temp_MIN']=='-OVER')].index
train_data.drop(train_over2, inplace=True)

test_over = test_data[(test_data['eva air out temp_MAX']=='+OVER') | (test_data['eva air out temp_MAX']=='-OVER')].index
test_data.drop(test_over, inplace=True)
test_over2 = test_data[(test_data['eva air out temp_MIN']=='+OVER') | (test_data['eva air out temp_MIN']=='-OVER')].index
test_data.drop(test_over2, inplace=True)
test_over3 = test_data[(test_data['outside temp_MAX']=='+OVER')].index
test_data.drop(test_over3, inplace=True)


# In[ ]:





# In[3]:


train_data_X = train_data.iloc[:, :-1]
train_data_y = train_data.iloc[:, -1]

test_data_X = test_data.iloc[:, :-1]
test_data_y = test_data.iloc[:, -1]


# In[62]:


print(train_data)
print(train_data_X)
print(train_data_y)


# In[64]:


print(test_data_y)


# In[63]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

forest = RandomForestClassifier()
forest.fit(train_data_X, train_data_y)





# In[64]:


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


# In[75]:


from sklearn.metrics import accuracy_score

pred = forest.predict(test_data_X)
accuracy = accuracy_score(test_data_y, pred)
print('{:4f}'.format(accuracy))


# Feautre selection : random forest

# In[76]:


# feature selection
from sklearn.feature_selection import RFECV, RFE

forest2 = RandomForestClassifier()
rfecv = RFECV(estimator=forest2, step=1, cv=5, scoring='accuracy', verbose=2)
rfecv.fit(train_data_X, train_data_y)


# In[79]:


print(rfecv.support_.sum())

print('optimial number of features : %d' % rfecv.n_features_)

plt.figure()
plt.xlabel("number of features selected")
plt.ylabel("cross validation score (accuracy)")
plt.plot(
range(1, len(rfecv.grid_scores_)*1 + 1, 1), rfecv.grid_scores_,
)


# In[80]:


print(rfecv.ranking_)


# In[81]:


print(rfecv.estimator_.feature_importances_ )


# In[89]:


mask = rfecv.get_support()
features = np.array(train_data_X.columns)
best_features = features[mask]

print('all_features : ', train_data_X.shape[1])
print(features)
print()
print('Selected best : ', best_features.shape[0])
print(features[mask])


# In[22]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

selected_train_X = train_data_X.loc[:,['T2_comp out_MIN', 'T7_evap in_MAX', 'inside temp_MIN', 'inside temp_MAX', 
                                    'eva air in temp_MIN', 'eva air out temp_MIN']]
selected_test_X = test_data_X.loc[:,['T2_comp out_MIN', 'T7_evap in_MAX', 'inside temp_MIN', 'inside temp_MAX', 
                                    'eva air in temp_MIN', 'eva air out temp_MIN']]
forest_selected = RandomForestClassifier()
forest_selected.fit(selected_train_X, train_data_y)
pred_for_selected = forest_selected.predict(selected_test_X)
saccuracy = accuracy_score(test_data_y, pred_for_selected)
print('{:4f}'.format(saccuracy))


# In[40]:


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


# In[41]:


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


# In[42]:


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


# In[47]:


#회사에서 언급한 것들 + 입출구 차 학습 후(min) 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN','T8_evap out1_MIN', 
                                 'T8_evap out2_MIN',  'inside temp_MIN',
                                 'T8_1-T7_MIN',  'T8_2-T7_MIN']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T8_evap out1_MIN',  
                                 'T8_evap out2_MIN',  'inside temp_MIN', 
                               'T8_1-T7_MIN',  'T8_2-T7_MIN']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[48]:


#회사에서 언급한 것들 + 입출구 차 학습(max) 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MAX',  'T8_evap out1_MAX', 
                                 'T8_evap out2_MAX',  'inside temp_MAX',
                                 'T8_1-T7_MAX',  'T8_2-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MAX',  'T8_evap out1_MAX', 
                                  'T8_evap out2_MAX','inside temp_MAX'
                              , 'T8_1-T7_MAX', 'T8_2-T7_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[49]:


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


# In[50]:


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


# In[52]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX']]
sfor_x_com= test_data_X.loc[:,['T7_evap in_MIN', 'T7_evap in_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[53]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T8_evap out1_MIN', 'T8_evap out1_MAX']]
sfor_x_com= test_data_X.loc[:,['T8_evap out1_MIN', 'T8_evap out1_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[54]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,[
                                 'T8_evap out2_MIN', 'T8_evap out2_MAX']]
sfor_x_com= test_data_X.loc[:,['T8_evap out2_MIN', 'T8_evap out2_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[55]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['inside temp_MIN', 'inside temp_MAX']]
sfor_x_com= test_data_X.loc[:,['inside temp_MIN', 'inside temp_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[56]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['T8_1-T7_MIN', 'T8_1-T7_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# In[57]:


#회사에서 언급한 것들 + 입출구 차 학습 후 결과 rbf 커널 사용
sfor_X_com = train_data_X.loc[:,['T8_2-T7_MIN', 'T8_2-T7_MAX']]
sfor_x_com= test_data_X.loc[:,['T8_2-T7_MIN', 'T8_2-T7_MAX']]
selected_for = RandomForestClassifier()
selected_for.fit(sfor_X_com , train_data_y)
pred_for_for = selected_for.predict(sfor_x_com)
for_acc= accuracy_score(test_data_y, pred_for_for)
print('{:4f}'.format(for_acc))


# random forest의 경우, 출입구 온도센서 값 + 고내 온도 + 출입구 온도센서 차이를 사용했을 때, 가장 높은 정확도를 보임
# 
# min, max값을 따로 사용할 경우, 정확도가 하락함. 만약, 하나만 사용한다면 max값이 더 나을듯
# 
# 센서를 하나만 쓴다면, 2번센서가 좋음
# 
# 값을 하나만 쓴다면 2번 센서와의 차이가 가장 좋음

# feature selection : SVM

# In[6]:


# feature selection
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV, RFE

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




