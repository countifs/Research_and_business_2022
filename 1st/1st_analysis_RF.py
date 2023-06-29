#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from preprocess_1st import preporcessing
train_data = pd.read_excel('-')
test_data = pd.read_excel('-')


# In[2]:

train_data_X, train_data_y, test_data_X,  test_data_y = preporcessing(train_data, test_data) #데이터 전처리



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






