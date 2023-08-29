#!/usr/bin/env python
# coding: utf-8

# In[256]:


#importing necessary libraries
import pandas as pd
import numpy as np


# In[257]:


#Reading data
wine = pd.read_csv("wine.csv")
wine.head(20)


# In[258]:


#Printing number of row and columns
wine.shape


# In[259]:


#checking number of nulls in a column 
wine.isnull().sum()


# In[260]:


#Encoding categorical features:
wine.info()


# In[261]:


#printing catagorical feature
wine['quality'].unique()


# In[262]:


#Label Encoder
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()

# Apply the encoding
wine["quality_enc"] = lenc.fit_transform(wine["quality"])

# Compare the two columns
print(wine[["quality", "quality_enc"]].head())
wine.shape


# In[263]:


from sklearn.preprocessing import MinMaxScaler

wine= wine.drop(['quality'], axis = 1)
list_of_features = ["fixed acidity","residual sugar","free sulfur dioxide","total sulfur dioxide","pH","alcohol"]

scaler = MinMaxScaler()
scaler.fit(wine[list_of_features])


wine[list_of_features] = scaler.transform(wine[list_of_features])


# In[264]:


#Min Max Scaling
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
scl.fit(wine)
wine_scl = scl.transform(wine)
wine_scl


# In[265]:


#preparing training set
x = wine.drop("quality_enc", axis = 1)
y =  wine["quality_enc"]


# In[266]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)
print(X_train.shape)
print(X_test.shape)


# In[267]:


#Performing Linear Regression
from sklearn.linear_model import LinearRegression
lir = LinearRegression()
lir.fit(X_train, Y_train)
pred = lir.predict(X_test)
pred


# In[268]:


#Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
#train the model
lr.fit(X_train, Y_train) 
predictions = lr.predict(X_test)
print(predictions)




# In[269]:


from sklearn.metrics import accuracy_score
lscore = (accuracy_score(Y_test, predictions))
print(lscore)


# In[270]:


# Creating Decision Tree

import seaborn as sbn
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier(criterion='entropy', random_state=1)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
score = accuracy_score(Y_pred,Y_test)
score


# In[271]:


#Comparing the accuracy in a Bar chart
import matplotlib.pyplot as plt
 

barWidth = 0.20
fig = plt.subplots(figsize =(12, 8))
logistic = [lscore]
dt = [score]
 

br1 = np.arange(len(logistic))
br2 = [x + barWidth for x in br1]
 

plt.bar(br1, logistic, color ='g', width = barWidth,
        edgecolor ='grey', label ='Logistic')
plt.bar(br2, dt, color ='b', width = barWidth,
        edgecolor ='grey', label ='Decision Tree')

plt.ylabel('Accuracy Score', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(logistic))],
        ['Logistic Vs Decision Tree'])
 
plt.legend()
plt.show()


# In[272]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

v1 = SVC(kernel="linear")

v1.fit(X_train, Y_train)

y_pred1 = v1.predict(X_test)

score_1 = accuracy_score(y_pred1, Y_test)
score_1

from sklearn.neural_network import MLPClassifier

v2 = MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=11000)
v2.fit(X_train, Y_train)

y_pred2 = v2.predict(X_test)

score_2 = accuracy_score(y_pred2, Y_test)
score_2

from sklearn.ensemble import RandomForestClassifier

v3 = RandomForestClassifier(n_estimators=50)
v3.fit(X_train, Y_train)

y_pred3 = v3.predict(X_test)

score_3 = accuracy_score(y_pred3, Y_test)
score_3

div = (wine.columns.shape[0]-1)//2
div

from sklearn.decomposition import PCA 
pca = PCA(n_components=div)

principal_components = pca.fit_transform(x)
principal_components

pca.explained_variance_ratio_

sum(pca.explained_variance_ratio_)

principal_df = pd.DataFrame(data=principal_components, columns=["principle component 1","principle component 2","principle component 3","principle component 4","principle component 5"])

main_df=pd.concat([principal_df,wine[["quality_enc"]]], axis=1)

main_df.head(10)


from sklearn.svm import SVC

v1 = SVC(kernel="linear")

v1.fit(X_train, Y_train)

y_pred1 = v1.predict(X_test)

score_11 = accuracy_score(y_pred1, Y_test)
score_11

from sklearn.neural_network import MLPClassifier

v2 = MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=11000)
v2.fit(X_train, Y_train)

y_pred2 = v2.predict(X_test)

score_22 = accuracy_score(y_pred2, Y_test)
score_22

from sklearn.ensemble import RandomForestClassifier

v3 = RandomForestClassifier(n_estimators=50)
v3.fit(X_train, Y_train)

y_pred3 = v3.predict(X_test)

score_33 = accuracy_score(y_pred3, Y_test)
score_33
 

barWidth = 0.20
fig = plt.subplots(figsize =(12, 8)) 

svm = [score_1, score_11]
mlp = [score_2, score_22]
rfc = [score_3, score_33]

br1 = np.arange(len(svm))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 

plt.bar(br1, svm, color ='c', width = barWidth,
        edgecolor ='grey', label ='SVM')
plt.bar(br2, mlp, color ='y', width = barWidth,
        edgecolor ='grey', label ='MLP')
plt.bar(br3, rfc, color ='m', width = barWidth,
        edgecolor ='grey', label ='RFC')
 
#Xticks
plt.ylabel('Score', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(svm))],
        ['Before PCA', 'After PCA'])
 
plt.legend()
plt.show()

