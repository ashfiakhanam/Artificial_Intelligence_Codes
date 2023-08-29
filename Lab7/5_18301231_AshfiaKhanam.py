# %%
#importing necessary libraries
import pandas as pd
import numpy as np

# %%
#Reading data
wine = pd.read_csv("wine.csv")
wine.head(20)

# %%
#Printing number of row and columns
wine.shape

# %%
#checking number of nulls in a column 
wine.isnull().sum()


# %%
#Encoding categorical features:
wine.info()

# %%
#printing catagorical feature
wine['quality'].unique()


# %%
#Label Encoder
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()

# Apply the encoding
wine["quality_enc"] = lenc.fit_transform(wine["quality"])

# Compare the two columns
print(wine[["quality", "quality_enc"]].head())
wine.shape

# %%
from sklearn.preprocessing import MinMaxScaler

wine= wine.drop(['quality'], axis = 1)
list_of_features = ["fixed acidity","residual sugar","free sulfur dioxide","total sulfur dioxide","pH","alcohol"]

scaler = MinMaxScaler()
scaler.fit(wine[list_of_features])


wine[list_of_features] = scaler.transform(wine[list_of_features])

# %%
#Min Max Scaling
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()
scl.fit(wine)
wine_scl = scl.transform(wine)
wine_scl

# %%
#preparing training set
x = wine.drop("quality_enc", axis = 1)
y =  wine["quality_enc"]


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)
print(X_train.shape)
print(X_test.shape)


# %%
#Performing Linear Regression
from sklearn.linear_model import LinearRegression
lir = LinearRegression()
lir.fit(X_train, Y_train)
pred = lir.predict(X_test)
pred

# %%
#Performing Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
#train the model
lr.fit(X_train, Y_train) 
predictions = lr.predict(X_test)
print(predictions)





# %%
from sklearn.metrics import accuracy_score
lscore = (accuracy_score(Y_test, predictions))
print(lscore)


# %%
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

# %%
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


