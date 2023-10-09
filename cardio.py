import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,classification_report

import seaborn as sns
import matplotlib.pyplot as plt

os.getcwd()

data = pd.read_csv('cardio.csv',sep=',')
data

df = pd.DataFrame(data)
df

df.info()

df.describe()

miss_val_perc = (df.isna().sum()/len(df))*100
miss_val_perc

## checking duplicacy in data and columns ##
df.bp_category_encoded.equals(df.bp_category)



df = df.drop(['id','age','bp_category_encoded'],axis =1)
df

sns.countplot(df, x="cardio")

le = LabelEncoder()
df['bp_category'] = le.fit_transform(df['bp_category'])
df


df['cardio'].value_counts()

x = df.drop(['cardio'],axis =1)
y = df.cardio

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
lr_acc = accuracy_score(y_test,y_pred)
print(lr_acc)
print("classification_report:",classification_report(y_test,y_pred))
cm = confusion_matrix(y_test,y_pred)
print(cm)


cm1 = ConfusionMatrixDisplay(cm,display_labels=['Absence','Presence'])
cm1.plot()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_y_pred = rf.predict(x_test)
rf_acc = accuracy_score(y_test,rf_y_pred)
print(rf_acc)
print("Classification_report:",classification_report(y_test,rf_y_pred))
c2 = confusion_matrix(y_test,rf_y_pred)
print(c2)

cm_rf = ConfusionMatrixDisplay(c2,display_labels=rf.classes_)
cm_rf.plot()

pickle.dump(lr,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,157,93,130,80,3,1,0,0,1,60,37.729,1]]))


