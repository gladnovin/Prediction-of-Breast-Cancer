#!usr/bin/python
#Packages
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

cancer=pd.read_csv("data.csv")
d=cancer.sort_values(by=['id'],ascending=True)
d.reset_index(drop=True,inplace=True)
d=d.drop("Unnamed: 32",axis=1) 
e=d['diagnosis'].value_counts()
print(e)
col=['radius_mean','texture_mean','perimeter_mean','radius_se','texture_se','perimeter_se','radius_worst','texture_worst','perimeter_worst']
X=cancer[col]
cancer['diagnosis'] = cancer['diagnosis'].map({'M':1,'B':0})
y=cancer.diagnosis
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=40)
model=GaussianNB()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
accuracy = accuracy_score(y_test,prediction)
print 'Accuracy of Naive Bayes: \n', accuracy, '\n'
confusion=confusion_matrix(y_test,prediction)
TP=confusion[1][1]
TN=confusion[0][0]
FP=confusion[0][1]
FN=confusion[1][0]
print 'confusion matrix of Naive Bayes: \n',confusion,'\n'
report=classification_report(y_test,prediction,digits=3)
print 'Precision of Naive Bayes:\n',report,'\n'
model1=LogisticRegression(solver='liblinear')
model1.fit(X_train,y_train)
prediction1=model1.predict(X_test)
accuracy1=accuracy_score(y_test,prediction1)
print'Accuracy of Logistic Regression:\n', accuracy1, '\n'
confusion1=confusion_matrix(y_test,prediction1)
report1=classification_report(y_test,prediction1,digits=3)
print 'confusion matrix of Logistic Regression: \n', confusion1,'\n'
print 'Precision of Logistic Regression: \n',report1,'\n'
y_p=model.predict_proba(X_test)[:,1]
d['radius_mean'][d['diagnosis']=='M'].plot.hist(alpha=0.5,color='red')
d['radius_mean'][d['diagnosis']=='B'].plot.hist(alpha=0.5,color='blue')
d['texture_worst'][d['diagnosis']=='M'].plot.hist(alpha=0.5,color='red')
d['texture_worst'][d['diagnosis']=='B'].plot.hist(alpha=0.5,color='blue')
d['perimeter_worst'][d['diagnosis']=='M'].plot.hist(alpha=0.5,color='red')
d['perimeter_worst'][d['diagnosis']=='B'].plot.hist(alpha=0.5,color='blue')
plt.show()
plt.rcParams['font.size']=14
plt.hist(y_p,bins=8)
plt.xlim(0,1)
plt.xlabel('Predicted Malignant')
plt.ylabel('Frequency')
g=plt.show()




