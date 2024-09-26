import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
#importing the dataset
data = pd.read_csv(r"D:\Data_Science&AI\ClassRoomMaterial\dataset\logit classification.csv")
# Check the data
data.info()
data.head(3)


#X independent variable  features or predictors
X=data.iloc[:,[2,3]].values
#y dependent variable tarhet value
y = data.iloc[:,-1].values

#Split the model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


#This scales the data to have a mean of 0 and a standard deviation of 1
#This scales the data to lie within a specified range, typically between 0 and 1.
#This scales the data to lie within the range -1 to 1 by dividing each value by the maximum absolute value of the feature.

#Scaling the X_train and X_test
from  sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic Regression Comes under Classification 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression() #Classifier is variable name
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

#classification performance metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,classification_report

#Confusion metrix
cm = confusion_matrix(y_test,y_pred)
print('Prediction for the confusion metrix',y_pred)

#find accuracy of the model
accuracy = accuracy_score(y_test,y_pred)
print('Model Accuracy:', accuracy)

#Classification report for the model
class_report = classification_report(y_test,y_pred)
print('Model Classification report:\n \n', class_report)

#bias(mean accuracy on the given test data and labels.)
bias = classifier.score(X_train,y_train)
print(bias)

##variance or validation for the model
variance = classifier.score(X_test,y_test)
print('model Variance or validation :' , variance)

#---------------------------------------------------------------------------
# Find the values by uising Normalizer
from  sklearn.preprocessing import Normalizer
nor = Normalizer()
X_train = nor.fit_transform(X_train)
X_test = nor.fit_transform(X_test)
y_pred1 = classifier.predict(X_test)

#Confusion metrix
cm1 = confusion_matrix(y_test,y_pred1)
print('Prediction for the confusion metrix',y_pred1)

#find accuracy of the model
accuracy1 = accuracy_score(y_test,y_pred1)
print('Model Accuracy:', accuracy1)

#Classification report for the model
class_report1 = classification_report(y_test,y_pred1)
print('Model Classification report:\n \n', class_report1)

#bias(mean accuracy on the given test data and labels.)
bias1 = classifier.score(X_train,y_train)
print(bias)

##variance or validation for the model
variance1 = classifier.score(X_test,y_test)
print('model Variance or validation :' , variance1)

#-----------------------------------------------------------------
#Future prediction or Future data need to pass to the existing model
#Reading another or updated datset to existing model
dataset1 = pd.read_csv(r"D:\Data_Science&AI\ClassRoomMaterial\dataset\Future_prediction2.csv")
dataset1

data1 = dataset1.copy()
data1

#Accessing the updated dataset columns or independent variables
dataset1 = dataset1.iloc[:,[2,3]].values

#Scale the model
sc = StandardScaler()
model = sc.fit_transform(dataset1)

y_pred2 = pd.DataFrame()

data1['y_pred2'] = classifier.predict(model)
print(data1)

dataset_updated = pd.concat([data1, data], axis=0)

data1.to_csv('pred_model1.csv')

import os
os.getcwd()