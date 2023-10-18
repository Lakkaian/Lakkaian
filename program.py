import numpy as np 
import pandas as pd   
import seaborn as sns  
import matplotlib.pyplot as plt 
from collections import Counter  
import os  

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score  
from sklearn.preprocessing import QuantileTransformer  
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier  
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split  
from sklearn.svm import SVC
data = pd.read_csv("dataset")
data['Glucose'] = data['Glucose'].replace(0, data['Glucose'].median()
data['BMI'] = data['BMI'].replace(0, data['BMI'].mean())
                                          data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].mean())  
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].mean())
                                          data.describe() 
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].median()
data.head()
data.isnull().sum()
                                                                                                
