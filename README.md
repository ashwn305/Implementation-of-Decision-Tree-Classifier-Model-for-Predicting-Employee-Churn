# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ASHWIN H 
RegisterNumber: 212225230024
*/
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['stayed','left'],filled=True)
plt.show()

```

## Output:
<img width="1248" height="242" alt="EXP8 ML 1 IMAGE" src="https://github.com/user-attachments/assets/54f59e41-889c-45db-9542-8ae6f8129eb5" />
<img width="564" height="389" alt="exp8 ml 2 image" src="https://github.com/user-attachments/assets/5fd05ebf-57ba-4655-8238-0d30388f7aad" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
