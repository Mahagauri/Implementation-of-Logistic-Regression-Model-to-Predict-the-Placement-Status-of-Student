# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Import the required packages and print the present data.
3. Print the placement data and salary data.
4. Find the null and duplicate values.
5. Using logistic regression find the predicted values of accuracy , confusion matrices.
6. Display the results.
7. End

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mahagauri P
RegisterNumber:  212224040181
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
# DATA HEAD
![image](https://github.com/user-attachments/assets/a8309b60-b4bc-41a2-9c76-9390cf3f9edf)

# DATA1 HEAD
![image](https://github.com/user-attachments/assets/6c3135d7-a14a-430f-b76e-1a5e06752231)

# ISNULL().SUM()
![image](https://github.com/user-attachments/assets/2731368e-c4dc-43cd-a36d-01d6d50fd65b)

# DATA DUPLICATE
![image](https://github.com/user-attachments/assets/e34386cf-b583-4287-93a9-bbb076f46bb9)

# PRINT DATA
![image](https://github.com/user-attachments/assets/bfbcbb12-f20b-4b15-ac7c-5d0a2feb4ef7)

# STATUS
![image](https://github.com/user-attachments/assets/eeef6dc4-0338-46dd-8b49-50368a9fc923)

# Y_PRED
![image](https://github.com/user-attachments/assets/60c80b57-7881-4a58-a5c7-1c1c1df626ee)

# ACCURACY
![image](https://github.com/user-attachments/assets/a9b6665e-1c45-4c00-8a07-7ab38951b005)

# CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/dd3c9124-a45f-4373-98c9-81caaa9072e9)

# CLASSIFICATION
![image](https://github.com/user-attachments/assets/006016bf-83fc-4532-8670-c27fc388c4ec)

# LR PREDICT
![image](https://github.com/user-attachments/assets/ba49351b-796e-490c-a6b1-04d0cdb3469b)









## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
