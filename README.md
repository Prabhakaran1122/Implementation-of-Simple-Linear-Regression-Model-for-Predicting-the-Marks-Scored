# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRABHAKARAN P
RegisterNumber:  212224040236
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('data.csv')
print(df.head(), df.tail())

X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

regressor = LinearRegression().fit(X_train, y_train)

y_pred = regressor.predict(X_test)

for X_data, y_data, title in [(X_train, y_train, "Training Set"), (X_test, y_test, "Testing Set")]:
    plt.scatter(X_data, y_data, color='red')
    plt.plot(X_data, regressor.predict(X_data), color='blue')
    plt.title(f"Hours vs Scores ({title})")
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE = {mse}\nMAE = {mae}\nRMSE = {rmse}")
```

## Output:
df.head()

![360997612-7896a547-4222-4cf9-8267-574447b2b3db](https://github.com/user-attachments/assets/fe29b874-10e8-4764-981e-49e147de935e)


<hr>

df.tail()

![230002671-0141a673-e817-4369-864e-44ed6a01e2b2](https://github.com/user-attachments/assets/16575fc7-192b-4bd0-a3c9-7cc54240922b)

Array values of X

![230002749-aa3741e9-c8fa-4acd-8fe2-3cb807a57d09](https://github.com/user-attachments/assets/d2bb8a96-a2f7-4950-b41f-70fc9916d44e)

Array values of Y

![230002869-92b558f0-dd55-4d06-ace3-53965de41215](https://github.com/user-attachments/assets/b3158d25-8f4b-4916-b759-9d7d17dc8782)



Predicted Values of Y

![230002985-9742f21e-db65-40af-ab25-a17911ac6b4c](https://github.com/user-attachments/assets/9e30cfca-76e5-44c2-8431-bcd0bca03fca)

Training Setgraph:

![230003157-6db35c24-28a2-4411-9252-f658409af75c](https://github.com/user-attachments/assets/6f700a0d-29df-4f58-a22a-663fc3c11821)



Set Testgraph:

![230003242-4e98fb4c-13ab-4b50-85e0-03bc9461d505](https://github.com/user-attachments/assets/c351ca1b-44df-418e-8629-d1540b5ce788)






Values of MSE, MAE and RMSE

![230003341-ce9b56c4-757e-4151-89ee-f1e2cd050923](https://github.com/user-attachments/assets/ccee47fb-59fa-4c6a-907b-9d4f770a29bb)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
