import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
import matplotlib.pyplot as plt
# Regression
# 1.0 = perfect
# 0.0 = bad
# negative = very bad

# Mae (mean absolute error)
# real = 80
# pred = 75
# error = 5
# MSE (Mean Squared Error)
file = 'student_performance.csv'
data = pd.read_csv(file, on_bad_lines='skip')

df = pd.DataFrame(data)
des = df.describe()
# print(des)

print(df.head())
print(df.info())

X = df[["hours_study", "sleep_hours", "phone_usage"]]
y = df["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42) # X is in put, y is output

model = LinearRegression()
model.fit(X_train, y_train)

re_score = model.score(X_test, y_test)
print(f"Model R-squared: {re_score:.2f}")
print(model.coef_)
print(model.intercept_)


def prediction(num1, num2, num3):
    new_data = [[num1, num2, num3]]
    pre = pd.DataFrame(new_data, columns=[
        "hours_study", "sleep_hours", "phone_usage"])
    result = model.predict(pre)
    return result


print("\n---Prediction---\n")
