# Ml workflow
# 1. Import libraries
# 2. Load data
# 3. Clean data
# 4. Explore data (describe, info)
# 5. Visualize
# 6. Define X and y
# 7. Train/test split
# 8. Train model
# 9. Evaluate model
# 10. Predict function
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import random
import os
# x = daily self study hours, attendance rate, subject, screen time, sleep hours
# y = total score, grade
file = 'student_performance.csv'

data = pd.read_csv(file)

# Convert to dataframe
df = pd.DataFrame(data)
print(df)

# # Drop the useless data
# data = df.drop(
#     columns=['student_id', 'class_participation'])
# # print(data.describe())

# # Devide data to daily
# data['daily_study_hours'] = data['weekly_self_study_hours'] / 7

# data['high_study'] = np.where(data['daily_study_hours'] > 3, 1, 0)
# data['high_attendance_rate'] = np.where(
#     data['attendance_percentage'] > 80, 1, 0)

# # Add extra data
# data['screen_time'] = (8 - (data['daily_study_hours'] * 1.0)).clip(lower=0)
# data['sleep_hours'] = (
#     24 - 10 - data['daily_study_hours'] - data['screen_time']).clip(lower=5)

# # Train
# X = data[['sleep_hours', 'screen_time', 'high_attendance_rate']]
# y = data['total_score']

# # Scale features to handle multicollinearity better
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42)

# model = Ridge(alpha=10.0)  # a blank brain with regularization
# model.fit(X_train, y_train)  # read 10000 data


# re_score = model.score(X_test, y_test)  # the final grade R square
# y_pred = model.predict(X_test)
# print(
#     "Coefficients [sleep_hours, screen_time, high_attendance_rate, high_study]:")
# print(model.coef_)
# print(f"Accuracy: {re_score:.4f}")

# result = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred
# })

# sample_data = result.sample(50)
# plt.figure(figsize=(9, 7))

# plt.scatter(sample_data['Actual'], sample_data['Predicted'],
#             color='blue', s=80, alpha=0.6, edgecolor='white', label='Student Data (Sample)')

# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
#          'r--', lw=2, label='Perfect Prediction')

# plt.grid(True, linestyle='--', alpha=0.7)

# plt.title(f"Visualizing 50 Random Students\nModel Accuracy: {re_score:.2f}")
# plt.xlabel("Actual Score")
# plt.ylabel("Predicted Score")
# plt.legend()
# plt.show()


# def prediction(num1, num2, num3):
#     new_data = [[num1, num2, num3]]
#     pre = pd.DataFrame(new_data, columns=[
#         "hours_study", "sleep_hours", "phone_usage"])
#     result = model.predict(pre)
#     return result
