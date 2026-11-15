import pandas as pd
from sklearn.tree import DecisionTreeRegressor


path = './melb_data.csv'
melb_data = pd.read_csv(path)
melbourne_data = melb_data.dropna(axis=0)
y = melb_data.Price
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melb_features]

print(melb_data.describe())
print(melb_data.columns)
print(melbourne_data.describe())
print(y.describe())
print(X.describe())
print(X.head())

melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
