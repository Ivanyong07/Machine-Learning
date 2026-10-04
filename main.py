import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = 'train.csv'
data = pd.read_csv(file_path, on_bad_lines='skip')
print(data.describe())

file_path = 'melb_data.csv'
data = pd.read_csv(file_path)
print(data.columns)

# Sample data
data = {
    'numeric': [1, 2, 3, 4, 5],
    'object': ['a', 'b', 'b', 'c', 'c']
}
# Create DataFrame
df = pd.DataFrame(data)
# Generate descriptive statistics
print(df.describe())


# STEP 1: LOAD (The "Database" Phase)
data = pd.read_csv('train.csv')

# STEP 1.5: ASSIGN (The "X and y" Phase)
# y is our "Answer" (Sale Price)
y = data.SalePrice

# X is our "Clues" (Features)
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF',
                 '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[feature_names]

# STEP 2: DEFINE THE MODEL (The "Robot" Phase)
# random_state=1 ensures we get the same results every time we run it
house_model = DecisionTreeRegressor(random_state=1)

# STEP 3: FIT (The "Training" Phase)
# The robot studies the relationship between X and y
house_model.fit(X, y)

print("Robot training complete!")

# STEP 4: PREDICT (The "Final Test" Phase)
# Let's ask the robot to guess the price of the first 5 houses in the CSV
predictions = house_model.predict(X.head())

print("\n--- Results ---")
print("Robot Guesses: ", predictions)
print("Actual Prices: ", y.head().tolist())


1. Prepare Data
data = {
    "hours_study": [1, 2, 3, 4, 5, 6, 7, 8],
    "sleep_hours": [8, 7, 6, 6, 5, 5, 4, 4],
    "score": [50, 55, 65, 70, 75, 85, 90, 95]
}

df = pd.DataFrame(data)

# 2. Split Features (X) and Target (y)
X = df[["hours_study", "sleep_hours"]]
y = df["score"]

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluation
r2_score = model.score(X_test, y_test)
print(f"Model R-squared: {r2_score:.2f}")

# 6. Making a Prediction
# Predicting score for 6 hours study and 6 hours sleep
new_input = pd.DataFrame([[1, 5]], columns=["hours_study", "sleep_hours"])
prediction = model.predict(new_input)
print(f"Predicted score: {prediction[0]:.2f}")
