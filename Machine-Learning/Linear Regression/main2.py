# Study hours ───────┐
# │
# Sleep hours ───────┼──→ Linear Regression ──→ Exam Score
# │
# Attendance ────────┘

X = [
    [2, 6, 80],
    [4, 7, 90],
    [6, 8, 95],
    [1, 5, 70]
]

y = [
    55,
    70,
    85,
    45,
]

def normalize(values):
    min_value = min(values)
    max_value = max(values)

    normalized = []

    for value in values:
        result = (value - min_value) / (max_value - min_value)
        normalized.append(result)

    return normalized

study_hours = normalize([row[0] for row in X])
sleep_hours = normalize([row[1] for row in X])
attendance = normalize([row[2] for row in X])

X = [
    [study_hours[i], sleep_hours[i], attendance[i]]
    for i in range(len(X))
]

w1 = 0.0
w2 = 0.0
w3 = 0.0
b = 0.0

# Training Rate
lr = 0.01
epochs = 5000

epochs_mess = []

for epoch in range(epochs):

    for j in range(len(X)):

        prediction = (
            w1 * X[j][0]
            + w2 * X[j][1]
            + w3 * X[j][2]
            + b
        )

        error = prediction - y[j]


        dw1 = 2 * error * X[j][0]
        dw2 = 2 * error * X[j][1]
        dw3 = 2 * error * X[j][2]
        db = 2 * error


        w1 -= lr * dw1
        w2 -= lr * dw2
        w3 -= lr * dw3
        b -= lr * db

    total_loss = 0
    for j in range(len(X)):
        prediction = (
            w1 * X[j][0]
            + w2 * X[j][1]
            + w3 * X[j][2]
            + b
        )

        loss = (prediction - y[j]) ** 2
        total_loss += loss
    
    average_mse = total_loss / len(X)
    epochs_mess.append(average_mse)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: MSE = {average_mse}")

print("\n--- Model Evaluation ---")

for j in range(len(X)):
    prediction = (
        w1 * X[j][0]
        + w2 * X[j][1]
        + w3 * X[j][2]
        + b
    )

    error = prediction - y[j]

    print(
        f"Student {j + 1}: "
        f"Actual = {y[j]}, "
        f"Predicted = {prediction:.2f}, "
        f"Error = {error:.2f}"
    )  