
w1  = 0.0
w2 = 0.0
w3 = 0.0
b = 0.0
lr = 0.01

epochs = 5000

X = [
    [0.2, 0.333333, 0.4],
    [0.6, 0.666667, 0.8],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0]
]

y = [55, 70, 85, 45]
epochs_mess = []

previous_mse = float('inf')
patience = 100
counter = 0
min_delta = 0.00001



for epoch in range(epochs):

    for j in range(len(X)):
        prediction = (
            w1 * X[j][0] + w2 * X[j][1] + w3 * X[j][2] + b
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
            w1 * X[j][0]+
            w2 * X[j][1]+
            w3 * X[j][2]+b
        )

        loss = (prediction - y[j]) ** 2
        total_loss += loss

    average_mse = total_loss / (len(X))
    epochs_mess.append(average_mse)

    if previous_mse - average_mse > min_delta:
        previous_mse = average_mse
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
print("Total epochs trained:", len(epochs_mess))
print("Final MSE:", epochs_mess[-1])
print("Final prediction:", prediction)
    # if (epoch + 1) % 1000 == 0:
    #     print(f"Epoch {epoch + 1}: MSE = {average_mse}")

# print("Initial:", epochs_mess[0])
# print("Initial MSE:", epochs_mess[0])
# print("After 1000 epochs:", epochs_mess[999])
# print("After 2500 epochs:", epochs_mess[2499])
# print("Final MSE:", epochs_mess[-1])

new_student = [3, 6, 75]
study_min, study_max = 1, 6
sleep_min, sleep_max = 5, 8
attendance_min, attendance_max = 70, 95

new_student_normalize = [
    (new_student[0] - study_min) / (study_max - study_min),
    (new_student[1] - sleep_min) / (sleep_max - sleep_min),
    (new_student[2] - attendance_min) / (attendance_max - attendance_min)
]

prediction = (
    w1 * new_student_normalize[0] + 
    w2 * new_student_normalize[1] + 
    w3 * new_student_normalize[2] + 
    b
)

print(prediction)