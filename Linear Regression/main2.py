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

w = 0.5
b = 0
lr = 0.1

for i in range(100):
    for j in range(len(X)):

        prediction = w * X[]

        print(
            "step:", i,
            "student:", j,
        )
