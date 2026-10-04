import numpy as np

x = 1  # input
y = 3  # correct ans

w = 0.5  # weight
b = 0  # bias

lr = 0.1  # learning rate

# linear regression formula
# ^y​=wx+b

for i in range(10):

    # 1. Forward pass
    prediction = w * x + b  # 0.5 but correct ans is 3 so it wrong

    # 2. Calculate error
    loss = (prediction - y) ** 2  # 6.25

    # 3. Backpropagation: calculate gradients
    dw = 2 * (prediction - y) * x  # change w (dw = 2 × (0.5 - 3) × 1= -5)
    db = 2 * (prediction - y)

    # 4. Gradient descent: update weights
    w = w - lr * dw
    b = b - lr * db

    print(
        "step", i,
        "prediction:", prediction,
        "loss:", loss,
        "w:", w,
        "b:", b
    )
