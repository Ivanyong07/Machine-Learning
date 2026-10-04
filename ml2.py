import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create data
np.random.seed(0)
x = np.sort(5 * np.random.rand(50, 1), axis=0)
y = np.sin(x) + np.random.randn(50, 1) * 0.2

# Function to train model


def model(degree):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    x_test = np.linspace(0, 5, 100).reshape(100, 1)
    y_pred = model.predict(poly.transform(x_test))

    plt.scatter(x, y)
    plt.plot(x_test, y_pred)
    plt.title(f"Degree {degree}")
    plt.show()


# Try different models
model(1)   # High bias (underfit)
model(5)   # Good balance
model(15)  # High variance (overfit)
