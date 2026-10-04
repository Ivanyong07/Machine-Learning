import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
colors = [10, 20, 30, 40, 50]  # numeric values for colormap

plt.scatter(
    x, y,
    s=10,                # size of dots
    c=colors,             # color mapped to values
    cmap='plasma',        # colormap
    marker='p',           # triangle markers
    alpha=0.7,            # transparency
    edgecolors='black',   # outline color
    linewidths=1.5,       # outline thickness
    label='Data points'   # legend label
)

plt.colorbar(label="Value scale")  # adds color scale bar
plt.legend()
plt.title("Detailed Scatter Plot")
plt.show()
