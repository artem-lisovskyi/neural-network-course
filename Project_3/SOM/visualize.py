import numpy as np
from matplotlib import pyplot as plt

from SOM.model import SOM

from sklearn import datasets
from sklearn.preprocessing import scale

digits = datasets.load_digits(n_class=10)
data = digits.data
data = scale(data)
num = digits.target

neighbor_functions = ["bubble", "gaussian", "triangle"]

mutable_functions = {
    "none": lambda origin, iteration: origin,
    "linear": lambda origin, iteration: origin / (1 + iteration / (20 / 2)),
    "exp,max": lambda origin, iteration: origin * np.exp(- iteration / 20),
    "exp,max*2": lambda origin, iteration: origin * np.exp(- iteration / 40),
}

fig, axes = plt.subplots(3, 4, figsize=(12, 12))

for i, neighbor in enumerate(neighbor_functions):
    for j, (name, func) in enumerate(mutable_functions.items()):
        target_ax = axes[i][j]

        som = SOM(
            size=(30, 30),
            feature=64,
            learning_rate=0.5,
            max_iterations=20,
            neighbor_function=neighbor,
            mutable_update=func,
            first_show=1
        )

        som.fit(data, num)

        target_ax.plot(range(som.max_iterations), som.errors)

        if j == 0:
            target_ax.set_ylabel(neighbor)

        if i == 0:
            target_ax.title.set_text(name)

fig.show()

pass
