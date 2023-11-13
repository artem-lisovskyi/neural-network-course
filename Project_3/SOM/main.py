import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import scale

from SOM.model import SOM

digits = datasets.load_digits(n_class=10)
data = digits.data
data = scale(data)
num = digits.target

som = SOM(
    size=(30, 30),
    feature=64,
    learning_rate=0.9,
    max_iterations=20,
    neighbor_function="gaussian",
)

som.fit(data, num)

plt.plot(range(som.max_iterations), som.errors)
plt.title(som.title)
plt.show()

plt.figure(figsize=(8, 8))
plt.title(som.title)
wmap = {}
im = 0
for x, t in zip(data, num):
    w = som.get_winner(x)
    wmap[w] = im
    plt.text(w[0], w[1], str(t), color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold', 'size': 11})
    im = im + 1
plt.axis([0, som.size[0], 0, som.size[1]])
plt.show()