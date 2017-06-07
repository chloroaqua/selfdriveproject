import numpy as np
from matplotlib import pyplot as plt, cm


history = np.load('res50_trans_loss.npy')
x = np.arange(1, 33793)
print(history.shape)
print(x.shape)

plt.plot(x, history)
plt.title("ResNet50 Transfer Model")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.show()