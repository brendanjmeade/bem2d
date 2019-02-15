import numpy as np
import matplotlib.pyplot as plt

npzfile = np.load("model_run_huge.npz")
history = npzfile['arr_0']
times = npzfile['arr_1']

plt.close("all")
# for i in range(0, times.size):
plt.figure
plt.plot(history[3000, 0::3], linewidth=0.5)


plt.figure()
plt.contourf(history[:, 0::3])
plt.show(block=False)
