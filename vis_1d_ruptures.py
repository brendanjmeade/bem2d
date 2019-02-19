import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

npzfile = np.load("model_run_huge.npz")
history_even = npzfile['arr_0']
times_even = npzfile['arr_1']

# for i in range(0, times.size):
# plt.figure
# plt.plot(history[3000, 0::3], linewidth=0.5)

plt.figure()
plt.contourf(history_even[:, 0::3])
plt.title("even")
plt.colorbar()
plt.show(block=False)

npzfile = np.load("model_run_huge_uneven.npz")
history_uneven = npzfile['arr_0']
times_uneven = npzfile['arr_1']

plt.figure()
plt.contourf(history_uneven[:, 0::3])
plt.title("uneven")
plt.colorbar()
plt.show(block=False)

plt.figure()
plt.contourf(np.abs(history_even[:, 0::3] - history_uneven[:, 0::3]) / history_even[:, 0::3] * 100)
plt.title("even - uneven")
plt.colorbar()
plt.show(block=False)