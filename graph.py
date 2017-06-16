import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import master_optimize as optimize

# N = 20
#
# coupling = 0.3
# omega = .5
# target = operations.target_iSWAP()
#
# data = np.zeros((N, N))
# delta = np.linspace(0, 10.*coupling, N)
#
#
# optimizer = optimize.Optimizer(target, np.pi/2.)
#
# for i in range(N):
#     for j in range(N):
#         data[i, j] = optimizer.get_fidelity(omega, delta[i], omega, delta[j], coupling)
#     print(i)
#
# plt.clf()
# plt.imshow(data, origin='lower',  cmap='hot')
# plt.show()


omega1 = 5.5 * 2 * np.pi
omega2 = omega1
delta1 = 0.15 * 2 * np.pi
delta2 = 0.1 * 2 * np.pi

N = 10

g = np.linspace(0.001, .05, N)

tau_d = .4188 * 2 * np.pi
tau_r10 = .31 * 2 * np.pi
tau_r21 = .155 * 2 * np.pi

optimizer = optimize.OptimizerMaster(tau=[tau_d,tau_d, tau_r10, tau_r10, tau_r21, tau_r21], target="ISWAP")
data = []
for i in g:
    f = optimizer.get_fidelity(omega1, delta1, omega2, delta2, i)
    data += [f]

np.save("data/optimizer_3a_150_BH", data)

plt.plot(g, data, 'o-')
plt.show()
