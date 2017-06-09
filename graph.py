import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import optimize
import operations

# Generate some test data
N = 20

coupling = 0.3
omega = .5
target = operations.target_iSWAP()

data = np.zeros((N, N))
delta = np.linspace(0, 10.*coupling, N)


optimizer = optimize.Optimizer(target, np.pi/2.)

for i in range(N):
    for j in range(N):
        data[i, j] = optimizer.get_fidelity(omega, delta[i], omega, delta[j], coupling)
    print(i)

plt.clf()
plt.imshow(data, origin='lower',  cmap='hot')
plt.show()
