"""
Routines for adquiring data and plotting the results.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp
import master_optimize as moptimize
import optimize
import operations


# ------------------ 2D heatmap --------------------------
###########################################################
# N = 15
# #
# omega1 = 7.16 * 2 * np.pi
# omega2 = omega1
# coupling = 0.2 * 2 * np.pi
#
# data = np.zeros((N, N))
# delta = np.linspace(0.0001, 5.*coupling, N)
#
# tau_d = .4188
# tau_r10 = .31
# tau_r21 = .155
# tau = [tau_d, tau_d, tau_r10, tau_r10, tau_r21, tau_r21]
#
#
# initial_state1 = qtp.rand_ket(3)
# initial_state2 = qtp.rand_ket(3)
# state = qtp.tensor(initial_state1, initial_state2)
# state = state*state.dag()
#
# optimizer = optimize.Optimizer(target="CPHASE")
#
# for i in range(N):
#     for j in range(N):
#         data[i, j] = optimizer.get_fidelity(omega1, delta[i], omega1 + delta[j], delta[j], coupling)
#     print(i)
#
# np.save("data/cmap_fig4_cphase2.npy", data)
#
# plt.clf()
# plt.imshow(data, origin='lower',  cmap='hot')
# plt.colorbar()
# plt.show()

##########################################################
##########################################################
#
# omega1 = 5.5 * 2 * np.pi
# omega2 = omega1
# delta1 = 0.1 * 2 * np.pi
# delta2 = 0.1 * 2 * np.pi
# coupling = 2 * delta2
#
# evolution_time = np.pi/(2*coupling)
# H = operations.H_coupled_qutrit(omega1, delta1, omega2, delta2, coupling)
# U_evolution = (-1j * H * evolution_time).expm()
#
# P = operations.projector2qutrits()
# target = operations.target_iSWAP()
#
#
#
# theta1 = 2.49126735
# theta2 = np.linspace(0, np.pi * 2, 150)
# theta3 = -0.24952547
#
# data = []
# for i in theta2:
#     ZZ = operations.matrix_optimize(theta1, i, theta3)
#     U = ZZ * U_evolution
#     U = P * U * P.dag()
#     f = 1 - operations.fidelity(target, U)
#     data += [f]
#
# plt.plot(theta2, data, '-', label=r"$\theta_1, \theta_3$")
#
# theta1 = 1.5
# theta2 = np.linspace(0, np.pi * 2, 150)
# theta3 = +.1
#
# data = []
# for i in theta2:
#     ZZ = operations.matrix_optimize(theta1, i, theta3)
#     U = ZZ * U_evolution
#     U = P * U * P.dag()
#     f = 1 - operations.fidelity(target, U)
#     data += [f]
#
# plt.plot(theta2, data, '-', label=r"$\theta_1', \theta_3'$")
#
# plt.xlabel(r"$\theta_2$")
# plt.ylabel(r"$1 -F$")
# plt.axis([0,2*np.pi,0,3])
# plt.legend()
# plt.show()

############################################################
data  = np.load("data/cmap_fig4_iswap_MASTER.npy")
extent = 10
plt.clf()

plt.title("CPHASE")
plt.xlabel(r"$\Delta_B / g$")
plt.ylabel(r"$\Delta_A / g$")

plt.imshow(data, origin='lower',  cmap='hot', extent=[0,extent, 0, extent])
plt.colorbar()
plt.show()
