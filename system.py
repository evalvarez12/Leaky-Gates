import matplotlib.pyplot as plt
import numpy as np
import qutip as qtp
import operations

coupling = 0.3
omega = .5
delta = .5

evolution_time = np.pi/(2*coupling)
H = operations.H_coupled_qutrit(.5, .5, .5, .5, coupling)
# print(H)
U_evolution = (-1j * H * evolution_time).expm()
P = operations.projector_2qutrit()

M = operations.matrix_optimize(np.pi/2., np.pi, 0 )
idd = qtp.qeye([3, 3])

print("Using master equation")
g = operations.operators()
# tau = [.05, .05, .05, .05, .05, .05]
tau = [0, 0, 0, 0, 0, 0]

G_evolution_master  = operations.get_master_equation(H, g, tau)
# print(G_evolution_master)
U_evolution_master = ( G_evolution_master * evolution_time).expm()
# print(U_evolution_master)

UU = qtp.tensor(M*U_evolution, U_evolution.dag()*M.dag())

U_evolution_master = qtp.tensor(M, idd) * U_evolution_master * qtp.tensor(idd, M.dag())

comp = UU - U_evolution_master

print(operations.Fidelity(UU, U_evolution_master))
