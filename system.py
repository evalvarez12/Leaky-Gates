from eduardo import *
import matplotlib.pyplot as plt
import numpy as np
import qutip as qtp

coupling = 0.3
omega = .5
delta = .5

evolution_time = np.pi/(2*coupling)
H = H_coupled_qutrit(.5, 0, .5, 0, coupling)
# print(H)
U_evolution = (-1j * H * evolution_time).expm()
print(U_evolution)

P = proyector()
P = qtp.tensor(P, P)

ZZ = sigmaz_qutrit(np.pi/5., np.pi/5.)
print(ZZ)
U_qubit = P * ZZ * U_evolution * P.dag()
print(U_qubit)

# ZZ = qtp.tensor(qtp.gates.rz(np.pi), qtp.gates.rz(np.pi))
# ZZU = ZZ*U_qubit
# print(ZZU)

print("Using master equation")
w = -1
gz = qtp.Qobj(np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]]))
gx1 = qtp.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
gx2 = qtp.Qobj(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))


g = [set_single_2qutrit_gate(gz, 0), set_single_2qutrit_gate(gz, 1),
     set_single_2qutrit_gate(gx1, 0), set_single_2qutrit_gate(gx1, 1),
     set_single_2qutrit_gate(gx2, 0), set_single_2qutrit_gate(gx2, 1)]

# tau = [.05, .05, .05, .05, .05, .05]
tau = [0, 0, 0, 0, 0, 0]

G_evolution_master  = get_master_equation(H, g, tau)
# print(G_evolution_master)
U_evolution_master = ( G_evolution_master * evolution_time).expm()
# print(U_evolution_master)

UU = qtp.tensor(U_evolution, U_evolution.dag())

comp = UU - U_evolution_master

print(comp.norm())
print(Fidelity(UU, U_evolution_master))
