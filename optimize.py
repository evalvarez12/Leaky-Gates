import scipy as scipy
import scipy.optimize as optimize
from eduardo import *


def cost_func(x, P, U_evolution, U_target):
    """
    x : array like
    x[0] : parameter for exp(i*sigmaz_A*x[0])
    x[1] : parameter for exp(i*sigmaz_B*x[1])
    x[2] : parameter for exp(i*identity*x[1])
    this function calls fidelity and outputs 1-fidelity as a cost function
    this is fed into an optimizer
    """
    theta1 = x[0]
    theta2 = x[1]
    theta3 = x[2]
    unitary_product_phase = matrix_optimize(theta1, theta2, theta3)
    unitary = unitary_product_phase*U_evolution
    projected_unitary = P*unitary*P.dag()
    F = Fidelity(U_target, projected_unitary)
    infidelity = 1- F
    return infidelity


#optimizer
coupling = 0.3
omega = .5
delta = .5

evolution_time = np.pi/(2*coupling)
H = H_coupled_qutrit(.5, .5, .5, .5, coupling)
# print(H)
U_evolution = (-1j * H * evolution_time).expm()
P = proyector()
P = qtp.tensor(P, P)
U_target = target_iSWAP()
# print(U_target)
#anonymous call, vary only x
infidelity = lambda x: cost_func(x, P = P, U_evolution = U_evolution, U_target = U_target)

#setup the constraints
bnds = ((0,2*np.pi),(0,2*np.pi),(0,2*np.pi))
x0 = [1,0,.5]
# res = scipy.optimize.minimize(infidelity, x0, method= 'BFGS',bounds=bnds, tol= 1e-5)
res = scipy.optimize.minimize(infidelity, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})



#check
x1, x2, x3 = res.x
ZZ = matrix_optimize(x1, x2, x3)
U_qubit = P * ZZ * U_evolution * P.dag()
print(Fidelity(U_target, U_qubit))
