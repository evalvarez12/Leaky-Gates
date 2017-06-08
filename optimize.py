import scipy as scipy
import scipy.optimize as optimize

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
    
    projected_U_evolution = P*U_evolution*P.dag()
    projected_unitary_phase = P*unitary_product_phase*P.dag()
    unitary = projected_unitary_phase*projected_U_evolution
    F = Fidelity(U_target, unitary)
    infidelity = 1- F
    return infidelity
    

U_target = target_iSWAP()
#optimizer

#anonymous call, vary only x
infidelity = lambda x: cost_func(x, P = P, U_evolution = U_evolution, U_target = U_target)
#setup the constraints
bnds = ((0,2*np.pi),(0,2*np.pi),(0,2*np.pi))
x0 = [np.pi,np.pi,0]
res = scipy.optimize.minimize(infidelity, x0, method= 'Nelder-Mead',bounds=bnds, tol= 1e-10)
