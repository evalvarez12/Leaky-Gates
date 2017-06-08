import scipy as scipy
import scipy.optimize as optimize
import operations
import numpy as np

class Optimizer:
    def __init__(self, target="iSWAP"):
        if target == "iSWAP":
            self.Target = operations.target_iSWAP()
            self.evolution_time = np.pi/2.
            self.P = operations.proyector_2qutrit()


    def _cost_func(self, x, U_evolution):
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
        unitary_product_phase = operations.matrix_optimize(theta1, theta2, theta3)

        projected_U_evolution = self.P*U_evolution*self.P.dag()
        projected_unitary_phase = self.P*unitary_product_phase*self.P.dag()
        unitary = projected_unitary_phase*projected_U_evolution
        F = operations.Fidelity(self.Target, unitary)
        infidelity = 1 - F
        return infidelity

    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        U_evolution = (-1j * H * evolution_time).expm()
        return U_evolution

    def _minimize(self, U_evolution):
        infidelity = lambda x: self._cost_func(x, U_evolution = U_evolution)
        #setup the constraints
        bnds = ((0,2*np.pi),(0,2*np.pi),(0,2*np.pi))
        x0 = [np.pi,np.pi,0]
        res = scipy.optimize.minimize(infidelity, x0, method= 'Nelder-Mead', tol= 1e-10)
        # return res
        #check
        x1, x2, x3 = res.x
        ZZ = operations.matrix_optimize(x1, x2, x3)
        U_qubit = self.P * ZZ * U_evolution * self.P.dag()
        # print(Fidelity(U_target, U_qubit))
        return operations.Fidelity(self.Target, U_qubit)

    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling):
        U = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        return self._minimize(U)



coupling = 0.3
omega = .5
delta = .5

optimizer = Optimizer("iSWAP")
print(optimizer.get_fidelity(omega, delta, omega, delta, coupling))


#
# evolution_time = np.pi/(2*coupling)
# H = H_coupled_qutrit(.5, .5, .5, .5, coupling)
# # print(H)
# U_evolution = (-1j * H * evolution_time).expm()
# # print(U_evolution)
#
# P = proyector()
# P = qtp.tensor(P, P)
# U_target = target_iSWAP()
#
#
# #optimizer
#
# #anonymous call, vary only x
