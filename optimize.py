import scipy as scipy
import scipy.optimize as optimize
import operations
import numpy as np
import qutip as qtp

class Optimizer:
    def __init__(self, target, evolution_time, master=False):
            self.Target = target
            self.evolution_time = evolution_time
            self.master = master
            if master:
                d = target.dims[0]
                self.P = qtp.qeye(d)
            else:
                self.P = operations.projector_2qutrit()



    def set_master_tau(self, tau):
        self.tau = tau

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

        if self.master:
            unitary_product_phase = qtp.tensor(unitary_product_phase, unitary_product_phase)

        projected_U_evolution = self.P*U_evolution*self.P.dag()
        projected_unitary_phase = self.P*unitary_product_phase*self.P.dag()
        unitary = projected_unitary_phase*projected_U_evolution
        F = operations.Fidelity(self.Target, unitary)
        infidelity = 1 - F
        return infidelity

    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        if self.master:
            g = operations.operators()
            G_evolution_master  = operations.get_master_equation(H, g, self.tau)
            U_evolution_master = ( G_evolution_master * evolution_time).expm()
            return U_evolution_master
        else:
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
        if self.master:
            ZZ = qtp.tensor(ZZ, ZZ.dag())
        U_qubit = self.P * ZZ * U_evolution * self.P.dag()
        # print(Fidelity(U_target, U_qubit))
        return operations.Fidelity(self.Target, U_qubit)

    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling):
        U = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        return self._minimize(U)



coupling = 0.3
omega = .5
delta = .5
target = operations.target_iSWAP()
optimizer = Optimizer(target, np.pi/2.)
print(optimizer.get_fidelity(omega, delta, omega, delta, coupling))

print("USING MASTER")

coupling = 0.3
omega = .5
delta = .5
target = operations.target_iSWAP_master()
optimizer = Optimizer(target, np.pi/2., master=True)
optimizer.set_master_tau([0, 0, 0, 0, 0, 0])
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
