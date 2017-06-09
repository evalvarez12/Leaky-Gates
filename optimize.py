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
        projected_unitary_phase = self.P * unitary_product_phase * self.P.dag()
        unitary = projected_unitary_phase * U_evolution
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

        return 1-res.fun

    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling):
        U_evolution = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        U_projected = self.P*U_evolution*self.P.dag()
        return self._minimize(U_projected)


    # def _get_trace_distance(self, rho_sim, rho_target):
    #
    #     #rho_sim is reshaped density matrix
    #     #todo: Put the density matrix argument
    #     #calculates the trace distance between target..
    #     #..density matrix and materequation density matrix
    #     m, n = rho_sim.shape
    #     rho_sim = rho_sim.reshape(np.sqrt(m*n),np.sqrt(m*n))
    #     l,j = rho_target.shape
    #     rho_target = rho_target.reshape(np.sqrt(m*n),np.sqrt(m*n))
    #
    #     trace_distance = qtp.tracedist(rho_sim, rho_target, sparse=False, tol=0)
    #     return trace_distance
    # def _cost_trace_distance(self, x, rho_sim, rho_target):
    #     #calculates a cost function based on the trace distance
    #     #of calculated density matrix from target density matrix
    #     theta1 = x[0]
    #     theta2 = x[1]
    #     theta3 = x[2]
    #     m, n = rho_sim.shape
    #     rho_sim = rho_sim.reshape(np.sqrt(m*n),np.sqrt(m*n))
    #     unitary_product_phase = operations.matrix_optimize(theta1, theta2, theta3)
    #     rho_parametric = unitary_product_phase*rho_sim*unitary_product_phase.dag()
    #     cost_trace_distance = 1 - _get_trace_distance(self, rho_parametric, rho_target)
    #     return cost_trace_distance
    # def _minimize_trace_distance(self, rho_simulation):
    #     trace_fidelity = lambda x: self._cost_tace_distance(self, x, rho_sim = rho_sim, rho_target = rho_target
    #     #setup the constraints
    #     bnds = ((0,2*np.pi),(0,2*np.pi),(0,2*np.pi))
    #     x0 = [np.pi,np.pi,0]
    #     res = scipy.optimize.minimize(trace_fidelity, x0, method= 'Nelder-Mead', tol= 1e-10)
    #
    #     ZZ = operations.matrix_optimize(res.x[0], res.x[1], res.x[2])
    #
    #     # print(Fidelity(U_target, U_qubit))
    #     return operations.Fidelity(self.Target, U_qubit)

coupling = 2313
omega = 50
delta = 15
target = operations.target_iSWAP()
optimizer = Optimizer(target, 231.)
print(optimizer.get_fidelity(omega, delta, omega, delta, coupling))

#
# coupling = 0.3
# omega = .5
# delta = .5
# target = operations.target_iSWAP_master()
# optimizer = Optimizer(target, np.pi/2., master=True)
# optimizer.set_master_tau([0, 0, 0, 0, 0, 0])
# print(optimizer.get_fidelity(omega, delta, omega, delta, coupling))
#
# #
# # evolution_time = np.pi/(2*coupling)
# # H = H_coupled_qutrit(.5, .5, .5, .5, coupling)
# # # print(H)
# # U_evolution = (-1j * H * evolution_time).expm()
# # # print(U_evolution)
# #
# P = proyector()
# P = qtp.tensor(P, P)
# U_target = target_iSWAP()
#
#
# #optimizer
#
# #anonymous call, vary only x
