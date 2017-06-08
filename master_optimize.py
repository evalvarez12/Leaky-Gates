import scipy as scipy
import scipy.optimize as optimize
import operations
import numpy as np
import qutip as qtp

class OptimizerM:
    def __init__(self, target, evolution_time):
            self.Target = target
            print(target)
            self.evolution_time = evolution_time
            self.P = operations.projector_2qutrit()


    def set_master_tau(self, tau):
        self.tau = tau

    def _cost_func(self, x, U_evolution, initial_state):
        """
        x : array like
        x[0] : parameter for exp(i*sigmaz_A*x[0])
        x[1] : parameter for exp(i*sigmaz_B*x[1])
        x[2] : parameter for exp(i*identity*x[1])
        this function calls fidelity and outputs 1-fidelity as a cost function
        this is fed into an optimizer
        """
        print(x)
        theta1 = x[0]
        theta2 = x[1]
        theta3 = x[2]

        # state_vec = operations.vectorize_operator(state.full())
        state_vec = qtp.operator_to_vector(state).full()
        # print(U_evolution)
        # print(state_vec)
        state_evolved = U_evolution.full().dot(state_vec)


        state_evolved = operations.un_vectorize(state_evolved)
        # state_evolved = qtp.vector_to_operator(state_evolved)

        ZZ = operations.matrix_optimize(theta1, theta2, theta3)

        # iden = qtp.qeye([3,3])
        # ZZbig = qtp.tensor(ZZ, iden) + qtp.tensor(iden, ZZ)

        # U = ZZbig * U_evolution

        # Tbig = qtp.tensor(self.Target, iden) + qtp.tensor(iden, self.Target.dag())
        # Tbig = qtp.tensor(self.Target, self.Target.dag())
        # Do   U rho U+
        rho = ZZ.full().dot( state_evolved.dot( ZZ.dag().full()))
        rho = self.P.full().dot(rho.dot(self.P.dag().full()))
        comparation = (self.Target * state * self.Target.dag())
        comparation = (self.P * comparation * self.P.dag()).full()
        # F = qtp.tracedist(rho, comparation)
        # F = operations.Fidelity(Tbig, U)
        F = operations.trace_dist(rho,comparation)
        print(F)
        infidelity =  F
        return infidelity

    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        g = operations.operators()
        G_evolution_master  = operations.get_master_equation(H, g, self.tau)
        U_evolution_master = ( G_evolution_master * evolution_time).expm()
        return U_evolution_master

    def _minimize(self, state, U_evolution):
        infidelity = lambda x: self._cost_func(x, U_evolution = U_evolution, initial_state=state)
        #setup the constraints
        bnds = ((0,2*np.pi),(0,2*np.pi),(0,2*np.pi))
        x0 = [np.pi/2,np.pi/2,np.pi]
        res = scipy.optimize.minimize(infidelity, x0, method= 'Nelder-Mead', tol= 1e-10)
        # return res
        #check
        x1, x2, x3 = res.x
        print(res)

        return 1 - res.fun


    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling, state_rho):
        U = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        return self._minimize(state_rho, U)
# coupling = 0.3
# omega = .5
# delta = .5
# target = operations.target_iSWAP()
# optimizer = Optimizer(target, np.pi/2.)
# print(optimizer.get_fidelity(omega, delta, omega, delta, coupling))

print("USING MASTER")

coupling = 0.3
omega = .5
delta = .5
target = operations.target_iSWAP()
initial_state1 = qtp.rand_ket(3)
initial_state2 = qtp.rand_ket(3)
state = qtp.tensor(initial_state1, initial_state2)
state = state*state.dag()

state = operations.special_state()

optimizer = OptimizerM(target, np.pi/2)
optimizer.set_master_tau([0, 0, 0, 0, 0, 0])
print(optimizer.get_fidelity(omega, delta, omega, delta, coupling, state))

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
