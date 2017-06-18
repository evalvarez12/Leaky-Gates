import scipy as scipy
import scipy.optimize as optimize
import operations
import numpy as np
import qutip as qtp

class OptimizerMaster:
    def __init__(self, tau, target="ISWAP"):
            """Init function: Define the target gate of the optimizer."""
            if target == "ISWAP":
                self.target = operations.target_iSWAP_master()
                self.evolution_time = np.pi/2.
            if target == "CPHASE":
                self.target = operations.target_CPHASE_master()
                self.evolution_time = np.pi/np.sqrt(2)
            # self.target = qtp.tensor(self.target, self.target.dag())
            self.P = operations.projector2qutrits_master()
            self.tau = tau
            self.Id = qtp.qeye([3, 3])


    def _cost_func(self, x, U_evolution, state):
        """
        Cost function.
        Input x: araray, U_evolution: evolution operator.
        x = (theta1, theta2, theta3) the arguments to which the
        single qubit gates are optimized to minimize fidelity
        """
        theta1,  theta2, theta3 = x
        # single qubit rotations
        ZZ = operations.matrix_optimize(theta1, theta2, theta3)
        # U = qtp.tensor(ZZ, self.Id) * U_evolution * qtp.tensor(self.Id, ZZ.dag())

        #################################
        state_vec = qtp.operator_to_vector(state).full()
        state_evolved = U_evolution.full().dot(state_vec)
        state_evolved = operations.un_vectorize(state_evolved)
        rho = ZZ.full().dot( state_evolved.dot( ZZ.dag().full()))
        rho = self.P.full().dot(rho.dot(self.P.dag().full()))

        comp = self.P * state * self.P.dag()
        comparation = (self.target * comp * self.target.dag()).full()

        F = operations.trace_dist(rho, comparation)
        ######################################

        # project into the qubit space
        # U = qtp.tensor(self.P, self.Id) * U * qtp.tensor(self.Id, self.P.dag())
        # compute fidelity
        # F = operations.fidelity(self.target, U)
        infidelity = F
        return infidelity

    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the Hamiltonian and exponenciate it to obtain evolution operator."""
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        g = operations.operators()
        G_evolution_master = operations.get_master_equation(H, g, self.tau)
        U_evolution_master = (G_evolution_master * evolution_time).expm()
        return U_evolution_master

    def _minimize(self, U_evolution, state):
        """Funcion to call the minimization algorithm."""
        infidelity = lambda x: self._cost_func(x, U_evolution=U_evolution, state=state)
        x0 = [np.pi, np.pi/3, 0]
        res = scipy.optimize.basinhopping(infidelity, x0, T=.5, niter=5)
        print(res)
        return 1 - res.fun


    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling, state):
        """Compute the optimized fidelity for a set of system parameters."""
        U = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        return self._minimize(U, state)


omega1 = 5.5 * 2 * np.pi
omega2 = omega1
delta1 = 0.15 * 2 * np.pi
delta2 = 0.1 * 2 * np.pi

coupling = .05 * delta2

tau_d = .4188 * 0.1
tau_r10 = .31 * 0.1
tau_r21 = .155 * 0.1


initial_state1 = qtp.rand_ket(3)
initial_state2 = qtp.rand_ket(3)
state = qtp.tensor(initial_state1, initial_state2)
state = state*state.dag()

optimizer = OptimizerMaster(tau=[tau_d, tau_d, tau_r10, tau_r10, tau_r21, tau_r21], target="ISWAP")
print(optimizer.get_fidelity(omega1, delta1, omega2, delta2, coupling, state))
