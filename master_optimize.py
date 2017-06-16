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
            self.target = qtp.tensor(self.target, self.target.dag())
            self.P = operations.projector2qutrits_master()
            self.tau = tau
            self.Id = qtp.qeye([3, 3])


    def _cost_func(self, x, U_evolution):
        """
        Cost function.
        Input x: araray, U_evolution: evolution operator.
        x = (theta1, theta2, theta3) the arguments to which the
        single qubit gates are optimized to minimize fidelity
        """
        theta1,  theta2, theta3 = x
        # single qubit rotations
        ZZ = operations.matrix_optimize(theta1, theta2, theta3)
        U = qtp.tensor(ZZ, self.Id) * U_evolution * qtp.tensor(self.Id, ZZ.dag())

        # project into the qubit space
        U = qtp.tensor(self.P, self.Id) * U * qtp.tensor(self.Id, self.P.dag())
        # compute fidelity
        F = operations.fidelity(self.target, U)
        infidelity = 1 - F
        return infidelity

    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the Hamiltonian and exponenciate it to obtain evolution operator."""
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        g = operations.operators()
        G_evolution_master = operations.get_master_equation(H, g, self.tau)
        U_evolution_master = (G_evolution_master * evolution_time).expm()
        return U_evolution_master

    def _minimize(self, U_evolution):
        """Funcion to call the minimization algorithm."""
        infidelity = lambda x: self._cost_func(x, U_evolution=U_evolution)
        x0 = [np.pi, np.pi/3, 0]
        res = scipy.optimize.basinhopping(infidelity, x0, T=.2, niter=5)
        print(res)
        return 1 - res.fun


    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the optimized fidelity for a set of system parameters."""
        U = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        return self._minimize(U)


omega1 = 5.5 * 2 * np.pi
omega2 = omega1
delta1 = 0.15 * 2 * np.pi
delta2 = 0.1 * 2 * np.pi

coupling = .1 * delta2

tau_d = .4188
tau_r10 = .31
tau_r21 = .155

optimizer = OptimizerMaster(tau=[tau_d,tau_d, tau_r10, tau_r10, tau_r21, tau_r21], target="ISWAP")
print(optimizer.get_fidelity(omega1, delta1, omega2, delta2, coupling))
