import scipy as scipy
import scipy.optimize as optimize
import operations
import numpy as np
import qutip as qtp

class Optimizer:
    """
    Optimizer class.
    Uses the qubit-qubit scheme to simulate the evolution of the system
    at a given time. Single qubit gates Z1(theta), Z2(theta) and I(theta)
    are the used in a optimizer function to maximize the fidelity with
    a specific target gate.
    """

    def __init__(self, target="ISWAP"):
        """Init function: Define the target gate of the optimizer."""
        if target == "ISWAP":
            self.target = operations.target_iSWAP()
            self.evolution_time = np.pi/2.
        if target == "CPHASE":
            self.target = operations.target_CPHASE()
            self.evolution_time = np.pi/np.sqrt(2)
        self.P = operations.projector2qutrits()

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
        U = ZZ * U_evolution

        # collapse the operatarions
        U = self.P * U * self.P.dag()

        # calculate fidelity
        f = operations.fidelity(self.target, U)
        # 1 - fidelity: used for the minimization alg.
        infidelity = 1 - f
        return infidelity

    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the Hamiltonian and exponenciate it to obtain evolution operator."""
        # compute the Hamiltonian of the entire system
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        self.H = H
        U_evolution = (-1j * H * evolution_time).expm()
        return U_evolution

    def _minimize(self, U_evolution):
        """Funcion to call the minimization algorithm."""
        # anonymous function to accomodate all the parameters
        infidelity = lambda x: self._cost_func(x, U_evolution = U_evolution)
        # initial guess for the optimizer
        x0 = [np.pi, np.pi, 0]
        # optimizer solution
        res = scipy.optimize.minimize(infidelity, x0, method='Nelder-Mead',tol=1e-10)
        print(res)
        return 1-res.fun

    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the optimized fidelity for a set of system parameters."""
        U_evolution = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
        return self._minimize(U_evolution)


# TESTING optimizer
coupling = .2 * np.pi
omega = .5 * 2 * np.pi
delta = 0.3 * np.pi
target = operations.target_iSWAP()
optimizer = Optimizer(target="ISWAP")
print(optimizer.get_fidelity(omega, 0, omega, delta, coupling))
