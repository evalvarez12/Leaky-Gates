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

    def __init__(self, target="ISWAP", tau=[0, 0, 0, 0, 0, 0], master=False):
        """Init function: Define the target gate of the optimizer."""
        self.master = master
        if master:
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
        else:
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

    def _cost_func_master(self, x, U_evolution, state):
        """
        Cost function.
        Input x: araray, U_evolution: evolution operator.
        x = (theta1, theta2, theta3) the arguments to which the
        single qubit gates are optimized to minimize fidelity
        """
        theta1,  theta2, theta3 = x
        # single qubit rotations
        ZZ = operations.matrix_optimize(theta1, theta2, theta3)
        state_vec = qtp.operator_to_vector(state).full()
        state_evolved = U_evolution.full().dot(state_vec)
        state_evolved = operations.un_vectorize(state_evolved)
        rho = ZZ.full().dot(state_evolved.dot(ZZ.dag().full()))
        rho = self.P.full().dot(rho.dot(self.P.dag().full()))

        comp = (self.target * state * self.target.dag()).full()
        comp = self.P * comp * self.P.dag()

        F = operations.trace_dist(rho, comp)

        infidelity = F
        return infidelity


    def _get_evolution(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the Hamiltonian and exponenciate it to obtain evolution operator."""
        # compute the Hamiltonian of the entire system
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        self.H = H
        U_evolution = (-1j * H * evolution_time).expm()
        return U_evolution

    def _get_evolution_master(self, freq1, anh1, freq2, anh2, coupling):
        """Compute the Hamiltonian and exponenciate it to obtain evolution operator."""
        evolution_time = self.evolution_time/coupling
        H = operations.H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling)
        g = operations.operators()
        G_evolution_master = operations.get_master_equation(H, g, self.tau)
        U_evolution_master = (G_evolution_master * evolution_time).expm()
        return U_evolution_master

    def _minimize(self, U_evolution):
        """Funcion to call the minimization algorithm."""
        # anonymous function to accomodate all the parameters
        infidelity = lambda x: self._cost_func(x, U_evolution = U_evolution)
        # initial guess for the optimizer
        x0 = [np.pi, np.pi, 0]
        # optimizer solution
        # bnds = [(0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        minimizer_kwargs = {"method": "Nelder-Mead"}
        res = scipy.optimize.basinhopping(infidelity, x0, minimizer_kwargs=minimizer_kwargs, T=1., niter=15)
        print(res)
        return 1-res.fun

    def _minimize_master(self, U_evolution, state):
        """Funcion to call the minimization algorithm."""
        infidelity = lambda x: self._cost_func_master(x, U_evolution=U_evolution, state=state)
        x0 = [np.pi, np.pi, 0]
        res = scipy.optimize.basinhopping(infidelity, x0, T=1., niter=15)
        print(res)
        return 1 - res.fun

    def get_fidelity(self, freq1, anh1, freq2, anh2, coupling, state=0):
        """Compute the optimized fidelity for a set of system parameters."""
        if self.master:
            U_master = self._get_evolution_master(freq1, anh1, freq2, anh2, coupling)
            f = self._minimize_master(U_master, state)
        else:
            U_evolution = self._get_evolution(freq1, anh1, freq2, anh2, coupling)
            f = self._minimize(U_evolution)
        return f

# TESTING optimizer

def test_optimizer():
    """Dummy function to test the optimizer."""
    print("Testing optimizer")
    omega1 = 7.6 * 2 * np.pi
    coupling = 0.2 * 2 * np.pi
    delta1 = 2.5 * coupling
    delta2 = 2.5 * coupling
    omega2 = omega1 + delta2

    tau_d = .4188
    tau_r10 = .31
    tau_r21 = .155
    tau = [tau_d, tau_d, tau_r10, tau_r10, tau_r21, tau_r21]

    initial_state1 = qtp.rand_ket(3)
    initial_state2 = qtp.rand_ket(3)
    state = qtp.tensor(initial_state1, initial_state2)
    state = state*state.dag()

    optimizer = Optimizer(target="CPHASE", tau=tau, master=True)
    print(optimizer.get_fidelity(omega1, delta1, omega2, delta2, coupling, state))
