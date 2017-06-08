"""
Leaky gates implementation with master equation.

created-on: 02/06/17
"""
import numpy as np
import qutip as qtp


def basis_qutrit():
    ket0 = qtp.basis(3, 0)
    ket1 = qtp.basis(3, 1)
    ket2 = qtp.basis(3, 2)
    bra0 = qtp.dag(ket0)
    bra1 = qtp.dag(ket1)
    bra2 = qtp.dag(ket2)
    return ket0, ket1, ket2, bra0, bra1, bra2


def basis_qubit():
    ket0 = qtp.basis(2, 0)
    ket1 = qtp.basis(2, 1)
    bra0 = qtp.dag(ket0)
    bra1 = qtp.dag(ket1)
    return ket0, ket1, bra0, bra1

def proyector():
    bitKet0, bitKet1, bitBra0, bitBra1 = basis_qubit()
    triKet0, triKet1, triKet2, triBra0, triBra1, triBra2 = basis_qutrit()
    P = bitKet0*triBra0 + bitKet1*triBra1
    return P

def target_iSWAP():
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) - 1j*qtp.tensor(ket0, ket1)*qtp.tensor(bra1, bra0) \
        - 1j*qtp.tensor(ket1, ket0)*qtp.tensor(bra0, bra1) + qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    return A

def target_CNOT():
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) + qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        + qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    return A

def matrix_optimize(theta1, theta2, theta3):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) + qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        - qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    B =  qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) - qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        + qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)

    A = (1j * theta1 * A).expm()
    B = (1j * theta2 * B).expm()
    C = qtp.qeye([3,3])*np.exp(1j*theta3)
    return A*B*C

def H_single(freq1, anh1):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    H = freq1*ket1*bra1 + (2*freq1 - anh1)*ket2*bra2
    return H

def H_single2(freq1, anh1, freq2, anh2):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    qubit_term1 = freq1*ket1*bra1 + (2*freq1 - anh1)*ket2*bra2
    qubit_term2 = freq2*ket1*bra1 + (2*freq2 - anh2)*ket2*bra2
    qubit_term = qtp.tensor(qubit_term1, qubit_term2)
    return qubit_term

def H_coupling(coupling):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    term1 = (qtp.tensor(ket0, ket1)*qtp.tensor(bra1, bra0) + np.sqrt(2)*qtp.tensor(ket0, ket2)*qtp.tensor(bra1, bra1)
             + np.sqrt(2)*qtp.tensor(ket2, ket0)*qtp.tensor(bra1, bra1) + 2* qtp.tensor(ket1, ket2)*qtp.tensor(bra2, bra1))
    term2 = qtp.dag(term1)
    return coupling*(term1 + term2)


def Fidelity(U_target, U):
    d, _ = U_target.shape
    f = (U - U_target).norm()/d
    return 1 - f**2

def H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling):
    """
    freq 1 , 2 : splitting between bottom two levels (in GHZ)
    anh 1,2 : anharmonicity of two qubits in MHz
    g : coupling between qutrits
    """
    single_term1 = H_single(freq1, anh1)
    single_term2 = H_single(freq2, anh2)
    single_full = qtp.tensor(single_term1, single_term2)
    coupling_term = H_coupling(coupling)
    hamiltonian_full = single_full + coupling_term
    return hamiltonian_full



def qutrit_to_qubit_noDimRed(U_evolution):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    P = qtp.tensor(ket0, ket0)*qtp.dag(qtp.tensor(ket0, ket0)) + qtp.tensor(ket0, ket1)*qtp.dag(qtp.tensor(ket0,ket1)) + qtp.tensor(ket1,ket0)*qtp.dag(qtp.tensor(ket1,ket0)) + qtp.tensor(ket1,ket1)*qtp.dag(qtp.tensor(ket1,ket1))
    return qtp.dag(P)*(U_evolution)*(P)


def vectorize_operator(X):
    matrix = X.full()
    m, n = matrix.shape
    vec = matrix.reshape(m*n)
    return qtp.Qobj(vec)


def unitary_evolution_vectorized(H):
    d = H.dims[0]
    a = qtp.tensor(H, qtp.qeye(d))
    b = qtp.tensor(qtp.qeye(d), H)
    return -1j*(a - b)


def lindbladian_vectorized(A):
    #takes operator as an input upgrades it to a lindblad
    #operator is 9,9 quantum object
    d = A.dims[0]
    L = qtp.tensor(A.conj(), A) - 0.5*qtp.tensor(qtp.qeye(d), A*A.dag()) - 0.5*qtp.tensor(qtp.qeye(d), A*A.dag())
    return L


def get_master_equation(H, G, tau):
    """
    inputs :
    tau_dephase : dephasing rate between (0,1), (1,2) and (0,2) states
    tau_deco01: rate of decay from 1 to 0 state
    tau_deco12: rate of decay from 2 to 1 state
    calls lindbaldians and qutrit operators, all operators are qutip objects
    takes decoherence/dephasing coefficients as input
    """
    #todo: vecorized or not
    #function for rho
    unitary_term = unitary_evolution_vectorized(H)
    linbladian = unitary_term * 0
    for i in range(len(G)):
        linbladian += tau[i] * lindbladian_vectorized(G[i])
    return unitary_term + linbladian


def set_single_2qutrit_gate(X, pos):
    d = X.dims[0]
    if pos == 0:
        return qtp.tensor(X, qtp.qeye(d))
    if pos == 1:
        return qtp.tensor(qtp.qeye(d), X)
