
import numpy as np
import matplotlib.pyplot as plt
import qutip as qtp
from IPython.display import display, Math, Latex



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

def proyect():
    bitKet0, bitKet1, bitBra0, bitBra1 = basisQubit()
    triKet0, triKet1, triKet2, triBra0, triBra1, triBra2 = basis_kets()
    P = bitKet0*triBra0 + bitKet1*triBra1
    return P


def H_term(freq1, anh1, freq2, anh2):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_kets()
    qubit_term1 = freq1*ket1*bra1 + (2*freq1 - anh1)*ket2*bra2
    qubit_term2 = freq2*ket1*bra1 + (2*freq2 - anh2)*ket2*bra2
    qubit_term = qtp.tensor(qubit_term1, qubit_term2)
    return qubit_term


def coupling_hamiltonian(coupling):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_kets()
    term1 = (qtp.tensor(ket0, ket1)*qtp.tensor(bra1, bra0) + np.sqrt(2)*qtp.tensor(ket0, ket2)*qtp.tensor(bra1, bra1)
             + np.sqrt(2)*qtp.tensor(ket2, ket0)*qtp.tensor(bra1, bra1) + 2* qtp.tensor(ket1, ket2)*qtp.tensor(bra2, bra1))
    term2 = qtp.dag(term1)

    return coupling*(term1 + term2)



def dir_coupled_qutrit(freq1, anh1, freq2, anh2, coupling):
    """
    freq 1 , 2 : splitting between bottom two levels (in GHZ)
    anh 1,2 : anharmonicity of two qubits in MHz
    g : coupling between qutrits
    """
    qubit_term_hamiltonian = qubit_term(freq1, anh1, freq2, anh2)
    coupling_term = coupling_hamiltonian(coupling)
    hamiltonian_direct = qubit_term_hamiltonian + coupling_term
    return hamiltonian_direct



def qutrit_to_qubit_noDimRed(U_evolution):
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_kets()
    P = qtp.tensor(ket0, ket0)*qtp.dag(qtp.tensor(ket0, ket0)) + qtp.tensor(ket0, ket1)*qtp.dag(qtp.tensor(ket0,ket1)) + qtp.tensor(ket1,ket0)*qtp.dag(qtp.tensor(ket1,ket0)) + qtp.tensor(ket1,ket1)*qtp.dag(qtp.tensor(ket1,ket1))
    return qtp.dag(P)*(U_evolution)*(P)


def vectorize_operator(operator):
    matrix = operator.full()
    m, n = matrix.shape
    vec = matrix.reshape(m*n)
    return qtp.Qobj(vec)


def unitary_evolution_vectorized(H):
    d, _ = H.shape
    a = qtp.tensor(H, qtp.qeye(d))
    b = qtp.tensor(qtp.qeye(d), H)
    return 1j*(a - b)


# U_qubit = qutrit_to_qubit(U_evolution)
P = proyector()
P = qtp.tensor(P, P)
#
# iS = P * U_evolution * qtp.dag(P)
# iS * qtp.dag(iS)
