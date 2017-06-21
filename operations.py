"""
Operations used in the simulation
of the transmon qubits.

Qutip is used to optain the ket, bra state
representation of the qutrits.

created-on: 02/06/17
"""
import numpy as np
import qutip as qtp


def basis_qutrit():
    """Return all the basis states in the sinlge qutrit space."""
    ket0 = qtp.basis(3, 0)
    ket1 = qtp.basis(3, 1)
    ket2 = qtp.basis(3, 2)
    bra0 = qtp.dag(ket0)
    bra1 = qtp.dag(ket1)
    bra2 = qtp.dag(ket2)
    return ket0, ket1, ket2, bra0, bra1, bra2


def basis_qubit():
    """Return all the basis states in the sinlge qubit space."""
    ket0 = qtp.basis(2, 0)
    ket1 = qtp.basis(2, 1)
    bra0 = qtp.dag(ket0)
    bra1 = qtp.dag(ket1)
    return ket0, ket1, bra0, bra1

def projector():
    """Proyector from the qutrit space to the qubit space."""
    bitKet0, bitKet1, bitBra0, bitBra1 = basis_qubit()
    triKet0, triKet1, triKet2, triBra0, triBra1, triBra2 = basis_qutrit()
    P = bitKet0*triBra0 + bitKet1*triBra1
    return P

def projector2qutrits():
    """Proyector from the 2 qutrit space to 2 qubit space."""
    p = projector()
    return qtp.tensor(p, p)

def projector_master():
    """Proyector from qutrit space to qubit space without matrix dimension reducction."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    P = ket0*bra0 + ket1*bra1
    return P

def projector2qutrits_master():
    """Proyector from 2-qutrit space to 2-qubit space without matrix dimension reducction."""
    p = projector_master()
    return qtp.tensor(p, p)


def target_iSWAP():
    """Return iSWAP gate in qubit space."""
    ket0, ket1, bra0, bra1 = basis_qubit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) - 1j*qtp.tensor(ket0, ket1)*qtp.tensor(bra1, bra0) \
        - 1j*qtp.tensor(ket1, ket0)*qtp.tensor(bra0, bra1) + qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    return A


def target_iSWAP_master():
    """Return iSWAP gate in qutrit space."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) - 1j*qtp.tensor(ket0, ket1)*qtp.tensor(bra1, bra0) \
        - 1j*qtp.tensor(ket1, ket0)*qtp.tensor(bra0, bra1) + qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    return A


def target_CPHASE():
    """Return CPHASE gate in qubit space."""
    ket0, ket1, bra0, bra1 = basis_qubit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) + qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        + qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    return A


def target_CPHASE_master():
    """Return CPHASE gate in qutrit space."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) + qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        + qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    return A


def matrix_optimize(theta1, theta2, theta3):
    """Return matrix multiplication M=exp(theta3*id)*exp(theta2*Z_A)*exp(theta1*Z_B)."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    A = qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) + qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        - qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)
    B =  qtp.tensor(ket0, ket0)*qtp.tensor(bra0, bra0) - qtp.tensor(ket0, ket1)*qtp.tensor(bra0, bra1) \
        + qtp.tensor(ket1, ket0)*qtp.tensor(bra1, bra0) - qtp.tensor(ket1, ket1)*qtp.tensor(bra1, bra1)

    A = (1j * theta1 * A).expm()
    B = (1j * theta2 * B).expm()
    C = qtp.qeye([3, 3])*np.exp(1j*theta3)
    return A*B*C


def H_single(freq1, anh1):
    """Hamiltonian for the single transmon qubit."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    H = freq1*ket1*bra1 + (2*freq1 - anh1)*ket2*bra2
    return H


def H_coupling(coupling):
    """Hamiltoninan for the coupling between the transmon qubits."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    term1 = (qtp.tensor(ket0, ket1)*qtp.tensor(bra1, bra0) + np.sqrt(2)*qtp.tensor(ket0, ket2)*qtp.tensor(bra1, bra1)
             + np.sqrt(2)*qtp.tensor(ket2, ket0)*qtp.tensor(bra1, bra1) + 2* qtp.tensor(ket1, ket2)*qtp.tensor(bra2, bra1))
    term2 = qtp.dag(term1)
    return coupling*(term1 + term2)


def fidelity(U_target, U):
    """Fidelity between U_target and U."""
    d, _ = U_target.shape
    f = (U - U_target).norm()/d
    # print(f*d)
    return 1 - f**2


def H_coupled_qutrit(freq1, anh1, freq2, anh2, coupling):
    """
    Full Hamiltonian of the coupled qubit transmons.

    freq 1 , 2 : splitting between bottom two levels (in GHZ)
    anh 1,2 : anharmonicity of two qubits in MHz
    g : coupling between qutrits
    """
    idd = qtp.qeye(3)
    single_term1 = H_single(freq1, anh1)
    single_term2 = H_single(freq2, anh2)
    single_full = qtp.tensor(single_term1, idd) + qtp.tensor(idd, single_term2)
    coupling_term = H_coupling(coupling)
    hamiltonian_full = single_full + coupling_term
    return hamiltonian_full


def vectorize_operator(X):
    """Return density matrix X in vector form."""
    matrix = X
    m, n = matrix.shape
    vec = matrix.reshape(m*n)
    return vec


def un_vectorize(X):
    """Transform back density matrix from vector to matrix form."""
    # Only works on ket
    vector = X
    m = vector.shape[0]
    m = int(np.sqrt(m))
    matrix = vector.reshape(m, m)
    return matrix


def unitary_evolution_vectorized(H):
    """Vectorized representation for the full Hamiltonian."""
    d = H.dims[0]
    a = qtp.tensor(H, qtp.qeye(d))
    b = qtp.tensor(qtp.qeye(d), H)
    return -1j*(a - b)


def operators():
    """Return the operators for the dephasing and relaxation effects."""
    w = np.exp(1j*np.pi/3.)
    gz = qtp.Qobj(np.array([[1, 0, 0], [0, w, 0], [0, 0, w**2]]))
    gx1 = qtp.Qobj(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    gx2 = qtp.Qobj(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    g = [set_single_2qutrit_gate(gz, 0), set_single_2qutrit_gate(gz, 1),
         set_single_2qutrit_gate(gx1, 0), set_single_2qutrit_gate(gx1, 1),
         set_single_2qutrit_gate(gx2, 0), set_single_2qutrit_gate(gx2, 1)]
    return g


def lindbladian_vectorized(A):
    """Linbladian of a single operator in vecorized form."""
    # takes operator as an input upgrades it to a lindblad
    # operator is 9,9 quantum object
    d = A.dims[0]
    L = qtp.tensor(A.conj(), A) - 0.5*qtp.tensor(qtp.qeye(d), A*A.dag()) - 0.5*qtp.tensor(qtp.qeye(d), A*A.dag())
    return L


def get_master_equation(H, G, tau):
    """
    Return the full Master equation in vectorized form.

    inputs :
    H : Hamiltonian matrix for the unitary evolution.
    G : list of operators acting in the Linbladian.
    tau : vector containgin the rates of all the decoherence effects
    as tau=(dephasing_A, dephasing_B, relax10_A, relax10_B, relax21_A, relax21_B)
    """
    unitary_term = unitary_evolution_vectorized(H)
    linbladian = unitary_term * 0
    for i in range(len(G)):
        linbladian += tau[i] * lindbladian_vectorized(G[i])
    return unitary_term + linbladian


def set_single_2qutrit_gate(X, pos):
    """Single qutrit gate transform to 2-qutrit space."""
    d = X.dims[0]
    if pos == 0:
        return qtp.tensor(X, qtp.qeye(d))
    if pos == 1:
        return qtp.tensor(qtp.qeye(d), X)


def trace_dist(A,B):
    """Trace distance between two operators."""
    diff = A-B
    diff = np.transpose(np.conj(diff)).dot(diff)
    vals, _ = np.linalg.eig(diff)
    return float(np.real(0.5 * np.sum(np.sqrt(np.abs(vals)))))


def special_state():
    """Specail 2-qutrit state."""
    ket0, ket1, ket2, bra0, bra1, bra2 = basis_qutrit()
    s = qtp.tensor(ket0,ket0) + qtp.tensor(ket0,ket1) + qtp.tensor(ket1,ket0) + qtp.tensor(ket1,ket1)
    s = s/2
    return s * s.dag()
