def cost_func(x, P, U_evolution, U_target):
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
    theta3 = x[3]
    unitary_product_phase = matrix_optimize(theta1, theta2, theta3)
    unitary = unitary_product_phase*U_evolution
    projected_unitary = P*unitary*P.dag()
    F = fidelity(U_target, projected_unitary)
    infidelity = 1- F
    return infidelity

def cost_func(x, P, U_evolution, U_target):
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
    theta3 = x[3]
    unitary_product_phase = matrix_optimize(theta1, theta2, theta3)
    unitary = unitary_product_phase*U_evolution
    projected_unitary = P*unitary*P.dag()
    F = fidelity(U_target, projected_unitary)
    infidelity = 1- F
    return infidelity