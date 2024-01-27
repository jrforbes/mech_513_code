'''Use of cvxpy and mosek to design controller.

Steven Dahdah, modified by James Forbes
January 31, 2023

See
https://www.cvxpy.org/

For list of fuctions,
https://www.cvxpy.org/api_reference/cvxpy.atoms.html
'''
# %%
# Libraries
import cvxpy
import numpy as np
import control
from matplotlib import pyplot as plt


def main():
    # A matrix bottom row
    a0 = -1
    a1 = 3
    a2 = -2

    # A matrix
    A = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [-a0, -a1, -a2],
    ])

    # B matrix
    B = np.array([
        [0],
        [0],
        [1],
    ])

    # C matrix
    C = np.array([1, 0, 0])

    # D matrix
    D = np.array([0])

    Lamb_A, V_A = np.linalg.eig(A)
    print(f'Open-loop eigenvalues are = {Lamb_A}\n')

    # K_nom = np.array([[3.4495, 2.2975, 5.6871]])

    # Create variables
    F = cvxpy.Variable((1, 3))
    X = cvxpy.Variable((3, 3), symmetric=True)
    # Create objective
    objective = cvxpy.Minimize(cvxpy.norm(X - np.eye(3), 'fro'))
    # Create constraints
    epsilon = 1e-6
    constraints = [
        X >> epsilon,
        (A @ X) + (X @ A.T) - (B @ F) - (F.T @ B.T) << 0,
    ]
    # Create problem
    prob = cvxpy.Problem(objective, constraints)
    result = prob.solve(solver='MOSEK')
    # Extract results
    X_opt = X.value
    F_opt = F.value
    K_opt = np.linalg.solve(X_opt.T, F_opt.T)
    K_opt = K_opt.T
    A_cl = A - B @ K_opt
    # A_cl = A - B @ K_nom

    print(f'Gain matrix K = {K_opt}\n')

    Lamb_A_cl, V_A_cl = np.linalg.eig(A_cl)
    print(f'Closed-loop eigenvalues are = {Lamb_A_cl}\n')

    # System
    P_ol = control.ss(A, B, C, D)
    P_cl = control.ss(A_cl, B, C, D)
    x0 = np.array([10, -5, 2])  # Initial conditions
    # time
    dt = 1e-2
    t_start = 0
    t_end = 100
    t = np.arange(t_start, t_end, dt)

    # Step response
    t_ol, y_ol = control.initial_response(P_ol, t, x0)    
    t_cl, y_cl = control.initial_response(P_cl, t, x0)
    
    # Plotting parameters
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', size=14)
    plt.rc('lines', linewidth=2)
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$y(t)$ (units)')
    # Plot data
    # ax.plot(t_ol, y_ol, label=r'$y_{ol}(t)$', color='C0')
    ax.plot(t_cl, y_cl, label=r'$y_{cl}(t)$', color='C1')
    ax.legend(loc='upper right')
    fig.tight_layout()
    # fig.savefig('figs/control_IC_response.pdf')


if __name__ == '__main__':
    main()

# %%
