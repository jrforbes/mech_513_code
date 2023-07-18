"""python control basics.

J R Forbes, 2021/12/18
Based on
https://python-control.readthedocs.io/en/0.9.0/
https://python-control.readthedocs.io/en/0.9.0/control.html#function-ref
https://jckantor.github.io/CBE30338/
https://jckantor.github.io/CBE30338/05.03-Creating-Bode-Plots.html
"""

# %%
# Packagees
import numpy as np
from scipy import signal
import control
from matplotlib import pyplot as plt


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# time
dt = 1e-2
t_start = 0
t_end = 5
t = np.arange(t_start, t_end, dt)

# %%
# Create systems
# Firsr-order transfer function, P(s) = 1 / (tau * s + 1)
tau = 1 / 10
P_a = control.tf([1], [tau, 1])
P_a_num, P_a_den = np.array(P_a.num).ravel(), np.array(P_a.den).ravel()
# Convert to ss
P_a_ss = control.tf2ss(P_a_num, P_a_den)
A_a, B_a, C_a, D_a = P_a_ss.A, P_a_ss.B, P_a_ss.C, P_a_ss.D

# Mass-spring_damper system
m = 1  # kg, mass
d = 0.05  # N s / m, damper
k = 1  # N / m, spring
# Form state-space matrics.
A = np.array([[0, 1],
              [-k / m, -d / m]])
B = np.array([[0],
              [1 / m]])
C = np.array([[1, 0],
              [0, 1]])
D = np.array([[0],
              [0]])
x0 = np.array([0.25, -0.5])  # Initial conditions
n_x, n_u, n_y = A.shape[0], B.shape[1], C.shape[0]
P = control.ss(A, B, C, D)
# Convert to tf
P_tf_11 = control.ss2tf(A, B, C[0, :], D[0, :])
P_tf_11_num, P_tf_11_den = np.array(P_tf_11.num).ravel(), np.array(P_tf_11.den).ravel()

Lamb_A, V_A = np.linalg.eig(A)
print(f'plant eigenvalues are = {Lamb_A}\n')
# print(f'plant zeros = {control.zero(P)}\n')
# print(f'plant poles = {control.pole(P)}\n')


# %%
# Step response
t_P, y_P = control.step_response(P, t, x0)
y_P = y_P.reshape(n_y, -1)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$, $\dot{y}(t)$ (units)')
# Plot data
ax.plot(t_P, y_P[0, :], label='$y(t)$', color='C0')
ax.plot(t_P, y_P[1, :], label='$\dot{y}(t)$', color='C1')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_step_response.pdf')

# %%
# Impulse response
t_P, y_P = control.impulse_response(P, t, x0)
y_P = y_P.reshape(n_y, -1)

# Plot impulse response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$, $\dot{y}(t)$ (units)')
# Plot data
ax.plot(t_P, y_P[0, :], label='$y(t)$', color='C0')
ax.plot(t_P, y_P[1, :], label='$\dot{y}(t)$', color='C1')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_impulse_response.pdf')

# %%
# Initial condition (IC) response
t_P, y_P = control.initial_response(P, t, x0)
y_P = y_P.reshape(n_y, -1)

# Plot initial condition response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$, $\dot{y}(t)$ (units)')
# Plot data
ax.plot(t_P, y_P[0, :], label='$y(t)$', color='C0')
ax.plot(t_P, y_P[1, :], label='$\dot{y}(t)$', color='C1')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_IC_response.pdf')

# %%
# Time-domaine forced response
# Square wave input
u = signal.square(2 * np.pi / 2 * t)

# Forced response of each system
t_P, y_P = control.forced_response(P, t, u, x0)
y_P = y_P.reshape(n_y, -1)

# Plot forced response
fig, ax = plt.subplots(3, 1)
ax[0].set_ylabel(r'$u(t)$ (units)')
ax[1].set_ylabel(r'$y(t)$ (units)')
ax[2].set_ylabel(r'$\dot{y}(t)$ (units)')
# Plot data
ax[0].plot(t, u, label='input', color='C2')
ax[1].plot(t_P, y_P[0, :], label='$y(t)$', color='C0')
ax[2].plot(t_P, y_P[1, :], label='$\dot{y}(t)$', color='C1')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_square_wave_response.pdf')

# %%
# Bode plots
# Calculate freq, magnitude, and phase
w_shared = np.logspace(-3, 3, 1000)
mag_P, phase_P, w_P = control.bode(P[0, 0], w_shared, dB=True, deg=True)

# %%
# Analysis tools
DC_gain_P = control.dcgain(P[0, 0])
print(f'The DC gain of T(s) is {DC_gain_P}.\n')

# %%
# Controllability and observability
# Check system is controllable
Qc = control.ctrb(A, B)
rank_Qc = np.linalg.matrix_rank(Qc)
print(f'The rank of Qc is {rank_Qc}.')

# Check system is observable
Qo = control.obsv(A, C)
rank_Qo = np.linalg.matrix_rank(Qo)
print(f'The rank of Qo is {rank_Qo}.')

# Compute controllability Gramian
Wc = control.gram(P, 'c')
print(f'The controllability Gramian is\n', Wc)

# Compute observability Gramian
Wo = control.gram(P, 'o')
print(f'The observability Gramian is\n', Wo, '\n')

# %%
# Lyapunov equation
P_lyap = control.lyap(A.T, np.eye(2))
Lamb_P, V_P = np.linalg.eig(P_lyap)
print(f'Eigenvalues of P are = {Lamb_P}\n')

# %%
# Pole placement
K_place = control.place(A, B, [-1, -2])
A_cl_place = A - B @ K_place
cl_eig, cl_eig_vec = np.linalg.eig(A_cl_place)
print(f'The closed-loop eigenvalues are', cl_eig, '\n')

# %%
# LQR via care
Q = np.array([[10, 0],
              [0, 10]])  # State penalty
R = np.array([[1]])  # Control penalty
P_ric_care, cl_eig_care, K_care = control.care(A, B, Q, R)
print(f'The LQR gain is K = {K_care}')
  
# %%
# LQR via lqr
K_lqr, P_ric_lqr, cl_eig_lqr = control.lqr(P, Q, R)
# print(cl_eig_lqr)
print(f'The LQR gain is K = {K_lqr}\n')

# %%
# Form closed-loop system
K = control.ss([], [], [], K_lqr)
G_cl = control.feedback(P, K)
A_cl = G_cl.A
cl_eig, cl_eig_vec = np.linalg.eig(A_cl)
print(f'The closed-loop eigenvalues are', cl_eig)

# Step response of closed-loop system
t_G_cl, y_G_cl = control.step_response(G_cl, t, x0)
y_G_cl = y_G_cl.reshape(n_y, -1)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$x_1(t)$ (units)')
# Plot data
ax.plot(t_P, y_P[0, :], label='open-loop', color='C0')
ax.plot(t_G_cl, y_G_cl[0, :], label='closed-loop', color='C1')
ax.legend(loc='lower left')
fig.tight_layout()
# fig.savefig('figs/control_step_response_OL_vs_CL.pdf')

# %%
plt.show()
