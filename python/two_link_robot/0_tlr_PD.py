"""Two-link robot with (filtered) PD control.

MECH 513
J R Forbes, 2022/01/09
"""
# %%
# Import packages
import numpy as np
import control
from scipy import signal
from scipy import integrate
from scipy import constants as cst  # https://docs.scipy.org/doc/scipy/reference/constants.html
from matplotlib import pyplot as plt
import pathlib

# %%
# Functions and classes.


# This a the two-link robot class.
class TwoLinkRobot:
    def __init__(self, m1, L1, m2, L2):
        """Constructor

        Parameters
        ----------
        mp : float
            Mass of pendulum, kg
        Lp : float
            Length of pendulum, m
        mr : float
            Mass of rotational arm, kg
        Lr : float
            Length of rotational arm, m
        """
        self.m1 = m1
        self.L1 = L1
        self.m2 = m2
        self.L2 = L2

    # Using the @property decorator (https://www.programiz.com/python-programming/property) makes
    # this function behave like a read-only variable. Whenever it's accessed, its value is recomputed
    @property
    def _J1(self):
        """Mass moment of inerta associated with first link."""
        return self.m1 * (self.L1**2) / 3  # kg / m**2

    @property
    def _J2(self):
        """Mass moment of inerta associated with second link."""
        return self.m2 * (self.L2**2) / 3  # kg / m**2

    def _M(self, x):
        """Mass matrix."""

        # Extract states.
        th2 = x[1]  # th2

        mass = np.array([[self._J1 + self.m2 * (self.L1**2) + self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2, self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2 / 2],
                         [self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2 / 2,  self._J2]])
        mass = (mass + mass.T) / 2  # force the mass matrix to be to be symmetric

        return mass

    def _fnon(self, x):
        """Nonlinear terms in ODE."""

        # Extract states.
        th2 = x[1]  # th2
        dot_th1 = x[2]  # dot_th1
        dot_th2 = x[3]  # dot_th2
        dot_q = x[2:].reshape((-1, 1))  # angle rates

        a = np.array([[0], [- 1 / 2 * dot_th1 * (dot_th1 + dot_th2) * np.sin(th2) * self.L1 * self.L2 * self.m2]])
        dot_M = -np.sin(th2) * self.L1 * self.L2 * self.m2 * dot_th2 * np.array([[1, 0.5], [0.5, 0]])
        nonlinear_forces = dot_M @ dot_q - a

        return nonlinear_forces

    def energy(self, x):
        dot_q = x[2:].reshape((-1, 1))  # angle rates
        T = 1 / 2 * dot_q.T @ self._M(x) @ dot_q  # kinetic energy
        E = T
        return E

    def f(self, t, x, u):
        """Method for integration of ODE.

        dot_x = f(x, u) given x0

        Parameters
        ----------
        t : float
            time, seconds
        x : numpy.ndarray
            state, units
        u : numpy.ndarray
            input, units

        Returns
        -------
        numpy.ndarray :
            dot_x, units
        """
        # Extract states.
        dot_q = x[2:].reshape((-1, 1))

        RHS = u.reshape((-1, 1)) - self._fnon(x)
        ddot_q = np.linalg.solve(self._M(x), RHS)
        dot_x = np.vstack((dot_q, ddot_q))

        return dot_x.ravel()  # flatten the array

    def g(self, t, x):
        """Method for computing measurment.

        y = g(x, u)

        Parameters
        ----------
        t : float
            Time, seconds
        x : numpy.ndarray
            state, units

        Returns
        -------
        numpy.ndarray :
            y, units
        """
        # Measurments.
        y = x[:2].reshape((-1, 1))

        return y.ravel()  # flatten the array


# This a filtered PD control class.
class FilteredPDControl:
    def __init__(self, Kp, Kd, tau1, tau2):
        """Constructor for filter PD controller.

        Parameters
        ----------
        Kp : float
            Proportional gain.
        Kd : float
            Derivative gain.
        tau1 : float
            First derivative filter time constant.
        tau2 : float
            Second derivative filter time constant.
        """
        self.Kp = Kp
        self.Kd = Kd
        self.tau1 = tau1
        self.tau2 = tau2

    def f(self, t, xc, uc):
        """Control process model."""
        Ac = np.array([[-1 / self.tau1, 0],
                       [0, -1 / self.tau2]])

        Bc = self.Kd @ np.array([[-1 / self.tau1**2, 0],
                                 [0, -1 / self.tau2**2]])

        dot_xc = Ac @ xc.reshape((-1, 1)) + Bc @ uc.reshape((-1, 1))

        return dot_xc.ravel()

    def g(self, t, xc, uc):
        """Control measurment model."""
        Cc = np.eye(2)

        Dc = self.Kp + self.Kd @ np.array([[1 / self.tau1, 0], [0, 1 / self.tau2]])

        yc = Cc @ xc.reshape((-1, 1)) + Dc @ uc.reshape((-1, 1))

        return yc.ravel()



# %%
# Numerically integrate ODE.
# Time
dt = 1e-3
t_start = 0
t_end = 5
t = np.arange(t_start, t_end, dt)

# Physical properties.
m1 = 0.1  # kg, mass
L1 = 1.1  # m
m2 = 0.9  # kg, mass
L2 = 0.85  # m

# Initiate TwoLinkRobot instance to create an TwoLinkRobot object.
sys = TwoLinkRobot(m1, L1, m2, L2)  # this is the object
x_sys0 = np.array([0 / 180 * np.pi, 0 / 180 * np.pi, -0.05, 0.05])  # initial condition
r_des = np.array([20 / 180 * np.pi, 40 / 180 * np.pi])  # desired angles

# Initiate FilteredPDControl instane to create a FilteredPDControl object.
Kp = np.array([[15, 0], [0, 10]])  # N * m / rad
Kd = np.array([[5, 0], [0, 2.5]])  # N * m * s / rad
tau1 = 1 / 100
tau2 = 1 / 115
ctrl = FilteredPDControl(Kp, Kd, tau1, tau2)
x_ctrl0 = np.array([0, 0])

# Set up closed-loop IC.
x_cl0 = np.concatenate((x_sys0, x_ctrl0))

def closed_loop(t, x):
    """Closed-loop system."""
    # Split state.
    x_sys = x[:4]
    x_ctrl = x[4:]

    # Measurment.
    y = sys.g(t, x_sys)
    # Compute error.
    error = r_des - y
    # Compute control.
    u_ctrl = ctrl.g(t, x_ctrl, error)

    # Advance system state.
    dot_x_sys = sys.f(t, x_sys, u_ctrl)
    # Advance controller state.
    dot_x_ctrl = ctrl.f(t, x_ctrl, error)

    # Concatenate state derivatives.
    x_dot = np.concatenate((dot_x_sys, dot_x_ctrl))
    return x_dot

# Find time-domain response by integrating the ODE
sol = integrate.solve_ivp(
    closed_loop,
    (t_start, t_end),
    x_cl0,
    t_eval=t,
    rtol=1e-6,
    atol=1e-6,
    method='RK45',
)

# %%
# Extract states.
sol_x = sol.y
x_sys = sol_x[:4, :]
q = x_sys[:2, :]
dot_q = x_sys[2:, :]
th1 = sol_x[0, :]  # th1
th2 = sol_x[1, :]  # th2
dot_th1 = sol_x[2, :]  # dot_th1
dot_th2 = sol_x[3, :]  # dot_th2
x_ctrl = sol_x[4:, :]  # dot_th2
N = th1.size

# %%
# Compute error and control (for plotting purposes)
error_th = np.zeros((2, N))
u_ctrl = np.zeros((2, N))
for i in range(N):
    error_th[:, [i]] = r_des.reshape((-1, 1)) - q[:, [i]]
    u_ctrl[:, [i]] = np.reshape(ctrl.g(t, x_ctrl[:, i].ravel(), error_th[:, i].ravel()), (-1, 1))


# %%
# Copmute energy.
E0 = sys.energy(x_sys[:, 0])
E_abs = np.zeros(N, )
E_rel = np.zeros(N, )
for i in range(N):
    E_abs[i] = (sys.energy(x_sys[:, i]) - E0)
    E_rel[i] = np.abs(E_abs[i] / E0) * 100

# %%
# Plotting.
# Plotting parameters.
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
path = pathlib.Path('figs')
path.mkdir(exist_ok=True)

# Plot th1 and dot_th1.
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$\theta_1(t)$ (deg)')
ax[1].set_ylabel(r'$\dot{\theta}_1(t)$ (deg/s)')
# Plot data
ax[0].plot(t, th1 * 180 / np.pi)
ax[1].plot(t, dot_th1 * 180 / np.pi)
# for a in np.ravel(ax):
#     a.legend(loc='lower right')
fig.tight_layout()
# fig.savefig(path.joinpath('.pdf'))

# Plot th2 and dot_th2.
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$\theta_2(t)$ (deg)')
ax[1].set_ylabel(r'$\dot{\theta}_2(t)$ (deg/s)')
# Plot data
ax[0].plot(t, th2 * 180 / np.pi)
ax[1].plot(t, dot_th2 * 180 / np.pi)
# for a in np.ravel(ax):
#     a.legend(loc='lower right')
fig.tight_layout()
# fig.savefig(path.joinpath('.pdf'))

# Plot control.
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$u_1(t)$ (N/m)')
ax[1].set_ylabel(r'$u_2(t)$ (N/m)')
# Plot data
ax[0].plot(t, u_ctrl[0, :])
ax[1].plot(t, u_ctrl[1, :])
# for a in np.ravel(ax):
#     a.legend(loc='lower right')
fig.tight_layout()
# fig.savefig(path.joinpath('.pdf'))

# Plot energy.
fig, ax = plt.subplots(2, 1)
# Format axes
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
ax[0].set_ylabel(r'$E_{abs}(t)$ (J)')
ax[1].set_ylabel(r'$E_{rel}(t) \times 100\%$ (unitless)')
# Plot data
ax[0].plot(t, E_abs)
ax[1].plot(t, E_rel)
# for a in np.ravel(ax):
#     a.legend(loc='lower right')
fig.tight_layout()
# fig.savefig(path.joinpath('.pdf'))

# %%
# Plot show
plt.show()
