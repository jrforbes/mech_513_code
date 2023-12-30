"""Two-link robot, open-loop, with energy check.

MECH 513
J R Forbes, 2022/01/09

Motion equation are of the form

M(\theta_2) \ddot{q} + f_{non}(q, \dot{q}) = 0

where q = [\theta_1 \theta_2]^T
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
# Functions and classes

# This a the two-link robot class. We need this class to create an object to then numerically integrate the nonlinear ODE.
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

        # Extract states
        th2 = x[1]  # th2

        mass = np.array([[self._J1 + self.m2 * (self.L1**2) + self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2, self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2 / 2],
                         [self._J2 + np.cos(th2) * self.L1 * self.L2 * self.m2 / 2,  self._J2]])
        mass = (mass + mass.T) / 2  # force the mass matrix to be to be symmetric

        return mass

    def _fnon(self, x):
        """Nonlinear terms in ODE."""
        # Extract states
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

    def ode(self, t, x):
        """Method for integration of ODE.

        dot_x = f(x) given x0

        Parameters
        ----------
        t : float
            Time, seconds
        x : numpy.ndarray
            Input, units

        Returns
        -------
        numpy.ndarray :
            dot_x, units
        """
        # Extract states
        dot_q = x[2:].reshape((-1, 1))

        RHS = -self._fnon(x)
        ddot_q = np.linalg.solve(self._M(x), RHS)

        dot_x = np.vstack((dot_q, ddot_q))

        return dot_x.ravel()  # flatten the array


# %%
# Numerically integrate ode
# Time
dt = 1e-3
t_start = 0
t_end = 5
t = np.arange(t_start, t_end, dt)

# Physical properties
m1 = 0.1  # kg, mass
L1 = 1.1  # m
m2 = 0.9  # kg, mass
L2 = 0.85  # m

# Initiate TwoLinkRobot instance to create an TwoLinkRobot object
nonlinear_plant = TwoLinkRobot(m1, L1, m2, L2)  # this is the object

# Find time-domain response by integrating the ODE
x0 = np.array([-15 / 180 * np.pi, -15 / 180 * np.pi, 0.1, 0.1])  # initial condition
sol = integrate.solve_ivp(
    nonlinear_plant.ode,
    (t_start, t_end),
    x0,
    args=(),
    t_eval=t,
    rtol=1e-6,
    atol=1e-6,
    method='RK45',
)

# %%
# Extract states
# Note, the scipy defult is to define the output of the integration as y, thus sol.y is the numerically integrated solution to the ODE.
sol_x = sol.y
th1 = sol_x[0, :]  # th1
th2 = sol_x[1, :]  # th2
dot_th1 = sol_x[2, :]  # dot_th1
dot_th2 = sol_x[3, :]  # dot_th2
N = th1.size

# %%
# Compute energy
E0 = nonlinear_plant.energy(sol_x[:, 0])
E_abs = np.zeros(N,)
E_rel = np.zeros(N,)
for i in range(N):
    E_abs[i] = (nonlinear_plant.energy(sol_x[:, i]) - E0)
    E_rel[i] = np.abs(E_abs[i] / E0) * 100

# %%
# Plotting
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
path = pathlib.Path('figs')
path.mkdir(exist_ok=True)

# Plot th1 and dot_th1
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

# Plot th2 and dot_th2
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

# Plot energy
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
