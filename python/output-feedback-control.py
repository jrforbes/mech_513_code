"""Simulate output feedback control."""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


def main():
    """Simulate system with and without controller."""

    # Set plot options.
    plt.rc('axes', grid=True)
    plt.rc('grid', linestyle='--')

    # Set simulation time.
    t_span = (0, 10)
    t_step = 1e-3
    t_eval = np.arange(*t_span, t_step)

    # Create Duffing oscillator.
    sys = DuffingOscillator(m=1, k=0.5, d=0, beta=0.9)
    x0_sys = np.array([1, 0])

    # Simulate Duffing oscillator alone.
    sol = integrate.solve_ivp(
        lambda t, x: sys.f(t, x, np.array([0])),
        t_span,
        x0_sys,
        t_eval=t_eval,
    )

    # Create PD controlelr.
    ctrl = Pd(k_p=100, k_d=10, tau_f=0.01)
    x0_ctrl = np.array([0])

    # Set up closed-loop IC.
    x0_cl = np.concatenate((x0_sys, x0_ctrl))
    # Setpoint.
    x_sp = np.array([0])

    def closed_loop(t, x):
        """Closed-loop system."""
        # Split state.
        x_sys = x[:2]
        x_ctrl = x[2:]
        # Compute error.
        error = x_sp - x_sys[0]
        # Advance controller state.
        x_dot_ctrl = ctrl.f(t, x_ctrl, error)
        # Compute control output.
        u_ctrl = ctrl.g(t, x_ctrl, error)
        # Advance system state.
        x_dot_sys = sys.f(t, x_sys, u_ctrl)
        # Concatenate state derivatives.
        x_dot = np.concatenate((x_dot_sys, x_dot_ctrl))
        return x_dot

    # Simulate closed-loop system.
    sol_cl = integrate.solve_ivp(closed_loop, t_span, x0_cl, t_eval=t_eval)

    # Plot results.
    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    ax[0].plot(sol.t, sol.y[0, :])
    ax[0].plot(sol.t, sol_cl.y[0, :])
    ax[1].plot(sol.t, sol.y[1, :])
    ax[1].plot(sol.t, sol_cl.y[1, :])
    ax[0].set_ylabel(r'$x(t)$ (units)')
    ax[1].set_ylabel(r'$\dot{x}(t)$ (units/s)')
    for a in ax:
        a.set_xlabel(r'$t$ (s)')

    plt.show()


class DuffingOscillator():
    """Duffing oscillator."""

    def __init__(self, m, k, d, beta):
        """Instantiate ``DuffingOscillator``.

        Parameters
        ----------
        m : float
            Mass.
        k : float
            Stiffness.
        d : float
            Damping.
        beta : float
            Nonlinear coefficient.
        """
        self.m = m
        self.k = k
        self.d = d
        self.beta = beta

    def f(self, t, x, u):
        """Implement differential equation"""
        # Linear part
        A = np.array([
            [0, 1],
            [-self.k / self.m, -self.d / self.m],
        ])
        B = np.array([
            [0],
            [1 / self.m],
        ])
        # Nonlinear part
        nonlin = np.array([
            [0],
            [-self.beta * x[0]**3],
        ])
        # State derivative
        x_dot = A @ x.reshape((-1, 1)) + B @ u.reshape((-1, 1)) + nonlin
        return np.ravel(x_dot)


class Pd():
    """Proportional-derivative controller."""

    def __init__(self, k_p, k_d, tau_f):
        """Instantiate ``Pd``.

        Parameters
        ----------
        k_p : float
            Proportional gain.
        k_d : float
            Derivative gain.
        tau_f : float
            Derivative filter time constant.
        """
        self.k_p = k_p
        self.k_d = k_d
        self.tau_f = tau_f

    def f(self, t, x, u):
        A = -1 / self.tau_f
        B = -self.k_d / self.tau_f**2
        x_dot = A * x + B * u
        return x_dot

    def g(self, t, x, u):
        C = 1
        D = self.k_p + self.k_d / self.tau_f
        y = C * x + D * u
        return y


if __name__ == '__main__':
    main()
