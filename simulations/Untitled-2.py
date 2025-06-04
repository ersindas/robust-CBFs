# --------------------------------------------------------------------
# unicycle_cbf_grid_vs_zoopt.py
# --------------------------------------------------------------------
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- external modules from your repo --------------------------------
from CBF import Rob_CBF            # needs osqp
from Nonlinear_System import ControlAffine
from zoopt import Dimension, Parameter, Objective, Opt

# --------------------------------------------------------------------
# 0 · Plant and CBF definitions  (identical to your notebook) ---------
# --------------------------------------------------------------------
def f(x):  # drift
    return np.array([x[3]*np.cos(x[2]),
                     x[3]*np.sin(x[2]),
                     0,
                     0])

def g(x):  # actuation matrix
    return np.array([[0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]])

unicycle = ControlAffine(f, g, n=4, m=2)
a_lane   = 5                       # lane-keeping parameter

def h(x):                                       # lane CBF
    return -2*x[1]*x[3]*np.sin(x[2]) + a_lane*(1 - x[1]**2)

def dh(x):                                      # ∇h(x)
    return np.array([0,
                     -2*x[3]*np.sin(x[2]) - 2*a_lane*x[1],
                     -2*x[1]*x[3]*np.cos(x[2]),
                     -2*x[1]*np.sin(x[2])])

x0 = np.array([0, 0, 0.1, 0.1])     # initial state

# reference trajectory (unchanged)
A, ω = 1.5, 1
y_des      = lambda t: A*np.sin(ω*t)
y_dot_des  = lambda t: A*ω*np.cos(ω*t)
y_ddot_des = lambda t: -A*ω**2*np.sin(ω*t)
x_des      = lambda t: t
x_dot_des  = lambda t: 1
x_ddot_des = lambda t: 0

def ctrl(x, t):                    # feedback-linearising track-law
    k1, k2 = 4, 4
    v1 = -k1*(x[0]-x_des(t)) - k2*(x[3]*np.cos(x[2]) - x_dot_des(t)) + x_ddot_des(t)
    v2 = -k1*(x[1]-y_des(t)) - k2*(x[3]*np.sin(x[2]) - y_dot_des(t)) + y_ddot_des(t)
    V  = np.hstack([v1, v2])
    Tf = np.array([[-x[3]*np.sin(x[2]), np.cos(x[2])],
                   [ x[3]*np.cos(x[2]), np.sin(x[2])]])
    return np.linalg.inv(Tf) @ V

# time-varying state-error bound
delta = lambda x, t: 0.2 + 0.2*np.sin(0.01*t) * np.ones_like(x)

# “sensor” that respects the bound
est   = lambda x, t: x - np.tanh(x) * delta(x, t)

# --------------------------------------------------------------------
# 1 · Robust CBF objects (copied) ------------------------------------
# --------------------------------------------------------------------
alpha = 2.0
rob_cbf        = Rob_CBF(alpha, h, dh, unicycle, k1=2, k2=2)  # used in control loop
rob_cbf_sigma  = Rob_CBF(alpha, h, dh, unicycle, k1=2, k2=2)  # same object for σ̂

N_SIGMA_SAMPLES = 20            # random points in Δ-ball for σ̂

def sigma(xhat, kdes, d, k1, k2):
    """ σ̂(x̂)  =  max‖k(x)−k(x̂)‖ over random Δ-ball samples. """
    rob_cbf_sigma.k1, rob_cbf_sigma.k2 = k1, k2
    k_fun = lambda x: rob_cbf_sigma.filter(x, kdes)
    pts   = np.random.uniform(-d, d, size=(N_SIGMA_SAMPLES, 4)) + xhat
    return np.max(np.abs([k_fun(p) for p in pts] - k_fun(xhat)))

# --------------------------------------------------------------------
# 2 · Tuning methods --------------------------------------------------
# --------------------------------------------------------------------
def get_ks_zoopt(xhat, d, kdes):
    """Original ZoOpt search (unchanged)."""
    def cost(sol):
        k1, k2 = sol.get_x()
        return max(sigma(xhat, kdes, d, k1, k2) - k1, 0)/(2*k2) + 0.1*k1 + 0.1*k2
    dim  = Dimension(size=2, regs=[[0, 2], [0, 2]], tys=[True, True])
    obj  = Objective(cost, dim)
    sol  = Opt.min(obj, Parameter(budget=200))
    return sol.get_x()

def get_ks_grid(xhat, d, kdes, N=10):
    """10×10 grid search in the same box [0,2]×[0,2]."""
    g1 = np.linspace(0, 2, N)
    g2 = np.linspace(0, 2, N)
    best_val, best = np.inf, (1.0, 1.0)
    for k1 in g1:
        for k2 in g2:
            val = max(sigma(xhat, kdes, d, k1, k2) - k1, 0)/(2*k2) + 0.1*k1 + 0.1*k2
            if val < best_val:
                best_val, best = val, (k1, k2)
    return best

# --------------------------------------------------------------------
# 3 · Simulation parameters ------------------------------------------
# --------------------------------------------------------------------
T_end, dt_sample, dt_int = 10.0, 0.1, 1e-2
N_samples  = int(T_end / dt_sample)
N_int      = int(T_end / dt_int)

t_grid     = np.round(np.linspace(0, T_end, N_samples+1)[:-1], 3)
t_int      = np.round(np.linspace(0, T_end, N_int+1), 3)

x_state    = np.zeros((4, len(t_int)))
x_state[:, 0] = x0

# logs
k1_grid, k2_grid, t_grid_time = [], [], []
k1_opt , k2_opt , t_opt_time  = [], [], []

# --------------------------------------------------------------------
# 4 · Main loop  (GRID drives the plant; ZoOpt logged for comparison)
# --------------------------------------------------------------------
N_int_per_sample = int(dt_sample / dt_int)          # 0.1 / 0.01  → 10
t_int = np.arange(0.0, T_end + dt_int/2, dt_int)    # robust to fp round-off

for i, t_s in enumerate(t_grid):
    idx0 = i * N_int_per_sample          # index of current sample start
    idx1 = idx0 + N_int_per_sample       # index of next sample start

    x_true = x_state[:, idx0]            # state at t_s
    x_hat  = est(x_true, t_s)
    d_vec  = delta(x_true, t_s)
    u_des  = ctrl(x_true, t_s)

    # ---- ZoOpt tuner (timed, for comparison) ------------------------
    t0 = time.perf_counter()
    k1_o, k2_o = get_ks_zoopt(x_hat, d_vec, u_des)
    t_opt_time.append(time.perf_counter() - t0)
    k1_opt.append(k1_o);  k2_opt.append(k2_o)

    # ---- GRID tuner (timed, drives the controller) ------------------
    t0 = time.perf_counter()
    k1_g, k2_g = get_ks_grid(x_hat, d_vec, u_des)
    t_grid_time.append(time.perf_counter() - t0)
    k1_grid.append(k1_g); k2_grid.append(k2_g)

    # use GRID values online
    rob_cbf.k1, rob_cbf.k2 = k1_g, k2_g
    u_safe = rob_cbf.filter(x_hat, u_des)

    # ---- integrate plant for exactly dt_sample ----------------------
    t_span = (t_int[idx0], t_int[idx1])                 # [t_s, t_s+0.1]
    t_eval = t_int[idx0+1 : idx1+1]                     # interior points
    sol = solve_ivp(lambda _t, y: unicycle.RHS(y, u_safe),
                    t_span, x_true, t_eval=t_eval)
    x_state[:, idx0+1 : idx1+1] = sol.y


# --------------------------------------------------------------------
# 5 · Post-processing (extended) -------------------------------------
# --------------------------------------------------------------------
k1_grid, k2_grid = np.asarray(k1_grid), np.asarray(k2_grid)
k1_opt , k2_opt  = np.asarray(k1_opt ), np.asarray(k2_opt )
t_grid_time      = np.asarray(t_grid_time)   # sec per step
t_opt_time       = np.asarray(t_opt_time)

# --------------------------------------------------------------------
# 6 · Plots  ----------------------------------------------------------
# --------------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(t_grid, k1_grid,  label=r'$k_1$ grid')
plt.plot(t_grid, k1_opt,  '--', label=r'$k_1$ ZoOpt')
plt.xlabel('time [s]'); plt.title(r'Comparison of $k_1(t)$'); plt.legend()
plt.tight_layout(); plt.show()

# --- NEW: k2 overlay -------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(t_grid, k2_grid,  label=r'$k_2$ grid')
plt.plot(t_grid, k2_opt,  '--', label=r'$k_2$ ZoOpt')
plt.xlabel('time [s]'); plt.title(r'Comparison of $k_2(t)$'); plt.legend()
plt.tight_layout(); plt.show()

# --- Timing per step -------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(t_grid, 1e3*t_grid_time,  label='grid tuner')
plt.plot(t_grid, 1e3*t_opt_time,  label='ZoOpt tuner')
plt.xlabel('time [s]'); plt.ylabel('time per step [ms]')
plt.title('Tuning cost at each sampling instant'); plt.legend()
plt.tight_layout(); plt.show()

# --- Mean timing bar (kept) -----------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(['grid', 'ZoOpt'],
        [1e3*t_grid_time.mean(), 1e3*t_opt_time.mean()])
plt.ylabel('mean tuning cost per step [ms]')
plt.title('Average tuning cost'); plt.tight_layout(); plt.show()
