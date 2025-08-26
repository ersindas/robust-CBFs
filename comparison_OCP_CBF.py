import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import glob, pathlib, pandas as pd
from scipy.interpolate import interp1d   


def set_figure_defaults():
    plt.rcParams['lines.linewidth']      = 2
    plt.rcParams['lines.markersize']     = 2
    plt.rcParams['axes.linewidth']       = 0.75
    plt.rcParams['axes.labelsize']       = 16
    plt.rcParams['xtick.labelsize']      = 16
    plt.rcParams['ytick.labelsize']      = 16
    plt.rcParams['font.size']            = 16
    plt.rcParams['legend.fontsize']      = 12.7
    plt.rcParams['legend.title_fontsize']= 12.7
    plt.rcParams['legend.handlelength']  = 1.2
    plt.rcParams['xtick.direction']      = 'in'
    plt.rcParams['ytick.direction']      = 'in'
    plt.rcParams['text.usetex']          = True
    plt.rcParams['text.latex.preamble']  = r'\usepackage{amsmath}'

set_figure_defaults()

Kv     = 1.0
Kom    = 2.5
v_ref  = 0.25
c_freq = 2.9*np.pi/6
y_mag  = 1.5
y_shift= -0.35

eps      = 1e-6
wrap_mx  = lambda a: ca.atan2(ca.sin(a), ca.cos(a))

def k_des_casadi(x, t_k):
    x_pos, y_pos, psi = x[0], x[1], x[2]
    x_d   = v_ref * t_k
    y_d   = y_mag*ca.sin(c_freq*x_d) + y_shift
    ex    = x_d - x_pos
    ey    = y_d - y_pos
    dist  = ca.sqrt(ex**2 + ey**2 + eps)
    theta_d = ca.if_else(dist > 1e-3, ca.atan2(ey, ex + eps), psi)
    v_des = ca.if_else(Kv*dist <  v_max, Kv*dist, v_max)
    w_des = -Kom * wrap_mx(psi - theta_d)
    return ca.vertcat(v_des, w_des)

dt = 0.1;  N = 400
v_max, w_max = 2, 2
x_min, x_max = -1.0, 9.0
y_min, y_max = -1.5, 0.8
obstacles = [(2.5, 0.0, 1.0), (6.9, 0.0, 1.0)]
Q = np.diag([2.0, 2.0, 0.5]);   R = np.diag([1.0, 1.0])

delta_x = 0.05
delta_y = 0.1
delta_th = 0.00

wrap_np = lambda a: (a + np.pi) % (2*np.pi) - np.pi
wrap_mx = lambda a: ca.atan2(ca.sin(a), ca.cos(a))

def calculate_Jt(t_arr, x_arr, epsilon_x=0.5):
    """
    Calculates the J_t time optimality metric with time synchronization.
    """
    if len(t_arr) < 2:
        return 0.0

    start_indices = np.where(x_arr[:, 0] > 0.01)[0]
    start_idx = start_indices[0] if len(start_indices) > 0 else 0
    
    t_offset = t_arr[start_idx]
    t_ref_sync = np.maximum(0, t_arr - t_offset)

    pos_reference = np.array([ref_state(ti) for ti in t_ref_sync])[:, :2]
    pos_actual = x_arr[:, :2]

    positional_error = np.linalg.norm(pos_actual - pos_reference, axis=1)

    indicator = (positional_error >= epsilon_x).astype(float)

    Jt = np.trapz(indicator, t_arr)
    return float(Jt)

def ref_state(t):
    v_ref, y_mag, c_freq, y_shift = 0.25, 1.5, 2.9*np.pi/6, -0.35
    x_d = v_ref * t
    y_d = y_mag * np.sin(c_freq * x_d) + y_shift
    th  = np.arctan2(y_mag * c_freq * np.cos(c_freq * x_d), v_ref)
    return np.array([x_d, y_d, th])

def h_eval_xy(x, y):
    vals = [x - x_min, x_max - x, y - y_min, y_max - y]
    for ox, oy, r in obstacles:
        vals.append((x-ox)**2 + (y-oy)**2 - r**2)
    return np.min(vals)

def h_traj(X):
    return np.array([h_eval_xy(X[k,0], X[k,1]) for k in range(X.shape[0])])

x_sym = ca.MX.sym('x', 3)
u_sym = ca.MX.sym('u', 2)
f_sym = ca.Function('f', [x_sym, u_sym],
        [ca.vertcat(x_sym[0] + dt*u_sym[0]*ca.cos(x_sym[2]),
                    x_sym[1] + dt*u_sym[0]*ca.sin(x_sym[2]),
                    wrap_mx(x_sym[2] + dt*u_sym[1]))])

U_guess = np.zeros((2, N));  U_guess[0, :] = 0.25
X_guess = np.zeros((3, N+1)); X_guess[:, 0] = np.array([0.0, -0.35, 0.0])
for k in range(N):
    px, py, th = X_guess[:, k]
    v, w = U_guess[:, k]
    X_guess[0, k+1] = px + dt*v*np.cos(th)
    X_guess[1, k+1] = py + dt*v*np.sin(th)
    X_guess[2, k+1] = wrap_np(th + dt*w)

base_pos_margin = np.hypot(delta_x, delta_y)
abs_v_guess = np.abs(U_guess[0, :])
yaw_inc     = abs_v_guess * (2 * np.sin(delta_th / 2)) * dt
yaw_margin  = np.cumsum(yaw_inc)
step_margin = base_pos_margin + np.append(yaw_margin, yaw_margin[-1])

opti = ca.Opti()
X = opti.variable(3, N+1)
U = opti.variable(2, N)

opti.subject_to(X[:, 0] == X_guess[:, 0])
cost = 0

for k in range(N):
    t_k   = k * dt
    kdes  = k_des_casadi(X[:,k], t_k)
    cost += ca.sumsqr(U[:,k] - kdes)
    opti.subject_to(X[:, k+1] == f_sym(X[:, k], U[:, k]))
    opti.subject_to(opti.bounded(-v_max, U[0, k], v_max))
    opti.subject_to(opti.bounded(-w_max, U[1, k], w_max))
    m = step_margin[k]
    opti.subject_to(X[0, k] >= x_min + m)
    opti.subject_to(X[0, k] <= x_max - m)
    opti.subject_to(X[1, k] >= y_min + m)
    opti.subject_to(X[1, k] <= y_max - m)
    for ox, oy, r in obstacles:
        dx = X[0, k] - ox
        dy = X[1, k] - oy
        opti.subject_to(dx**2 + dy**2 >= (r + m)**2)

mN = step_margin[N]
opti.subject_to(X[0, N] >= x_min + mN)
opti.subject_to(X[0, N] <= x_max + 0 - mN)
opti.subject_to(X[1, N] >= y_min + mN)
opti.subject_to(X[1, N] <= y_max - mN)
for ox, oy, r in obstacles:
    dxN = X[0, N] - ox
    dyN = X[1, N] - oy
    opti.subject_to(dxN**2 + dyN**2 >= (r + mN)**2)

opti.minimize(cost)
opti.set_initial(X, X_guess)
opti.set_initial(U, U_guess)
opti.solver('ipopt', {"ipopt.print_level": 0, "print_time": False})
sol = opti.solve()
X_opt = sol.value(X).T
U_opt = sol.value(U).T
t_u = np.arange(U_opt.shape[0]) * dt

opti2 = ca.Opti()
Xn = opti2.variable(3, N+1)
Un = opti2.variable(2, N)
opti2.subject_to(Xn[:, 0] == X_guess[:, 0])
cost2 = 0
for k in range(N):
    t_k   = k * dt
    kdes  = k_des_casadi(Xn[:,k], t_k)
    cost2 += ca.sumsqr(Un[:,k] - kdes)
    opti2.subject_to(Xn[:, k+1] == f_sym(Xn[:, k], Un[:, k]))
    opti2.subject_to(opti2.bounded(-v_max, Un[0, k], v_max))
    opti2.subject_to(opti2.bounded(-w_max, Un[1, k], w_max))
    opti2.subject_to(Xn[0, k] >= x_min)
    opti2.subject_to(Xn[0, k] <= x_max)
    opti2.subject_to(Xn[1, k] >= y_min)
    opti2.subject_to(Xn[1, k] <= y_max)
    for ox, oy, r in obstacles:
        dx = Xn[0, k] - ox
        dy = Xn[1, k] - oy
        opti2.subject_to(dx**2 + dy**2 >= (r)**2)
opti2.subject_to(Xn[0, N] >= x_min)
opti2.subject_to(Xn[0, N] <= x_max)
opti2.subject_to(Xn[1, N] >= y_min)
opti2.subject_to(Xn[1, N] <= y_max)
for ox, oy, r in obstacles:
    dxN = Xn[0, N] - ox
    dyN = Xn[1, N] - oy
    opti2.subject_to(dxN**2 + dyN**2 >= (r)**2)
opti2.minimize(cost2)
opti2.set_initial(Xn, X_guess)
opti2.set_initial(Un, U_guess)
opti2.solver('ipopt', {"ipopt.print_level": 0, "print_time": False})
sol2 = opti2.solve()
X_opt_nom = sol2.value(Xn).T
U_opt_nom = sol2.value(Un).T
t_u_nom = np.arange(U_opt_nom.shape[0]) * dt

robust_logs  = sorted(glob.glob("data/cbf_log_*_robust.csv")  + glob.glob("data/cbf_log_*_robust.cvs"))
nominal_logs = sorted(glob.glob("data/cbf_log_*_nominal.csv") + glob.glob("data/cbf_log_*_nominal.cvs"))

fixed_logs   = sorted(glob.glob("data/cbf_log_*_fixed_gammas.csv") + glob.glob("data/cbf_log_*_fixed_gammas.cvs"))
tunable_logs = sorted(glob.glob("data/cbf_log_*_tunable_gammas.csv") + glob.glob("data/cbf_log_*_tunable_gammas.cvs"))

if robust_logs:
    log_path = robust_logs[-1]
    cbf_all = np.loadtxt(log_path, delimiter=",", skiprows=1)
    cbf_t   = cbf_all[:, 0]
    cbf_xy  = cbf_all[:, 1:3]
    cbf_uv  = cbf_all[:, 4:6]
    cbf_h   = cbf_all[:, 6]
    cbf_g1  = cbf_all[:, 7]
    cbf_g2  = cbf_all[:, 8]
    nominal_mode = False

    fx_t = fx_xy = fx_uv = fx_h = fx_g1 = fx_g2 = None
    if fixed_logs:
        fx_all = np.loadtxt(fixed_logs[-1], delimiter=",", skiprows=1)
        fx_t, fx_xy = fx_all[:,0], fx_all[:,1:3]
        fx_uv, fx_h = fx_all[:,4:6], fx_all[:,6]
        fx_g1, fx_g2 = fx_all[:,7], fx_all[:,8]

    tn_t = tn_xy = tn_uv = tn_h = tn_g1 = tn_g2 = None
    if tunable_logs:
        tn_all = np.loadtxt(tunable_logs[-1], delimiter=",", skiprows=1)
        tn_t, tn_xy = tn_all[:,0], tn_all[:,1:3]
        tn_uv, tn_h = tn_all[:,4:6], tn_all[:,6]
        tn_g1, tn_g2 = tn_all[:,7], tn_all[:,8]

    tcbf_color = 'tab:purple'
    fcbf_color = 'tab:brown'
    tcbf_ls    = '--'    
    fcbf_ls    = '--'    

    fig_all = plt.figure(figsize=(8.5, 8.5), constrained_layout=True)
    gs = fig_all.add_gridspec(nrows=3, ncols=2, height_ratios=[1.25, 1, 1])
    ax_top = fig_all.add_subplot(gs[0, :])
    ax_v   = fig_all.add_subplot(gs[1, 0])
    ax_h   = fig_all.add_subplot(gs[1, 1])
    ax_w   = fig_all.add_subplot(gs[2, 0])
    ax_g   = fig_all.add_subplot(gs[2, 1])

    ax_top.fill_between([x_min, x_max], y_min, y_max, color='lime', alpha=0.15, label=r'{safe set}', zorder=0)
    ax_top.axhline(y_min, ls='-', color='red', lw=1.5, label=r'{boundary}')
    ax_top.axhline(y_max, ls='-', color='red', lw=1.5)
    ax_top.vlines([x_min, x_max], y_min, y_max, ls='-', color='r', lw=1.5)

    pts   = X_opt[:, :2]
    vseg  = np.diff(pts, axis=0)
    segN  = min(vseg.shape[0], U_opt.shape[0])
    vseg  = vseg[:segN]
    n_hat = np.column_stack([-vseg[:, 1], vseg[:, 0]])
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True) + 1e-12
    yaw_inc    = np.abs(U_opt[:segN, 0]) * (2*np.sin(delta_th/2)) * dt
    yaw_margin = np.cumsum(yaw_inc)
    half_w = (np.abs(n_hat[:, 0]) * delta_x + np.abs(n_hat[:, 1]) * delta_y) + yaw_margin
    offs  = half_w[:, None] * n_hat
    upper = pts[:segN] + offs
    lower = pts[:segN] - offs
    tube  = np.vstack([upper, lower[::-1]])
    ax_top.add_patch(Polygon(tube, closed=True, facecolor='cyan', alpha=0.5, edgecolor='none', label=None, zorder=3))

    for i, (ox, oy, r) in enumerate(obstacles):
        ax_top.add_patch(Circle((ox, oy), r, facecolor='white', edgecolor='none', lw=0, zorder=2))
        lab = r'{obstacle}' if i == 0 else None
        ax_top.add_patch(Circle((ox, oy), r, facecolor='none', edgecolor='red', lw=1.5, zorder=5, label=lab))

    vseg_cbf  = np.diff(cbf_xy, axis=0)
    n_hat_cbf = np.column_stack([-vseg_cbf[:,1], vseg_cbf[:,0]])
    n_hat_cbf /= np.linalg.norm(n_hat_cbf, axis=1, keepdims=True) + 1e-12
    if (cbf_uv is not None) and (cbf_t is not None) and (len(cbf_t) >= 2):
        dt_cbf       = np.diff(cbf_t)
        v_abs_cbf    = np.abs(cbf_uv[:-1, 0])
        yaw_step_cbf = v_abs_cbf * (2*np.sin(delta_th/2)) * dt_cbf
        yaw_cum_cbf  = np.cumsum(yaw_step_cbf)
    else:
        yaw_cum_cbf  = np.zeros(n_hat_cbf.shape[0])
    half_w_cbf = (np.abs(n_hat_cbf[:,0]) * delta_x + np.abs(n_hat_cbf[:,1]) * delta_y) + yaw_cum_cbf
    offs_cbf   = half_w_cbf[:, None] * n_hat_cbf
    upper_cbf  = cbf_xy[:-1] + offs_cbf
    lower_cbf  = cbf_xy[:-1] - offs_cbf
    tube_cbf   = np.vstack([upper_cbf, lower_cbf[::-1]])
    ax_top.add_patch(Polygon(tube_cbf, closed=True, facecolor='grey', alpha=0.5, edgecolor='none', label=None, zorder=3))

    if fx_xy is not None and fx_uv is not None:
        vseg_fx  = np.diff(fx_xy, axis=0)
        n_hat_fx = np.column_stack([-vseg_fx[:,1], vseg_fx[:,0]])
        n_hat_fx /= np.linalg.norm(n_hat_fx, axis=1, keepdims=True) + 1e-12
        if fx_t is not None and len(fx_t) >= 2:
            dt_fx   = np.diff(fx_t)
            v_abs_fx= np.abs(fx_uv[:-1,0])
            yaw_fx  = np.cumsum(v_abs_fx * (2*np.sin(delta_th/2)) * dt_fx)
        else:
            yaw_fx  = np.zeros(n_hat_fx.shape[0])
        half_w_fx = (np.abs(n_hat_fx[:,0])*delta_x + np.abs(n_hat_fx[:,1])*delta_y) + yaw_fx
        offs_fx   = half_w_fx[:,None]*n_hat_fx
        upper_fx  = fx_xy[:-1] + offs_fx
        lower_fx  = fx_xy[:-1] - offs_fx
        tube_fx   = np.vstack([upper_fx, lower_fx[::-1]])
        # ax_top.add_patch(Polygon(tube_fx, closed=True, facecolor=fcbf_color, alpha=0.18, edgecolor='none', label=None, zorder=3))

    if tn_xy is not None and tn_uv is not None:
        vseg_tn  = np.diff(tn_xy, axis=0)
        n_hat_tn = np.column_stack([-vseg_tn[:,1], vseg_tn[:,0]])
        n_hat_tn /= np.linalg.norm(n_hat_tn, axis=1, keepdims=True) + 1e-12
        if tn_t is not None and len(tn_t) >= 2:
            dt_tn   = np.diff(tn_t)
            v_abs_tn= np.abs(tn_uv[:-1,0])
            yaw_tn  = np.cumsum(v_abs_tn * (2*np.sin(delta_th/2)) * dt_tn)
        else:
            yaw_tn  = np.zeros(n_hat_tn.shape[0])
        half_w_tn = (np.abs(n_hat_tn[:,0])*delta_x + np.abs(n_hat_tn[:,1])*delta_y) + yaw_tn
        offs_tn   = half_w_tn[:,None]*n_hat_tn
        upper_tn  = tn_xy[:-1] + offs_tn
        lower_tn  = tn_xy[:-1] - offs_tn
        tube_tn   = np.vstack([upper_tn, lower_tn[::-1]])
        # ax_top.add_patch(Polygon(tube_tn, closed=True, facecolor=tcbf_color, alpha=0.18, edgecolor='none', label=None, zorder=3))

    t_grid = np.arange(0, (N+1)*dt, dt)
    ref = np.array([ref_state(t) for t in t_grid])
    ax_top.plot(ref[:,0], ref[:,1], 'm', label=r'{reference}', zorder=4)
    ax_top.plot(X_opt[:,0], X_opt[:,1], 'b', label=r'{R-COCP}', zorder=4)
    ax_top.plot(cbf_xy[:,0], cbf_xy[:,1], 'k', lw=2, label=r'{R-CBF-QP}', zorder=4)
    if fx_xy is not None:
        ax_top.plot(fx_xy[:,0], fx_xy[:,1], color=fcbf_color, lw=2, alpha=0.7, linestyle=fcbf_ls, label=r'{R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)}', zorder=4)
    if tn_xy is not None:
        ax_top.plot(tn_xy[:,0], tn_xy[:,1], color=tcbf_color, lw=2, alpha=0.7, linestyle=tcbf_ls, label=r'{R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)}', zorder=4)
    ax_top.set_xlabel(r'$x~[\text{m}]$'); ax_top.set_ylabel(r'$y~[\text{m}]$')
    ax_top.set_xlim(x_min-0.1, x_max+0.1); ax_top.set_ylim(y_min-0.5, y_max+0.5)
    ax_top.set_xticks([-1, 0, 2, 4, 6, 8, 9]); ax_top.set_aspect('equal'); ax_top.grid(False)

    handles, labels = ax_top.get_legend_handles_labels()
    unique = {l: h for h, l in zip(handles, labels)}
    unique[r'{obstacle}'] = Line2D([0], [0], linestyle='None', marker='o', markersize=9, markeredgewidth=2, markeredgecolor='red', markerfacecolor='none')
    order = [r'{safe set}', r'{R-CBF-QP}', r'{boundary}', r'{R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)}', r'{obstacle}', r'{R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)}',   r'{reference}', r'{R-COCP}']
    order = [o for o in order if o in unique]
    ax_top.legend([unique[o] for o in order], order, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.47, 1.27), bbox_transform=ax_top.transAxes)

    ax_v.plot(cbf_t[:cbf_uv.shape[0]], cbf_uv[:, 0], 'k', label='R-CBF-QP')
    if fx_uv is not None:
        ax_v.plot(fx_t[:fx_uv.shape[0]], fx_uv[:, 0], color=fcbf_color, alpha=0.7, linestyle=fcbf_ls, label='R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)')
    if tn_uv is not None:
        ax_v.plot(tn_t[:tn_uv.shape[0]], tn_uv[:, 0], color=tcbf_color, alpha=0.7, linestyle=tcbf_ls, label='R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)')
    ax_v.plot(t_u, U_opt[:, 0], 'b', label='R-COCP')
    ax_v.set_ylabel(r'$v~[\text{m/s}]$')
    ax_v.set_xlim(0, t_u[-1]+0.1); ax_v.set_ylim(-0.5, 2.0)
    ax_v.margins(x=0, y=0)
    hdl, lbl = ax_v.get_legend_handles_labels()
    remap = {l:h for h,l in zip(hdl,lbl)}
    ord_v = ['R-CBF-QP','R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)','R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)','R-COCP']
    ord_v = [o for o in ord_v if o in remap]
    # ax_v.legend([remap[o] for o in ord_v], ord_v, frameon=False, loc='upper right', ncol=2)

    ax_w.plot(cbf_t[:cbf_uv.shape[0]], cbf_uv[:, 1], 'k', label='R-CBF-QP')
    if fx_uv is not None:
        ax_w.plot(fx_t[:fx_uv.shape[0]], fx_uv[:, 1], color=fcbf_color, alpha=0.7, linestyle=fcbf_ls, label='R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)')
    if tn_uv is not None:
        ax_w.plot(tn_t[:tn_uv.shape[0]], tn_uv[:, 1], color=tcbf_color, alpha=0.7, linestyle=tcbf_ls, label='R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)')
    ax_w.plot(t_u, U_opt[:, 1], 'b', label='OCP')
    ax_w.set_ylabel(r'$\omega~[\text{rad/s}]$'); ax_w.set_xlabel(r'$t~[\text{s}]$')
    ax_w.set_xlim(0, t_u[-1]+0.1); ax_w.set_ylim(-1.5, 2.0)
    ax_w.margins(x=0, y=0)
    hdl, lbl = ax_w.get_legend_handles_labels()
    remap = {l:h for h,l in zip(hdl,lbl)}
    ord_w = ['R-CBF-QP','R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)','R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)','OCP']
    ord_w = [o for o in ord_w if o in remap]
    # ax_w.legend([remap[o] for o in ord_w], ord_w, frameon=False, loc='upper right', ncol=2)

    ax_h.plot(cbf_t, cbf_h, 'k', label=r'R-CBF-QP')
    if fx_h is not None:
        ax_h.plot(fx_t, fx_h, color=fcbf_color, linestyle=fcbf_ls, label='R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)')
    if tn_h is not None:
        ax_h.plot(tn_t, tn_h, color=tcbf_color, linestyle=tcbf_ls, label='R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)')
    ax_h.plot(t_grid, h_traj(X_opt), 'b', label='OCP')
    ax_h.axhline(0.0, linestyle='--', linewidth=1)
    ax_h.set_ylabel(r'$h$'); ax_h.set_xlim(cbf_t[0], cbf_t[-1])
    ax_h.margins(x=0, y=0)
    hdl, lbl = ax_h.get_legend_handles_labels()
    remap = {l:h for h,l in zip(hdl,lbl)}
    ord_h = ['CBF','R-CBF-QP (${\gamma_1 \!=\! 1.4, \gamma_2 \!=\! 0.3}$)','R-CBF-QP (${\gamma_1(h), \gamma_2(h)}$)','OCP']
    ord_h = [o for o in ord_h if o in remap]
    # ax_h.legend([remap[o] for o in ord_h], ord_h, frameon=False, loc='upper right', ncol=2)

    ax_g.plot(cbf_t, cbf_g1, 'r', label=r'$\Bar{\gamma}_1(\mathbf{\hat{x}})$')
    ax_g.plot(cbf_t, cbf_g2, 'g', label=r'$\Bar{\gamma}_2(\mathbf{\hat{x}})$')
    # if fx_g1 is not None:
    #     ax_g.plot(fx_t, fx_g1, color='r', linestyle=fcbf_ls, alpha=0.9, label=None)
    # if fx_g2 is not None:
    #     ax_g.plot(fx_t, fx_g2, color='g', linestyle=fcbf_ls, alpha=0.9, label=None)
    if tn_g1 is not None:
        ax_g.plot(tn_t, tn_g1, color='k', linestyle='--', alpha=0.9, label=r'$\gamma_1(h)$')
    if tn_g2 is not None:
        ax_g.plot(tn_t, tn_g2, color='b', linestyle='--', alpha=0.9, label=r'$\gamma_2(h)$')
    ax_g.set_ylabel(r'$\gamma$'); ax_g.set_xlabel(r'$t~[\text{s}]$')
    ax_g.set_xlim(cbf_t[0], cbf_t[-1]); ax_g.margins(x=0, y=0)
    # ax_g.legend(frameon=False, loc='upper right', ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.27), bbox_transform=ax_top.transAxes)
    # ax_g.margins(x=0, y=0); ax_g.legend(frameon=False, loc='upper right', ncol=1, bbox_to_anchor=(1.06, 1.06))
    ax_g.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_g.set_ylabel(r'$\gamma$'); ax_g.set_xlabel(r'$t~[\text{s}]$')
    ax_g.set_xlim(cbf_t[0], cbf_t[-1]); ax_g.margins(x=0, y=0)

    ax_g.legend(
        frameon=False, loc='upper right', ncol=1,
        bbox_to_anchor=(1.01, 1.02),   
        handlelength=1.0,              
        handletextpad=0.2,            
        labelspacing=0.2,             
        columnspacing=0.5,             
        borderpad=0.2,                 
        borderaxespad=0.03             
    )

    ax_g.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plt.savefig('Figure_combined_ocp_cbf_robust.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_combined_ocp_cbf_robust.pdf', format='pdf')

    plt.show()

if nominal_logs:
    log_path = nominal_logs[-1]
    cbf_all = np.loadtxt(log_path, delimiter=",", skiprows=1)
    cbf_t_n   = cbf_all[:, 0]
    cbf_xy  = cbf_all[:, 1:3]
    cbf_uv_n  = cbf_all[:, 4:6]
    cbf_h   = cbf_all[:, 6]
    cbf_g1  = cbf_all[:, 7]
    cbf_g2  = cbf_all[:, 8]
    nominal_mode = True

    nodrd_logs = sorted(glob.glob("data/cbf_log_*_noDRD.csv") + glob.glob("data/cbf_log_*_noDRD.cvs"))
    cbf2_t = cbf2_xy = cbf2_uv = cbf2_h = None
    if nodrd_logs:
        cbf2_all = np.loadtxt(nodrd_logs[-1], delimiter=",", skiprows=1)
        cbf2_t   = cbf2_all[:, 0]
        cbf2_xy  = cbf2_all[:, 1:3]
        cbf2_uv  = cbf2_all[:, 4:6]
        cbf2_h   = cbf2_all[:, 6]

    X_kd = np.empty_like(X_opt_nom)
    U_kd = np.empty_like(U_opt_nom)
    X_kd[0] = X_opt_nom[0]
    for k in range(N):
        t_k = k * dt
        u_des = np.array(k_des_casadi(ca.DM(X_kd[k]), t_k).full()).ravel()
        v_k  = np.clip(u_des[0], -v_max, v_max)
        w_k  = np.clip(u_des[1], -w_max, w_max)
        U_kd[k] = [v_k, w_k]
        th = X_kd[k, 2]
        X_kd[k+1, 0] = X_kd[k, 0] + dt * v_k * np.cos(th)
        X_kd[k+1, 1] = X_kd[k, 1] + dt * v_k * np.sin(th)
        X_kd[k+1, 2] = wrap_np(th + dt * w_k)

    kd_color = 'tab:orange'
    nodrd_color = 'tab:brown'

    fig_all = plt.figure(figsize=(8.5, 8.5), constrained_layout=True)
    gs = fig_all.add_gridspec(nrows=3, ncols=2, height_ratios=[1.25, 1, 1])
    ax_top = fig_all.add_subplot(gs[0, :])
    ax_v   = fig_all.add_subplot(gs[1, 0])
    ax_h   = fig_all.add_subplot(gs[1, 1])
    ax_w   = fig_all.add_subplot(gs[2, 0])
    ax_g   = fig_all.add_subplot(gs[2, 1])

    ax_top.fill_between([x_min, x_max], y_min, y_max, color='lime', alpha=0.15, label=r'{safe set}', zorder=0)
    ax_top.axhline(y_min, ls='-', color='red', lw=1.5, label=r'{boundary}')
    ax_top.axhline(y_max, ls='-', color='red', lw=1.5)
    ax_top.vlines([x_min, x_max], y_min, y_max, ls='-', color='r', lw=1.5)

    pts   = X_opt_nom[:, :2]
    vseg  = np.diff(pts, axis=0)
    segN  = min(vseg.shape[0], U_opt_nom.shape[0])
    vseg  = vseg[:segN]
    n_hat = np.column_stack([-vseg[:, 1], vseg[:, 0]])
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True) + 1e-12
    yaw_inc    = np.abs(U_opt_nom[:segN, 0]) * (2*np.sin(delta_th/2)) * dt
    yaw_margin = np.cumsum(yaw_inc)
    half_w = (np.abs(n_hat[:, 0]) * delta_x + np.abs(n_hat[:, 1]) * delta_y) + yaw_margin
    offs  = half_w[:, None] * n_hat
    upper = pts[:segN] + offs
    lower = pts[:segN] - offs
    tube  = np.vstack([upper, lower[::-1]])
    # if not nominal_mode:
    #     ax_top.add_patch(Polygon(tube, closed=True, facecolor='grey', alpha=0.5, edgecolor='none', label=r'{uncertainty tube, OCP}', zorder=3))

    for i, (ox, oy, r) in enumerate(obstacles):
        ax_top.add_patch(Circle((ox, oy), r, facecolor='white', edgecolor='none', lw=0, zorder=2))
        lab = r'{obstacle}' if i == 0 else None
        ax_top.add_patch(Circle((ox, oy), r, facecolor='none', edgecolor='red', lw=1.5, zorder=5, label=lab))

    vseg_cbf  = np.diff(cbf_xy, axis=0)
    n_hat_cbf = np.column_stack([-vseg_cbf[:,1], vseg_cbf[:,0]])
    n_hat_cbf /= np.linalg.norm(n_hat_cbf, axis=1, keepdims=True) + 1e-12
    if (cbf_uv_n is not None) and (cbf_t_n is not None) and (len(cbf_t_n) >= 2):
        dt_cbf       = np.diff(cbf_t_n)
        v_abs_cbf    = np.abs(cbf_uv_n[:-1, 0])
        yaw_step_cbf = v_abs_cbf * (2*np.sin(delta_th/2)) * dt_cbf
        yaw_cum_cbf  = np.cumsum(yaw_step_cbf)
    else:
        yaw_cum_cbf  = np.zeros(n_hat_cbf.shape[0])
    half_w_cbf = (np.abs(n_hat_cbf[:,0]) * delta_x + np.abs(n_hat_cbf[:,1]) * delta_y) + yaw_cum_cbf
    offs_cbf   = half_w_cbf[:, None] * n_hat_cbf
    upper_cbf  = cbf_xy[:-1] + offs_cbf
    lower_cbf  = cbf_xy[:-1] - offs_cbf
    tube_cbf   = np.vstack([upper_cbf, lower_cbf[::-1]])
    # if not nominal_mode:
    #     ax_top.add_patch(Polygon(tube_cbf, closed=True, facecolor='cyan', alpha=0.3, edgecolor='none', label=r'{uncertainty tube, CBF}', zorder=3))

    t_grid = np.arange(0, (N+1)*dt, dt)
    ref = np.array([ref_state(t) for t in t_grid])
    ax_top.plot(ref[:,0], ref[:,1], 'm', label=r'{reference}', zorder=4)
    ax_top.plot(X_opt_nom[:,0], X_opt_nom[:,1], 'b', label=r'{COCP}', zorder=4)
    ax_top.plot(cbf_xy[:,0], cbf_xy[:,1], 'k', lw=2, label=r'{R-CBF-QP}', zorder=4)
    if cbf2_xy is not None:
        ax_top.plot(cbf2_xy[:,0], cbf2_xy[:,1], color=nodrd_color, label=r'{CBF-QP}', zorder=4)
        ax_top.plot(cbf2_xy[-1, 0], cbf2_xy[-1, 1], marker='o', linestyle='None',markersize=10, markerfacecolor=nodrd_color,
            markeredgecolor=nodrd_color, markeredgewidth=1.5)
    ax_top.plot(X_kd[:,0], X_kd[:,1], color=kd_color, linestyle='--', label=r'{$\mathbf{k_d}$}', zorder=4)

    ax_top.set_xlabel(r'$x~[\text{m}]$'); ax_top.set_ylabel(r'$y~[\text{m}]$')
    ax_top.set_xlim(x_min-0.1, x_max+0.1); ax_top.set_ylim(y_min-0.5, y_max+0.5)
    ax_top.set_xticks([-1, 0, 2, 4, 6, 8, 9]); ax_top.set_aspect('equal'); ax_top.grid(False)

    handles, labels = ax_top.get_legend_handles_labels()
    unique = {l: h for h, l in zip(handles, labels)}
    unique[r'{obstacle}'] = Line2D([0], [0], linestyle='None', marker='o', markersize=9, markeredgewidth=2, markeredgecolor='red', markerfacecolor='none')
    order = [r'{safe set}', r'{R-CBF-QP}', r'{boundary}', r'{COCP}',  r'{obstacle}', r'{CBF-QP}',  r'{reference}', r'{$\mathbf{k_d}$}']
    order = [o for o in order if o in unique]
    ax_top.legend([unique[o] for o in order], order, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.27), bbox_transform=ax_top.transAxes)

    ax_v.plot(t_u_nom, U_opt_nom[:, 0], 'b', label='COCP')
    ax_v.plot(cbf_t_n[:cbf_uv_n.shape[0]], cbf_uv_n[:, 0], 'k', label='R-CBF-QP')
    if cbf2_uv is not None:
        ax_v.plot(cbf2_t[:cbf2_uv.shape[0]], cbf2_uv[:, 0], color=nodrd_color, label='CBF-QP')
    ax_v.plot(t_u_nom, U_kd[:, 0], color=kd_color, linestyle='--', label=r'$\mathbf{k_d}$')
    ax_v.set_ylabel(r'$v~[\text{m/s}]$')
    ax_v.set_xlim(0, t_u_nom[-1]+0.1); ax_v.set_ylim(-0.5, 2.0)
    # ax_v.margins(x=0, y=0); ax_v.legend(frameon=False, loc='upper right', ncol=2, bbox_to_anchor=(1.00, 1.04), 
    #      bbox_transform=ax_v.transAxes, borderaxespad=0.0)

    ax_w.plot(t_u_nom, U_opt_nom[:, 1], 'b', label='COCP')
    ax_w.plot(cbf_t_n[:cbf_uv_n.shape[0]], cbf_uv_n[:, 1], 'k', label='R-CBF-QP')
    if cbf2_uv is not None:
        ax_w.plot(cbf2_t[:cbf2_uv.shape[0]], cbf2_uv[:, 1], color=nodrd_color, label='CBF-QP')
    ax_w.plot(t_u_nom, U_kd[:, 1], color=kd_color, linestyle='--', label=r'$\mathbf{k_d}$')
    ax_w.set_ylabel(r'$\omega~[\text{rad/s}]$'); ax_w.set_xlabel(r'$t~[\text{s}]$')
    ax_w.set_xlim(0, t_u_nom[-1]+0.1); ax_w.set_ylim(-1.5, 2.0)
    # ax_w.margins(x=0, y=0); ax_w.legend(frameon=False, loc='upper right', ncol=2,
    #         bbox_to_anchor=(1.00, 1.04), bbox_transform=ax_w.transAxes, borderaxespad=0.0)

    ax_h.plot(t_grid, h_traj(X_opt_nom), 'b', label='COCP')
    ax_h.plot(cbf_t_n, cbf_h, 'k', label=r'R-CBF-QP')
    if cbf2_h is not None:
        ax_h.plot(cbf2_t, cbf2_h, color=nodrd_color, label=r'CBF-QP')
    ax_h.axhline(0.0, linestyle='--', linewidth=1)
    ax_h.set_ylabel(r'$h$'); ax_h.set_xlim(cbf_t_n[0], cbf_t_n[-1])
    # ax_h.margins(x=0, y=0); ax_h.legend(frameon=False, loc='upper right', ncol=1, bbox_to_anchor=(0.88, 1.05))

    ax_g.plot(cbf_t_n, np.round(cbf_g1, 1), 'r', label=r'$\Bar{\gamma}_1(\mathbf{\hat{x}})$')
    ax_g.plot(cbf_t_n, np.round(cbf_g2, 1), 'g', linestyle='--', label=r'$\Bar{\gamma}_2(\mathbf{\hat{x}})$')
    ax_g.set_ylabel(r'$\gamma$'); ax_g.set_xlabel(r'$t~[\text{s}]$')
    ax_g.set_xlim(cbf_t_n[0], cbf_t_n[-1]); ax_g.margins(x=0, y=0)
    ax_g.legend(frameon=False, loc='upper right')
    ax_g.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plt.savefig('Figure_combined_ocp_cbf_nominal.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Figure_combined_ocp_cbf_nominal.pdf', format='pdf')

    plt.show()


# ===========================================================

Q_u = np.diag([1.0, 1.0])         
R_x = np.diag([1.0, 1.0, 1])     

t_grid_rob = np.arange(X_opt.shape[0])      * dt   # robust OCP
t_grid_nom = np.arange(X_opt_nom.shape[0])  * dt   # nominal OCP

x_opt_fun_rob = interp1d(t_grid_rob, X_opt,      axis=0, kind='linear',
                         assume_sorted=True, fill_value="extrapolate")
u_opt_fun_rob = interp1d(t_u,         U_opt,     axis=0, kind='linear',
                         assume_sorted=True, fill_value="extrapolate")
x_opt_fun_nom = interp1d(t_grid_nom,  X_opt_nom, axis=0, kind='linear',
                         assume_sorted=True, fill_value="extrapolate")
u_opt_fun_nom = interp1d(t_u_nom,     U_opt_nom, axis=0, kind='linear',
                         assume_sorted=True, fill_value="extrapolate")

def J_track(t, x, u, baseline='robust', R=R_x, Q=Q_u):
    if baseline == 'robust':
        xref = x_opt_fun_rob(t)
        uref = u_opt_fun_rob(t)
    else:
        xref = x_opt_fun_nom(t)
        uref = u_opt_fun_nom(t)

    dx = x - xref
    du = u - uref

    state_term = np.einsum('ij,jk,ik->i', dx, R, dx)
    input_term = np.einsum('ij,jk,ik->i', du, Q, du)

    integrand = t * state_term + input_term
    return float(np.trapz(integrand, t))

def unpack(log_array):
    """From a 9-column *.csv* row-array →  t, X(3), U(2)."""
    return log_array[:,0], log_array[:,1:4], log_array[:,4:6]

# robust CBF
if robust_logs:
    t_cbf,  X_cbf,  U_cbf  = unpack(np.loadtxt(robust_logs[-1],  delimiter=',', skiprows=1))
    track_costs = {'CBF (robust)':     J_track(t_cbf, X_cbf, U_cbf, 'robust')}

    if fixed_logs:
        t_fx,  X_fx,  U_fx  = unpack(np.loadtxt(fixed_logs[-1],   delimiter=',', skiprows=1))
        track_costs['Fixed CBF']      = J_track(t_fx, X_fx, U_fx, 'robust')

    if tunable_logs:
        t_tn,  X_tn,  U_tn  = unpack(np.loadtxt(tunable_logs[-1], delimiter=',', skiprows=1))
        track_costs['Tunable CBF']    = J_track(t_tn, X_tn, U_tn, 'robust')

    track_costs['OCP (robust)']         = J_track(t_u, X_opt[:-1], U_opt, 'robust')

# nominal CBF
if nominal_logs:
    t_cbf_n, X_cbf_n, U_cbf_n = unpack(np.loadtxt(nominal_logs[-1], delimiter=',', skiprows=1))
    track_costs['CBF (nom)']           = J_track(t_cbf_n, X_cbf_n, U_cbf_n, 'nominal')

    if nodrd_logs:
        t_nd, X_nd, U_nd = unpack(np.loadtxt(nodrd_logs[-1], delimiter=',', skiprows=1))
        track_costs['CBF w/o DRD']      = J_track(t_nd, X_nd, U_nd, 'nominal')

    track_costs['OCP (nom)']           = J_track(t_u_nom, X_opt_nom[:-1], U_opt_nom, 'nominal')

# 
print("\n------- state + input tracking cost J -------------")
for key, val in track_costs.items():
    print(f"{key:<15s}:  {val:8.3f}")
print("----------------------------------------------------\n")


# J_t 
Jt_costs = {}
epsilon_x_val = 0.5 

# Robust -
if robust_logs:
    t_rcocp = np.arange(X_opt.shape[0]) * dt
    Jt_costs['R-COCP'] = calculate_Jt(t_rcocp, X_opt, epsilon_x_val)
    
    Jt_costs['R-CBF-QP (online)'] = calculate_Jt(t_cbf, X_cbf, epsilon_x_val)
    if fixed_logs:
        Jt_costs['R-CBF-QP (fixed)'] = calculate_Jt(t_fx, X_fx, epsilon_x_val)
    if tunable_logs:
        Jt_costs['R-CBF-QP (tunable)'] = calculate_Jt(t_tn, X_tn, epsilon_x_val)

# Nominal
if nominal_logs:
    t_cocp_nom = np.arange(X_opt_nom.shape[0]) * dt
    Jt_costs['COCP'] = calculate_Jt(t_cocp_nom, X_opt_nom, epsilon_x_val)
    
    Jt_costs['R-CBF-QP (nominal)'] = calculate_Jt(t_cbf_n, X_cbf_n, epsilon_x_val)
    if nodrd_logs:
        Jt_costs['CBF-QP'] = calculate_Jt(t_nd, X_nd, epsilon_x_val)

    t_kd = np.arange(X_kd.shape[0]) * dt
    Jt_costs['k_d'] = calculate_Jt(t_kd, X_kd, epsilon_x_val)


if Jt_costs:
    print(f"\n------- J_t time optimality cost (epsilon = {epsilon_x_val} m) -------")
    sorted_keys = [
        'k_d', 'COCP', 'R-COCP', 'CBF-QP', 'R-CBF-QP (nominal)',
        'R-CBF-QP (online)', 'R-CBF-QP (fixed)', 'R-CBF-QP (tunable)'
    ]
    valid_sorted_keys = [key for key in sorted_keys if key in Jt_costs]

    for key in valid_sorted_keys:
        val = Jt_costs[key]
        print(f"{key:<22s}:  {val:8.3f} s")
    print("----------------------------------------------------\n")