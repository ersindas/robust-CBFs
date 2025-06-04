"""
compare_hdot_table.py
---------------------------------------------------------------
Compare ẋh from JAX automatic differentiation with a SymPy
closed-form expression, for

    h(x,y,θ,t) = h₀(x,y) − a₀ V(x,y,θ,t)

Two modes:
    MODE = "analytic"  → both branches differentiate the *exact* h₀.
    MODE = "table"     → both differentiate the bilinear look-up table.

Requires:  pip install "jax[cpu]" sympy matplotlib
---------------------------------------------------------------
"""
MODE = "analytic"          # <- set to "table" for table-only test
# ---------------------------------------------------------------
import jax, jax.numpy as jnp
import jax.scipy.ndimage as jsp_nd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------- CBF/constants -------------------------------------
a0, Kv, alpha, alpha_q = 1.0, 2.0, 3.0, 4.0

# ---------- reference trajectory ------------------------------
x_ref_j = lambda t: 0.6 * t
y_ref_j = lambda t: 0.3 * jnp.sin(0.8 * t)
x_ref_np= lambda t: 0.6 * t
y_ref_np= lambda t: 0.3 * np.sin(0.8 * t)

# ---------- continuous (exact) h₀(x,y) ------------------------
def h0_cont_j(xy):
    x, y = xy
    return 1.0 - (y - 0.2*jnp.sin(3.0*x))**2

def h0_cont_np(x, y):
    return 1.0 - (y - 0.2*np.sin(3.0*x))**2

# ---------- build a coarse lookup table -----------------------
Nx, Ny = 101, 121
x_min, x_max = -2.0,  2.0
y_min, y_max = -1.5,  1.5
x_grid = jnp.linspace(x_min, x_max, Nx)
y_grid = jnp.linspace(y_min, y_max, Ny)
Xg, Yg = jnp.meshgrid(x_grid, y_grid, indexing="xy")
h0_table = h0_cont_j((Xg, Yg))          # sample exact function

def h0_interp(xy):
    """Bilinear interpolation of the table at (x,y)."""
    x, y = xy
    cx = (x - x_min) * (Nx - 1) / (x_max - x_min)
    cy = (y - y_min) * (Ny - 1) / (y_max - y_min)
    coords = jnp.array([[cy], [cx]])    # shape (2,1)
    return jsp_nd.map_coordinates(h0_table, coords,
                                  order=1, mode="nearest")[0]

# ---------- choose which h₀ each branch will use --------------
if MODE == "analytic":
    h0_j, h0_np = h0_cont_j, h0_cont_np
else:                              # MODE == "table"
    h0_j = h0_interp
    h0_np = lambda x, y: np.asarray(h0_interp(jnp.array([x, y])))
grad_h0_j = jax.grad(lambda xy: h0_j(xy))

# ---------- V(x,y,θ,t) ----------------------------------------
def V_fn(x, y, th, t):
    xc =  0.5*jnp.sin(0.5*t)
    yc = -0.2*jnp.cos(0.3*t)
    r  = 0.8
    dist2 = (x-xc)**2 + (y-yc)**2
    return 1.0 - jnp.cos(th) * jnp.exp(-(dist2)/(r**2))

# ---------- barrier & JAX automatic derivative ----------------
def barrier_j(state, t):
    x, y, th = state
    return h0_j(jnp.array([x, y])) - a0*V_fn(x, y, th, t)

def f_uni_j(state, u):
    x, y, th = state
    v, om    = u
    return jnp.array([v*jnp.cos(th), v*jnp.sin(th), om])

@jax.jit
def hdot_j(state, u, t):
    dhdx = jax.grad(barrier_j, argnums=0)(state, t)   # ∇_state h
    dhdt = jax.grad(barrier_j, argnums=1)(state, t)   # explicit ∂/∂t
    return jnp.dot(dhdx, f_uni_j(state, u)) + dhdt

# ---------- SymPy analytic  ḣ ---------------------------------
t = sp.symbols('t')
x, y, th = sp.symbols('x y th')
v_sym, om_sym = sp.symbols('v om')

if MODE == "analytic":
    h0_sym = 1 - (y - 0.2*sp.sin(3*x))**2
    custom_dict = {}
else:         # MODE == "table": use same bilinear table in NumPy
    h_tab = sp.Function('h_tab')
    h0_sym = h_tab(x, y)
    custom_dict = {'h_tab': (lambda xx,yy: h0_np(xx, yy))}

hx, hy = sp.diff(h0_sym, x), sp.diff(h0_sym, y)
x_ref = 0.6*t
y_ref = 0.3*sp.sin(0.8*t)
vp_x = Kv*(x_ref - x); vp_y = Kv*(y_ref - y)
a_expr = hx*vp_x + hy*vp_y + alpha*h0_sym
b_expr = hx**2 + hy**2
lam = (-a_expr + sp.sqrt(a_expr**2 + alpha_q*b_expr**2))/(2*b_expr)
vs_x = vp_x + lam*hx; vs_y = vp_y + lam*hy
ths  = sp.atan2(vs_y, vs_x)

V_sp = 1 - sp.cos(th) * sp.exp(-((x-0.5*sp.sin(0.5*t))**2 +
                                 (y+0.2*sp.cos(0.3*t))**2)/0.64)
h_sym = h0_sym - a0*V_sp
hdot_sym = (sp.diff(h_sym, t) +
            sp.diff(h_sym, x)*v_sym*sp.cos(th) +
            sp.diff(h_sym, y)*v_sym*sp.sin(th) +
            sp.diff(h_sym, th)*om_sym)
hdot_np = sp.lambdify((x,y,th,v_sym,om_sym,t),
                      hdot_sym, modules=['numpy', custom_dict])

# ---------- trajectory and evaluation -------------------------
N  = 200
ts = np.linspace(0, 4.0, N)
rng = np.random.default_rng(3)
states = np.column_stack(( x_ref_np(ts)+0.05*rng.standard_normal(N),
                           y_ref_np(ts)+0.05*rng.standard_normal(N),
                           rng.uniform(-np.pi, np.pi, N) ))
controls = np.column_stack(( 0.4+0.1*rng.standard_normal(N),
                            -0.2+0.1*rng.standard_normal(N) ))

hd_auto = np.array([hdot_j(jnp.array(st), jnp.array(u), tau)
                    for st,u,tau in zip(states, controls, ts)], dtype=float)
hd_sym  = np.array([hdot_np(*st, *u, tau)
                    for st,u,tau in zip(states, controls, ts)], dtype=float)

# ---------- plots ---------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(ts, hd_sym,  lw=2, label="SymPy")
plt.plot(ts, hd_auto,'--',lw=2, label="JAX autodiff")
plt.xlabel("time [s]"); plt.ylabel("ḣ")
plt.title(f"Derivative comparison   (MODE = '{MODE}')")
plt.legend()

plt.figure(figsize=(8,2.2))
plt.plot(ts, np.abs(hd_sym - hd_auto))
plt.xlabel("time [s]"); plt.ylabel("|Δḣ|")
plt.title("absolute error   max = {:.2e}"
          .format(np.max(np.abs(hd_sym - hd_auto))))
plt.tight_layout(); plt.show()
