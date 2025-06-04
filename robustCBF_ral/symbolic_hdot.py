import sympy as sp

# Symbols and functions
x, y, theta, t = sp.symbols('x y theta t')
a0, alpha, alpha_q, K_v = sp.symbols('a0 alpha alpha_q K_v', positive=True)
h0 = sp.Function('h0')(x, y)

# First- and second-order derivatives
h0_x = sp.diff(h0, x)
h0_y = sp.diff(h0, y)
h0_xx = sp.diff(h0_x, x)
h0_xy = sp.diff(h0_x, y)
h0_yy = sp.diff(h0_y, y)

grad_h0 = sp.Matrix([h0_x, h0_y])
hessian_h0 = sp.Matrix([[h0_xx, h0_xy], [h0_xy, h0_yy]])

# Reference signals
x_d = sp.Function('x_d')(t)
y_d = sp.Function('y_d')(t)
x_d_dot = sp.diff(x_d, t)
y_d_dot = sp.diff(y_d, t)

# Proportional controller
r_e = sp.Matrix([x_d - x, y_d - y])
v_p = K_v * r_e

# Filter terms
a = grad_h0.dot(v_p) + alpha * h0
b = grad_h0.dot(grad_h0)
# q = alpha_q * b
lambda_sym = (-a + sp.sqrt(a**2 + alpha_q * b**2)) / (2 * b)

# Filtered velocity and heading
v_s = v_p + lambda_sym * grad_h0
v_sx, v_sy = v_s[0], v_s[1]
theta_s = sp.atan2(v_sy, v_sx)
V = 1 - sp.cos(theta - theta_s)
h = h0 - a0 * V

# Compute partial derivatives
dh_dx = sp.diff(h, x).doit()
dh_dy = sp.diff(h, y).doit()
dh_dtheta = sp.diff(h, theta)
dh_dt = sp.diff(h, t).doit()

# Simplify using common subexpression elimination (CSE)
repl, [dh_dx_s, dh_dy_s, dh_dtheta_s, dh_dt_s] = sp.cse(
    [dh_dx, dh_dy, dh_dtheta, dh_dt],
    symbols=sp.numbered_symbols('z')
)

# Display replacements and simplified expressions
print("=== Common Subexpressions ===")
for i, (var, expr) in enumerate(repl):
    print(f"{var} = {expr}")

print("\n=== Simplified Partial Derivatives ===")
print("dh/dx =", dh_dx_s)
print("dh/dy =", dh_dy_s)
print("dh/dtheta =", dh_dtheta_s)
print("dh/dt =", dh_dt_s)
