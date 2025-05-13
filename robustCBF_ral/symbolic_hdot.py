import sympy as sp

t = sp.symbols('t', real=True)
x = sp.Function('x')(t)
y = sp.Function('y')(t)
theta = sp.Function('theta')(t)
x_d = sp.Function('x_d')(t)
y_d = sp.Function('y_d')(t)
v = sp.Function('v')(t)
omega = sp.Function('omega')(t)
a0, Kv, alpha, alpha_q = sp.symbols('a0 Kv alpha alpha_q', positive=True)

h0 = sp.Function('h0')(x, y)
h0_x = sp.Derivative(h0, x)
h0_y = sp.Derivative(h0, y)

v_p_x = Kv * (x_d - x)
v_p_y = Kv * (y_d - y)

a_expr = h0_x * v_p_x + h0_y * v_p_y + alpha * h0
b_expr = h0_x**2 + h0_y**2
lambda_expr = (-a_expr + sp.sqrt(a_expr**2 + alpha_q * b_expr**2)) / (2 * b_expr)

v_s_x = v_p_x + lambda_expr * h0_x
v_s_y = v_p_y + lambda_expr * h0_y
theta_s = sp.atan2(v_s_y, v_s_x)

h_expr = h0 - a0 * (1 - sp.cos(theta - theta_s))
dh_dt = sp.diff(h_expr, t)
dh_dt_unicycle = dh_dt.subs({
    sp.diff(x, t): v * sp.cos(theta),
    sp.diff(y, t): v * sp.sin(theta),
    sp.diff(theta, t): omega
})

hx, hy = sp.symbols('h_x h_y')
hxx, hxy, hyy = sp.symbols('h_xx h_xy h_yy')

repl_map = {
    sp.Derivative(h0, x): hx,
    sp.Derivative(h0, y): hy,
    sp.Derivative(h0, x, x): hxx,
    sp.Derivative(h0, x, y): hxy,
    sp.Derivative(h0, y, x): hxy,
    sp.Derivative(h0, y, y): hyy,
}

expr_short = dh_dt_unicycle.xreplace(repl_map)

expr_short = sp.simplify(expr_short)

expr_short