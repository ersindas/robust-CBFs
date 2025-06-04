import sympy as sp

# ----------------------------------------------------------------------
# 0. Symbols and functions
# ----------------------------------------------------------------------
x, y, theta, t = sp.symbols('x y theta t', real=True)
v, w           = sp.symbols('v w',  real=True)          # control inputs
a0, alpha, alpha_q, K_v = sp.symbols('a0 alpha alpha_q K_v', positive=True)

h0 = sp.Function('h0')(x, y)          # abstract map h0(x,y)

# ----------------------------------------------------------------------
# 1. ∇h0 and reference-tracking velocity
# ----------------------------------------------------------------------
grad_h0_x = sp.diff(h0, x)
grad_h0_y = sp.diff(h0, y)
grad_h0_norm_squared = grad_h0_x**2 + grad_h0_y**2        # b(x,y)

x_d = sp.Function('x_d')(t)
y_d = sp.Function('y_d')(t)
v_p_x = K_v * (x_d - x)
v_p_y = K_v * (y_d - y)

# manual dot product (avoids dot()/symfun clash)
a_filter = grad_h0_x*v_p_x + grad_h0_y*v_p_y + alpha*h0
b_filter = grad_h0_norm_squared

# half-Sontag gain λ(a,b)
q_b = alpha_q * b_filter
lambda_func = (-a_filter + sp.sqrt(a_filter**2 + q_b * b_filter**2)) / (2*b_filter)

# safe velocity v_s
v_s_x = v_p_x + lambda_func * grad_h0_x
v_s_y = v_p_y + lambda_func * grad_h0_y
theta_s = sp.atan2(v_s_y, v_s_x)

# barrier h(x,y,θ,t)
V = 1 - sp.cos(theta - theta_s)
h = h0 - a0*V

# ----------------------------------------------------------------------
# 2. Total time derivative ẋh
# ----------------------------------------------------------------------
x_dot = v * sp.cos(theta)
y_dot = v * sp.sin(theta)
theta_dot = w

dh_dt = (sp.diff(h, x)*x_dot +
         sp.diff(h, y)*y_dot +
         sp.diff(h, theta)*theta_dot +
         sp.diff(h, t))

# ----------------------------------------------------------------------
# 3. Replace ∂h0 placeholders *before* simplify
# ----------------------------------------------------------------------
hx, hy        = sp.symbols('h_x h_y')
hxx, hxy, hyy = sp.symbols('h_xx h_xy h_yy')

subs_grad = {
    sp.diff(h0, x):       hx,
    sp.diff(h0, y):       hy,
    sp.diff(h0, x, x):    hxx,
    sp.diff(h0, x, y):    hxy,
    sp.diff(h0, y, x):    hxy,
    sp.diff(h0, y, y):    hyy,
}

dh_dt = dh_dt.xreplace(subs_grad)
# dh_dt = sp.simplify(dh_dt)

# ----------------------------------------------------------------------
# 4. Extract control-affine coefficients A, B1, B2
# ----------------------------------------------------------------------
try:
    B1 = sp.collect(dh_dt, v).coeff(v, 1)
    B2 = sp.collect(dh_dt, w).coeff(w, 1)
    A  = sp.simplify(dh_dt - B1*v - B2*w)
except Exception:                       # fallback if collect fails
    A  = dh_dt.subs({v:0, w:0})
    B1 = dh_dt.subs({v:1, w:0}) - A
    B2 = dh_dt.subs({v:0, w:1}) - A

dh_dt_affine = A + B1*v + B2*w          # sanity check form

# ----------------------------------------------------------------------
# 5. Optional aggressive simplification
# ----------------------------------------------------------------------
# from sympy import simplify, trigsimp, factor_terms
# A_s  = simplify(trigsimp(factor_terms(A)))
# B1_s = simplify(trigsimp(factor_terms(B1)))
# B2_s = simplify(trigsimp(factor_terms(B2)))

# ----------------------------------------------------------------------
# 6. Display and save
# ----------------------------------------------------------------------
print("\nComponents of the control-affine form:")
print("A_s  =", A)
print("B1_s =", B1)
print("B2_s =", B2)
print("\nFinal:  ẋh = A_s  +  B1_s·v  +  B2_s·w")

with open("symbolic_results_simplified.txt", "w") as f:
    f.write("A_s  = {}\n\n".format(A))
    f.write("B1_s = {}\n\n".format(B1))
    f.write("B2_s = {}\n".format(B2))
