# import rospy
import cvxpy as cp
import csv
import os
import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# bilinear interpolation for coordinate "(i,j)" on a 2-D grid "h"
def bilinear_interpolation(h, i, j):

    i1f = np.floor(i)
    j1f = np.floor(j)
    i2f = np.ceil(i)
    j2f = np.ceil(j)

    i1 = int(i1f)
    j1 = int(j1f)
    i2 = int(i2f)
    j2 = int(j2f)

    f = 0.0
    if (i1 != i2) and (j1 != j2):
        f1 = (i2f - i) * h[i1,j1] + (i - i1f) * h[i2,j1]
        f2 = (i2f - i) * h[i1,j2] + (i - i1f) * h[i2,j2]
        f = (j2f - j) * f1 + (j - j1f) * f2
    elif (i1 != i2):
        f = (i2f - i) * h[i1,int(j)] + (i - i1f) * h[i2,int(j)]
    elif (j1 != j2):
        f = (j2f - j) * h[int(i),j1] + (j - j1f) * h[int(i),j2]
    else:
        f = h[int(i),int(j)]

    return float(f)


# Extract h0 Value and Gradient
def get_h0(h, x, y):

    imax = 230.0 # Grid I Dimension (Negative Y-Direction)
    jmax = 1000.0 # Grid J Dimension (Positive X-Direction)
    ds = 0.01 # Grid Resolution

    x0, y0 = -1.00, -1.50 

    ir = (y - y0) / ds
    jr = (x - x0) / ds

    ic = np.fmin(np.fmax(0.0, ir), imax-1.0) 
    jc = np.fmin(np.fmax(0.0, jr), jmax-1.0)

    h0 = bilinear_interpolation(h, ic, jc)

    if (ic!=ir) and (jc!=jr):
        h0 -= np.sqrt(np.abs(ir-ic)**2 + np.abs(jr-jc)**2) * ds
    elif (ic!=ir):
        h0 -= np.abs(ir-ic) * ds
    elif (jc!=jr):
        h0 -= np.abs(jr-jc) * ds

    i_eps = 5.0
    j_eps = 5.0
    ip = np.fmin(np.fmax(0.0, ic + i_eps), imax-1.0)
    im = np.fmin(np.fmax(0.0, ic - i_eps), imax-1.0)
    jp = np.fmin(np.fmax(0.0, jc + j_eps), jmax-1.0)
    jm = np.fmin(np.fmax(0.0, jc - j_eps), jmax-1.0)
    hxp = bilinear_interpolation(h, ic, jp)
    hxm = bilinear_interpolation(h, ic, jm)
    hyp = bilinear_interpolation(h, ip, jc)
    hym = bilinear_interpolation(h, im, jc)
    dhdx = (hxp - hxm) / ((jp-jm)*ds)
    dhdy = (hyp - hym) / ((ip-im)*ds)

    # Hessian
    hpp = bilinear_interpolation(h, ip, jp)
    hpm = bilinear_interpolation(h, im, jp)
    hmp = bilinear_interpolation(h, ip, jm)
    hmm = bilinear_interpolation(h, im, jm)

    d2hdx2 = ((hxp - h0) / ((jp-jc)*ds) - (h0 - hxm) / ((jc-jm)*ds)) / ((jp-jm)/2.0*ds)
    d2hdy2 = ((hyp - h0) / ((ip-ic)*ds) - (h0 - hym) / ((ic-im)*ds)) / ((ip-im)/2.0*ds)
    d2hdxdy = (hpp + hmm - hpm - hmp) / ((jp-jm)*(ip-im)*ds*ds)

    def is_real(x):
        if not isinstance(x,(int,float)):
            return False
        return not (np.isnan(x) or np.isinf(x))

    if not is_real(d2hdx2):
        d2hdx2 = 0.0
    if not is_real(d2hdy2):
        d2hdy2 = 0.0
    if not is_real(d2hdxdy):
        d2hdxdy = 0.0
    if not is_real(dhdx):
        if jc == 0:
            dhdx = 1.0
        elif jc == jmax:
            dhdx = -1.0
    if not is_real(dhdy):
        if ic == 0:
            dhdy = 1.0
        elif ic == imax:
            dhdy = -1.0

    return h0, dhdx, dhdy, d2hdx2, d2hdy2, d2hdxdy


class SSF:
    """Robust Control Barrier Functions-Based Safety Filter
          SSF takes in the nominal control signal(u_nominal) and filters it to
          generate u_filtered. Optimization problem
          u* = argmin || u_des - u ||^2
          s.t.            L_f h(x) + L_g h(x) u - k_1 ||Lgh(x)|| - k_2^2 ||Lgh(x)||^2 + a(h(x)) >= 0
                          - u_max <= u <= u_max
                                |  u_nominal   |        |   u_filtered
              nominal control   |  -------->   |  SSF   |   ----------->
    """

    counter = 0

    def __init__(
            self,
            v_limit = 2,  
            w_limit = 2,    
            dt = 0.02,  
            alpha = 3,  
            v_ref = 0.25,
            delta_obs = 0.3,
            obs_x = np.array([2.5,  6.9]),
            obs_y = np.array([0.0,0.0]),
            data_dict=None
    ):
        """
        Limitations on tracks, linear and angular velocities (control inputs) are enforced
        """
        if data_dict is None:
            data_dict = {"h1": 0}
        self.data_dict = data_dict

        self.x_init = None
        self.y_init = None
        self.psi_init = None

        self.Kv = 1.0
        self.Kom = 2.5

        # a forward speed to use for x_des(t)
        self.v_ref = v_ref

        self.c = 2.9 * np.pi / 6 # frequency parameter for the narrow passage
        self.b = 0.0             # phase shift for the narrow passage
        self.a = 0.0             # vertical offset for the narrow passage

        self.y_0_init = -0.35

        # tunable heading-weight 
        self.delta_obs = delta_obs

        # velocity limits
        self.v_limit = v_limit  
        self.w_limit = w_limit  
       
        self.dt = dt  # sampling time (fixed)

        # CBF paramters
        self.alpha = alpha  # cbf class-k function parameter

        # initial robustness parameters
        self.k_1 = cp.Parameter(value=0.001)
        self.k_2 = cp.Parameter(value=0.001)

        # Read the CSV files
        imax = 230.0 # Grid I Dimension (Negative Y-Direction)
        jmax = 1000.0 # Grid J Dimension (Positive X-Direction)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        # csv_path = os.path.join(script_dir, 'poisson_safety_grid_2_5.csv')
        csv_path = os.path.join(script_dir, 'poisson_safety_grid_1_5.csv')
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            # data = list(reader)
            data = [[float(val) for val in row] for row in reader]

        self.hvalues = np.reshape(data, (int(imax),int(jmax)))

        self.tunable = cp.Parameter()

        # decision variables
        self.track_safe_acc_lin = cp.Variable()  # v
        self.track_safe_acc_ang = cp.Variable()  # omega

        self.nominal_linear_acc = cp.Parameter() 
        self.nominal_angular_acc = cp.Parameter()  

        # for the position dependent constraint
        self.pose_xi = cp.Parameter()   # x-axis in the world frame
        self.pose_yi = cp.Parameter()   # y-axis in the world frame
        self.pose_psi = cp.Parameter()  # theta angle

        self.nominal_linear = cp.Parameter()
        self.nominal_angular = cp.Parameter()

        self.sin_psi = cp.Parameter()
        self.cos_psi = cp.Parameter()

        self.conspar = cp.Parameter() 
        self.conspar_ang   = cp.Parameter()  
        self.conspar_timevarying  = cp.Parameter()  
        self.cbf_alpha = cp.Parameter()  
        self.cons_pose1 = cp.Parameter()
        self.cons_pose2 = cp.Parameter()

        # cost function
        cost = cp.square(self.track_safe_acc_lin - self.nominal_linear) + cp.square(
            self.track_safe_acc_ang - self.nominal_angular)
        # print(cost.is_dcp(dpp=True))  # is the cost Disciplined Parametrized Programming (DPP)
        
        # robust CBF with poisson
        constr = [self.track_safe_acc_lin * self.conspar + self.track_safe_acc_ang * self.conspar_ang - self.tunable + self.conspar_timevarying + self.cbf_alpha >= 0,
                  cp.abs(self.track_safe_acc_lin) <= self.v_limit,
                  cp.abs(self.track_safe_acc_ang) <= self.w_limit]

        self.prob = cp.Problem(cp.Minimize(cost), constr)

        # to check whether the problem is DCP and/or DPP
        print(self.prob.is_dcp(dpp=True))  # is the problem DPP, constraints should be affine in parameters
        print(self.prob.is_dcp())          # is the problem Disciplined Convex Programming (DCP)

    def xy_ref(self, state, t, y_mag = 1.5, c = None, y_shift = 0.0):
        if self.x_init is None:
                self.x_init = np.zeros((3, 1)) 

        if c is not None:
            self.c = c

        x   = state[0, 0]
        y   = state[1, 0]
        
        x_des = self.v_ref * t  
        y_des = y_mag * np.sin(self.c * x_des + y_shift) + self.y_0_init
        return x_des, y_des
    

    def k_des(self, state, t, y_mag = 1.5, c = None, y_shift = 0.0):
        """
        state = [x, y, psi]^T
        We define a reference x_des(t) that moves forward in x at speed self.v_ref,
        and then compute y_des from the sinusoid.
        returns theta_d_dot (analytic time derivative of theta_d).
        """
        if self.x_init is None:
                self.x_init = np.zeros((3, 1)) 

        if c is not None:
            self.c = c

        x   = state[0, 0]
        y   = state[1, 0]
        psi = state[2, 0]
        
        psi = wrap_to_pi(psi)

        # constant reference input to test the controller
        # x_des = 3.0
        # y_des = 0.0
        
        x_des = self.v_ref * t  
        y_des = y_mag * np.sin(self.c * x_des + y_shift) + self.y_0_init

        # distance and heading error
        error = np.array([[x_des], [y_des]]) - np.array([[x], [y]])
        dist = np.linalg.norm(error)
        theta_d = np.arctan2(error[1, 0], error[0, 0])
        theta_d = wrap_to_pi(theta_d)
        
        # use this only for the constant ref. input
        # threshold = 0.05
        # if dist < threshold:
        #    return 0.0, 0.0

        linear = self.Kv * dist
        angular = self.Kom * wrap_to_pi(theta_d - psi)

        # reference‑trajectory velocities
        x_des_dot = self.v_ref
        y_des_dot = y_mag* self.c * self.v_ref * np.cos(self.c * x_des + y_shift)

        # robot body‑frame velocity ( we approximate with the commanded one)
        x_dot = linear * np.cos(psi)
        y_dot = linear * np.sin(psi)

        # error time‑derivatives
        ex, ey     = error.flatten()
        ex_dot     = x_des_dot - x_dot
        ey_dot     = y_des_dot - y_dot

        denom = ex**2 + ey**2
        theta_d_dot = 0.0 if denom < 1e-5 else (ex*ey_dot - ey*ex_dot) / denom

        vp       = np.array([self.Kv*ex, self.Kv*ey])          
        dvp_dx   = np.array([-self.Kv, 0.0])
        dvp_dy   = np.array([0.0, -self.Kv])
        dvp_dt   = np.array([self.Kv*x_des_dot, self.Kv*y_des_dot]) # partial derivative wrt t
        # dvp_dt   = np.array([self.Kv*ex_dot, self.Kv*ey_dot])


        return {
            "linear_cmd":  linear,
            "angular_cmd": angular,
            "theta_des":   theta_d,
            "theta_d_dot": theta_d_dot,
            "ex": ex, "ey": ey,
            "x_des_dot": x_des_dot, "y_des_dot": y_des_dot,
            "vp": vp,
            "dvp_dx": dvp_dx, "dvp_dy": dvp_dy, "dvp_dt": dvp_dt
        }

        # return linear, angular, theta_d, theta_d_dot, ex, ey
    

    def control_ssf(self, state, u_nominal):

        self.pose_xi.value = state[0]
        self.pose_yi.value = state[1]
        psi = wrap_to_pi(state[2])

        self.sin_psi.value = np.sin(psi)
        self.cos_psi.value = np.cos(psi)
        self.pose_psi.value = psi

        t = SSF.counter * self.dt

        # #############################################
        # # we use this for the classical safety filter
        # self.nominal_linear.value, self.nominal_angular.value, theta_des, theta_dot_des, ex, ey = self.k_des(
        #     np.array([[self.pose_xi.value, self.pose_yi.value, self.pose_psi.value]]).T, t)

        k = self.k_des(np.array([[self.pose_xi.value, self.pose_yi.value,self.pose_psi.value]]).T, t)
        
        # theta_des     = k["theta_des"]
        # theta_dot_des = k["theta_d_dot"]
        # ex, ey        = k["ex"], k["ey"]

        # tracker velocity and Jacobians
        vp      = k["vp"]
        dvp_dx  = k["dvp_dx"]
        dvp_dy  = k["dvp_dy"]
        dvp_dt  = k["dvp_dt"]       
        
        # h0, dhdx, dhdy = get_h0(self.hvalues, self.pose_xi.value, self.pose_yi.value)
        h0, dhdx, dhdy, d2hdx2, d2hdy2, d2hdxdy = get_h0(self.hvalues, self.pose_xi.value, self.pose_yi.value)

        a0      = self.delta_obs
        alpha_f = self.alpha
        alpha_q = 0.1

        grad_h0 = np.array([dhdx, dhdy])
        Hh0     = np.array([[d2hdx2, d2hdxdy], [d2hdxdy, d2hdy2]])

        # a(x,y,t), b(x,y) 
        a_val = grad_h0 @ vp + alpha_f * h0
        b_val = grad_h0 @ grad_h0 + 1e-12

        da_dx = grad_h0 @ dvp_dx + Hh0[:,0] @ vp + alpha_f * dhdx
        da_dy = grad_h0 @ dvp_dy + Hh0[:,1] @ vp + alpha_f * dhdy
        da_dt = grad_h0 @ dvp_dt
        
        db_dx = 2*(dhdx*d2hdx2 + dhdy*d2hdxdy)
        db_dy = 2*(dhdx*d2hdxdy + dhdy*d2hdy2)

        # \lambda(a, b)
        S      = np.sqrt(a_val**2 + alpha_q*b_val**2)
        lam    = (-a_val + S) / (2*b_val)
        lam_a  = (-1 + a_val/S) / (2*b_val)
        # lam_b  =  alpha_q*b_val/(2*S) - (a_val+S)/(2*b_val**2)
        lam_b =  alpha_q/(2*S) + (a_val - S)/(2*b_val**2)
        dlam_dx = lam_a*da_dx + lam_b*db_dx
        dlam_dy = lam_a*da_dy + lam_b*db_dy
        dlam_dt = lam_a*da_dt

        # safe velocity 
        vs      = vp + lam*grad_h0
        dvs_dx  = dvp_dx + dlam_dx*grad_h0 + lam*Hh0[:,0]
        dvs_dy  = dvp_dy + dlam_dy*grad_h0 + lam*Hh0[:,1]
        dvs_dt  = dvp_dt + dlam_dt*grad_h0

        # \theta_s 
        Ns      = vs[0]**2 + vs[1]**2 
        def dtheta(dvs): return (vs[0]*dvs[1] - vs[1]*dvs[0]) / Ns
        # theta_s = np.arctan2(vs[1], vs[0])
        theta_s = wrap_to_pi(np.arctan2(vs[1], vs[0]))
        ths_x, ths_y, ths_t = dtheta(dvs_dx), dtheta(dvs_dy), dtheta(dvs_dt)

        # Lie derivatives 
        delta_th = wrap_to_pi(self.pose_psi.value - theta_s)
        sinDel   = np.sin(delta_th)
        hx = dhdx + a0*sinDel*ths_x
        hy = dhdy + a0*sinDel*ths_y
        hpsi = -a0*sinDel
        ht =  a0*sinDel*ths_t

        Lgv     = hx*self.cos_psi.value + hy*self.sin_psi.value
        Lgomega = hpsi
        Lt      = ht

        self.conspar.value             = Lgv
        self.conspar_ang.value         = Lgomega
        self.conspar_timevarying.value = Lt
        self.cbf_alpha.value           = self.alpha * (h0 - a0*(1 - np.cos(delta_th)))

        # ======================================================================
        # this is a magic trick: nominal input based on safe integrator velcoity, vs
        self.nominal_linear.value = np.linalg.norm(vs)
        # vs_direction = np.arctan2(vs[1],vs[0]) 
        theta_s = wrap_to_pi(np.arctan2(vs[1], vs[0]))
        self.nominal_angular.value = -self.Kom * wrap_to_pi(psi - theta_s)
        # ======================================================================

        # ======================================================================
        # if we want t o use k_des directly
        # k_out = self.k_des(np.array([[self.pose_xi.value,
        #                           self.pose_yi.value,
        #                           self.pose_psi.value]]).T, t) # same time variable
        # self.nominal_linear.value  = k_out["linear_cmd"]
        # self.nominal_angular.value = k_out["angular_cmd"]

        # or just uncomment this:
        # self.nominal_linear.value, self.nominal_angular.value = u_nominal 
        # ======================================================================

        # record h values
        h1 = self.cbf_alpha.value / self.alpha

        # Nominal state vector
        x_nom = np.array([self.pose_xi.value, self.pose_yi.value, self.pose_psi.value])
        Lg_vec_nom = np.array([Lgv, Lgomega])
        g_norm_nom = np.linalg.norm(Lg_vec_nom) + 1e-12
        u_base_nom = np.array([self.nominal_linear.value, self.nominal_angular.value])
        alpha_h_nom = self.cbf_alpha.value
        Lt_nom = Lt

        # gamma grid 
        gammas_vec   = np.linspace(1e-3, 4.0, 400)
        gammas1, gammas2 = np.meshgrid(gammas_vec, gammas_vec)
        gammas1_flat = gammas1.ravel()
        gammas2_flat = gammas2.ravel()
        n_gamma      = gammas1_flat.size

        # sample perturbed states
        N        = 100
        deltas   = np.array([0.05, 0.1, 0.00])
        X_pert   = x_nom + np.random.uniform(-deltas, deltas, size=(N, 3))
        x_p, y_p, psi_p = X_pert[:,0], X_pert[:,1], X_pert[:,2]

        h0_p = np.empty(N); dhdx_p = np.empty(N); dhdy_p = np.empty(N)
        d2hdx2_p = np.empty(N); d2hdy2_p = np.empty(N); d2hdxdy_p = np.empty(N)
        for i in range(N):
            h0_p[i], dhdx_p[i], dhdy_p[i], d2hdx2_p[i], d2hdy2_p[i], d2hdxdy_p[i] = get_h0(self.hvalues, x_p[i], y_p[i])
        grad_h0_p = np.column_stack((dhdx_p, dhdy_p))

        vs_p = np.zeros((N,2)); ths_x_p = np.zeros(N); ths_y_p = np.zeros(N); ths_t_p = np.zeros(N)
        theta_s_p = np.zeros(N)
        t_now = SSF.counter * self.dt
        for i in range(N):
            state_i = np.array([[x_p[i], y_p[i], psi_p[i]]]).T
            k_i     = self.k_des(state_i, t_now)
            vp_i    = k_i["vp"]; dvp_dx_i = k_i["dvp_dx"]; dvp_dy_i = k_i["dvp_dy"]; dvp_dt_i = k_i["dvp_dt"]
            a_i     = grad_h0_p[i] @ vp_i + alpha_f * h0_p[i]
            b_i     = grad_h0_p[i] @ grad_h0_p[i] + 1e-12
            S_i     = np.sqrt(a_i**2 + alpha_q * b_i**2)
            lam_i   = (-a_i + S_i) / (2*b_i)
            vs_i    = vp_i + lam_i * grad_h0_p[i]
            vs_p[i] = vs_i
            Ns_i    = vs_i[0]**2 + vs_i[1]**2 + 1e-12
            # derivatives for theta_s
            Hh0_i   = np.array([[d2hdx2_p[i], d2hdxdy_p[i]], [d2hdxdy_p[i], d2hdy2_p[i]]])
            da_dx_i = grad_h0_p[i] @ dvp_dx_i + Hh0_i[:,0] @ vp_i + alpha_f * dhdx_p[i]
            da_dy_i = grad_h0_p[i] @ dvp_dy_i + Hh0_i[:,1] @ vp_i + alpha_f * dhdy_p[i]
            lam_a_i = (-1 + a_i / S_i) / (2*b_i)
            lam_b_i = alpha_q / (2*S_i) + (a_i - S_i) / (2*b_i**2)
            db_dx_i = 2*(dhdx_p[i]*d2hdx2_p[i] + dhdy_p[i]*d2hdxdy_p[i])
            db_dy_i = 2*(dhdx_p[i]*d2hdxdy_p[i] + dhdy_p[i]*d2hdy2_p[i])
            dlam_dx_i = lam_a_i*da_dx_i + lam_b_i*db_dx_i
            dlam_dy_i = lam_a_i*da_dy_i + lam_b_i*db_dy_i
            dlam_dt_i = lam_a_i*(grad_h0_p[i] @ dvp_dt_i)
            dvs_dx_i  = dvp_dx_i + dlam_dx_i*grad_h0_p[i] + lam_i*Hh0_i[:,0]
            dvs_dy_i  = dvp_dy_i + dlam_dy_i*grad_h0_p[i] + lam_i*Hh0_i[:,1]
            dvs_dt_i  = dvp_dt_i + dlam_dt_i*grad_h0_p[i]
            theta_s_p[i] = wrap_to_pi(np.arctan2(vs_i[1], vs_i[0]))
            def dtheta_i(dvs): return (vs_i[0]*dvs[1] - vs_i[1]*dvs[0]) / Ns_i
            ths_x_p[i] = dtheta_i(dvs_dx_i); ths_y_p[i] = dtheta_i(dvs_dy_i); ths_t_p[i] = dtheta_i(dvs_dt_i)

        delta_th_p = wrap_to_pi(psi_p - theta_s_p)
        sinDel_p   = np.sin(delta_th_p)
        hx_p = dhdx_p + a0 * sinDel_p * ths_x_p
        hy_p = dhdy_p + a0 * sinDel_p * ths_y_p
        hpsi_p = -a0 * sinDel_p
        Lt_p   = a0 * sinDel_p * ths_t_p
        h_val_p    = h0_p - a0 * (1 - np.cos(delta_th_p))
        alpha_h_p  = self.alpha * h_val_p
        Lg_v_p     = hx_p * np.cos(psi_p) + hy_p * np.sin(psi_p)
        Lg_vec_p   = np.column_stack((Lg_v_p, hpsi_p))
        g_norm_p   = np.linalg.norm(Lg_vec_p, axis=1) + 1e-12
        # u_base_p   = np.tile(u_base_nom, (N,1))  
        u_base_p = np.column_stack((
            np.linalg.norm(vs_p, axis=1),
            -self.Kom * np.array([wrap_to_pi(psi_p[i] - theta_s_p[i]) for i in range(N)])))

        # Vectorized R-CBF QP solution for all gamma
        rhs_nom = -(Lt_nom + alpha_h_nom) + gammas1_flat * g_norm_nom + (gammas2_flat**2) * (g_norm_nom**2)
        proj_nom = Lg_vec_nom @ u_base_nom
        gain_nom = np.maximum(rhs_nom - proj_nom, 0.0) / (g_norm_nom**2)
        u_corr_nom = u_base_nom + gain_nom[:,None] * Lg_vec_nom          # un‑clipped for feasibility
        feasible_nom = (np.abs(u_corr_nom[:,0]) <= self.v_limit) & (np.abs(u_corr_nom[:,1]) <= self.w_limit)
        U_nom = u_corr_nom.copy()
        U_nom[:,0] = np.clip(U_nom[:,0], -self.v_limit, self.v_limit)
        U_nom[:,1] = np.clip(U_nom[:,1], -self.w_limit, self.w_limit)

        rhs_p = (-(Lt_p[:,None] + alpha_h_p[:,None])
                + gammas1_flat[None,:] * g_norm_p[:,None]
                + (gammas2_flat[None,:]**2) * (g_norm_p[:,None]**2))
        proj_p = np.sum(Lg_vec_p * u_base_p, axis=1)[:,None]
        gain_p = np.maximum(rhs_p - proj_p, 0.0) / (g_norm_p[:,None]**2)
        u_corr_p = u_base_p[:,None,:] + gain_p[...,None] * Lg_vec_p[:,None,:]     # (N,n_gamma,2) un‑clipped
        feasible_p = (np.abs(u_corr_p[...,0]) <= self.v_limit) & (np.abs(u_corr_p[...,1]) <= self.w_limit)
        U_pert = u_corr_p.copy()
        U_pert[...,0] = np.clip(U_pert[...,0], -self.v_limit, self.v_limit)
        U_pert[...,1] = np.clip(U_pert[...,1], -self.w_limit, self.w_limit)

        control_diff = U_pert - U_nom[None,:,:]                 # (N,n_gamma,2)
        sigma_vals = np.max(np.linalg.norm(control_diff, axis=2), axis=0)  # (n_gamma,)

        cost = np.maximum(sigma_vals - gammas1_flat, 0.0) / (2.0 * gammas2_flat + 1e-12)
        cost += 0.1*gammas1_flat + 0.1*gammas2_flat
        rho = 0.05  
        cost += rho * ((gammas1_flat - self.k_1.value)**2 + (gammas2_flat - self.k_2.value)**2)

        overall_feasible = feasible_nom & np.all(feasible_p, axis=0)
        cost[~overall_feasible] = 1e6
        best_idx = np.argmin(cost)

        ################################## robust alternatives

        # optimal robust CBF
        self.k_1.value =  float(gammas1_flat[best_idx]) 
        self.k_2.value =  float(gammas2_flat[best_idx])

        ########### fixed gamma, Robust CBF
        # self.k_1.value =  1.4
        # self.k_2.value =  0.3

        ########### tunable robust CBF
        # eps1 = np.exp(2.0 * max(h1, 0.0))
        # eps2 = np.exp(2.0 * max(h1, 0.0))
        # self.k_1.value = 1.4 / eps1
        # self.k_2.value = 0.3 / np.sqrt(eps2)

        ################################## 

        print(f"Optimized gamma1: {self.k_1.value}, gamma2: {self.k_2.value}")


        def rob_term(k1, k2, Lg_vec):
            nrm = np.linalg.norm(Lg_vec)
            # w   = np.exp(self.k_tune * h_val)
            return k1 * nrm  + (k2 **2 ) * (nrm**2)  # we use gamma_2^2 as in the paper
        
        # tunable robust CBF term
        self.tunable.value = rob_term(self.k_1.value, self.k_2.value, np.array([self.conspar.value, self.conspar_ang.value]))

        print( f"Yaw: {psi * 180 / 3.1415926 :6.2f},"
            f"x: {self.pose_xi.value}, "
            f"y: {self.pose_yi.value}, ")

        # Solve QP
        self.prob.solve(solver="OSQP")
        print(self.prob.status)

        linear_safe = self.track_safe_acc_lin.value
        angular_safe = self.track_safe_acc_ang.value

        print("h_1(x) = ", self.cbf_alpha.value / self.alpha)

        if self.prob.status != 'infeasible':
            u_safe = [linear_safe, angular_safe]
            # u_safe = [self.nominal_linear.value, self.nominal_angular.value] # to safety off

        else:
            u_safe = [self.nominal_linear.value, self.nominal_angular.value]

        u_safe = [
            np.clip(u_safe[0], a_min=-self.v_limit, a_max=self.v_limit),
            np.clip(u_safe[1], a_min=-self.w_limit, a_max=self.w_limit)
        ]

        print(
            f"Linear {self.nominal_linear.value:6.2f} -> {u_safe[0]:6.2}, "
            f"Angular: {self.nominal_angular.value:6.2f} -> {u_safe[1]:6.2f}, "
        )

        SSF.counter += 1

        self.data_dict = {
            "gamma_1": self.k_1.value,
            "gamma_2": self.k_2.value,
            # "delta": delta,
            "h1": h1,
            "u_baseline1": self.nominal_linear.value,
            "u_baseline2": self.nominal_angular.value,
            "u_filtered1": u_safe[0],
            "u_filtered2": u_safe[1],
        }

        return u_safe
