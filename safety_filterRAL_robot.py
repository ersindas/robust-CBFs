# import rospy
import cvxpy as cp
import csv
import os
import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Perform a bilinear interpolation for coordinate "(i,j)" on a 2-D grid "h"
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


# Extract h0 Value and Gradient <dh0/dx, dh0/dy>
def get_h0(h, x, y):

    imax = 230.0 # Grid I Dimension (Negative Y-Direction)
    jmax = 1000.0 # Grid J Dimension (Positive X-Direction)
    ds = 0.01 # Grid Resolution

       # Define X-Y Coordinate for Bottom Left Corner of Grid
    x0, y0 = -1.00, -1.50 

    # Fractional Index Corresponding to Current Position
    ir = (y - y0) / ds
    jr = (x - x0) / ds

    # Saturated Because of Finite Grid Size
    ic = np.fmin(np.fmax(0.0, ir), imax-1.0) 
    jc = np.fmin(np.fmax(0.0, jr), jmax-1.0)

    # Get Safety Function Value
    h0 = bilinear_interpolation(h, ic, jc)

    # If You Have Left The Grid, Use SDF to Get Back
    if (ic!=ir) and (jc!=jr):
        h0 -= np.sqrt(np.abs(ir-ic)**2 + np.abs(jr-jc)**2) * ds
    elif (ic!=ir):
        h0 -= np.abs(ir-ic) * ds
    elif (jc!=jr):
        h0 -= np.abs(jr-jc) * ds

    # Compute Gradient
    i_eps = 10.0
    j_eps = 10.0
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

    # Compute Hessian
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
          s.t.            L_f h(x) + L_g h(x) u - k_1 ||Lgh(x)|| - k_2 ||Lgh(x)||^2 + a(h(x)) >= 0
                          - u_max <= u <= u_max
                                |  u_nominal   |        |   u_filtered
              nominal control   |  -------->   |  SSF   |   ----------->
    """

    counter = 0

    def __init__(
            self,
            v_limit=1.5,  
            w_limit=1.5,    
            dt=0.1,  # stamp real time
            alpha=1,  # class-K function for CBF
            v_ref = 0.25,
            delta_obs = 0.5,
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

        scale = 0.1
        self.Kv = 10 * scale
        self.Kom = 15 * scale

        # a forward speed to use for x_des(t)
        self.v_ref = v_ref

        self.c = 2.9 * np.pi / 6 # frequency parameter for the narrow passage
        self.b = 0.0       # phase shift for the narrow passage
        self.a = 0.0       # vertical offset for the narrow passage

        self.y_0_init = -0.5

        # # lane keeping
        # self.d = (3.03) / 2 - 0.4 # the robot body shoud stay in the set, a circle around the robot

        # # obstacle-1  and obstacle-2
        # self.obs_x  = np.array([3.0,  6.0])   
        # self.obs_y  = np.array([0.0,  0.0])   
        # self.obs_R  = np.array([0.25,  0.25])   # radius

        # tunable heading-weight 
        self.delta_obs = delta_obs

        # velocity and acceleration limits
        self.v_limit = v_limit  
        self.w_limit = w_limit  
       
        self.dt = dt  # sampling time (fixed)

        # CBF paramters
        self.alpha = alpha  # cbf class-k function parameter

        # initial robustness parameters
        self.k_1 = 0 
        self.k_2 = 0 

        # # self.k_tune = -0.5 # if is tunable
        # self.k_tune = 0.0 # if is not tunable

        # Read the CSV files
        imax = 230.0 # Grid I Dimension (Negative Y-Direction)
        jmax = 1000.0 # Grid J Dimension (Positive X-Direction)
        # ds = 0.01 # Grid Resolution

        script_dir = os.path.dirname(os.path.realpath(__file__))
        # csv_path = os.path.join(script_dir, 'poisson_safety_grid_2_5.csv')
        csv_path = os.path.join(script_dir, 'poisson_safety_grid_1_5.csv')
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            # data = list(reader)
            data = [[float(val) for val in row] for row in reader]


        self.hvalues = np.reshape(data, (int(imax),int(jmax)))

        # Tunable-ISSf-ICBF parameter
        self.tunable = cp.Parameter()

        # decision variables
        self.track_safe_acc_lin = cp.Variable()  # v
        self.track_safe_acc_ang = cp.Variable()  # omega

        self.nominal_linear_acc = cp.Parameter()  # du_input is = (u_k - u_(k-1))/dt or acc_lin
        self.nominal_angular_acc = cp.Parameter()  # du_input_w is = (u_k - u_(k-1))/dt or acc_ang

        # for the position dependent constraint
        self.pose_xi = cp.Parameter()  # x-axis in the world frame
        self.pose_yi = cp.Parameter()  # y-axis in the world frame
        self.pose_psi = cp.Parameter()  # theta angle

        self.nominal_linear = cp.Parameter()
        self.nominal_angular = cp.Parameter()

        self.sin_psi = cp.Parameter()
        self.cos_psi = cp.Parameter()

        self.sinydx = cp.Parameter()
        self.cosydx = cp.Parameter()

        self.conspar = cp.Parameter()
        self.conspar_ang   = cp.Parameter()  # for omega, time varying
        self.conspar_timevarying  = cp.Parameter()  # for the time varying part

        self.cbf_alpha = cp.Parameter()  

        self.cons_pose1 = cp.Parameter()
        self.cons_pose2 = cp.Parameter()

        # cost function
        cost = cp.square(self.track_safe_acc_lin - self.nominal_linear) + cp.square(
            self.track_safe_acc_ang - self.nominal_angular)
        # print(cost.is_dcp(dpp=True))  # is the cost Disciplined Parametrized Programming (DPP)
        
        # tunable robust CBF with poisson
        constr = [self.track_safe_acc_lin * self.conspar + self.track_safe_acc_ang * self.conspar_ang - self.tunable + self.conspar_timevarying + self.cbf_alpha >= 0,
                  cp.abs(self.track_safe_acc_lin) <= self.v_limit,
                  cp.abs(self.track_safe_acc_ang) <= self.w_limit]

        self.prob = cp.Problem(cp.Minimize(cost), constr)

        # to check if whether the problem is DCP and/or DPP
        print(self.prob.is_dcp(dpp=True))  # is the problem DPP, constraints should be affine in parameters
        print(self.prob.is_dcp())  # is the problem Disciplined Convex Programming (DCP)

    def xy_ref(self, state, t,y_mag=1.5, c= None,y_shift = 0.0):
        if self.x_init is None:
                self.x_init = np.zeros((3, 1)) 

        if c is not None:
            self.c = c

        x   = state[0, 0]
        y   = state[1, 0]
        # psi = state[2, 0]
        
        # psi = wrap_to_pi(psi)

        # constant reference input to test the controller
        # x_des = 3.0
        # y_des = 0.0
        
        x_des = self.v_ref * t  # 6 meters in 152s → v_ref = 0.5, x starts with 0
        # x_des = self.x_init + self.v_ref  * t  # 6 meters in 152s → v_ref = 0.5
        y_des = y_mag * np.sin(self.c * x_des + y_shift) + self.y_0_init
        return x_des, y_des
    

    def k_des(self, state, t, y_mag=1.5, c= None, y_shift = 0.0):
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
        
        x_des = self.v_ref * t  # 6 meters in 152s → v_ref = 0.5, x starts with 0
        # x_des = self.x_init + self.v_ref  * t  # 6 meters in 152s → v_ref = 0.5
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
        angular = -self.Kom * wrap_to_pi(psi - theta_d)

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
        dvp_dt   = np.array([self.Kv*x_des_dot, self.Kv*y_des_dot])

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


    # Closed-form R-CBF-QP safe control  
    # def _k_safe(self, state_vec, gamma1, gamma2, t_now):
    #     x, y, psi = state_vec
    #     t = t_now

    #     h0, dhdx, dhdy, *_ = get_h0(self.hvalues, x, y)
    #     grad_h0   = np.array([dhdx, dhdy])
    #     Lfh       = 0.0                                 
    #     Lg_v      = grad_h0 @ np.array([np.cos(psi), np.sin(psi)])
    #     Lg_omega  = 0.0
    #     Lg_vec    = np.array([Lg_v, Lg_omega])
    #     Lg_norm   = np.linalg.norm(Lg_vec)
    #     alpha_h   = self.alpha * h0
    #     rhs       = -alpha_h + gamma1*Lg_norm + (gamma2**2)*(Lg_norm**2) - Lfh

    #     k_dict = self.k_des(state_vec.reshape(3, 1), t)
    #     k_nom  = np.array([k_dict["linear_cmd"], k_dict["angular_cmd"]])

    #     if Lg_vec @ k_nom >= rhs:
    #         return k_nom

    #     u_star = k_nom + (rhs - Lg_vec @ k_nom) / (Lg_norm**2 + 1e-12) * Lg_vec
    #     return u_star
    

    def control_ssf(self, state, u_nominal):
        #
        # u_nominal: [ v, w ]
        # states: x, y, \theta

        self.pose_xi.value = state[0]  # x-axis in the world frame
        self.pose_yi.value = state[1]  # y-axis in the world frame

        # x_temp = state[0]
        # y_temp = state[1]
        # psi_temp = state[2]
        # if SSF.counter == 0: # initial pose is always 0
        #     print(SSF.counter)
        #     self.x_init = x_temp
        #     self.y_init = y_temp
        #     self.psi_init = psi_temp
        #     self.pose_xi.value, self.pose_yi.value, psi = [0,0,0]
        # else:
        #     self.pose_xi.value = x_temp - self.x_init  # x-axis in the world frame
        #     self.pose_yi.value = y_temp - self.y_init  # y-axis in the world frame
        #     # psi = psi_temp - self.psi_init
        #     psi = wrap_to_pi(psi_temp - self.psi_init)

        self.pose_xi.value = state[0]
        self.pose_yi.value = state[1]
        psi = wrap_to_pi(state[2])

        self.sinydx.value = np.sin(self.c * self.pose_xi.value + self.b)
        self.cosydx.value = np.cos(self.c * self.pose_xi.value + self.b)

        self.sin_psi.value = np.sin(psi)
        self.cos_psi.value = np.cos(psi)
        self.pose_psi.value = psi


        #  for the safety constraint d^2- (y - sin(cx)^2
        # self.conspar.value = - 2 * (self.pose_yi.value - self.sinydx.value + self.a) * (
        #         self.sin_psi.value - self.c * self.cos_psi.value * self.cosydx.value)
        
        # self.cbf_alpha.value = self.alpha * (
        #             (self.d)**2 - (self.pose_yi.value - (self.sinydx.value + self.a))**2 )
        
        #  for the safety constraint d^2- y^2

        # if we want to use k_des:
        # time = number_of_calls * dt
        t = SSF.counter * self.dt

        # #############################################
        # # we use this for the classical safety filter
        # self.nominal_linear.value, self.nominal_angular.value, theta_des, theta_dot_des, ex, ey = self.k_des(
        #     np.array([[self.pose_xi.value, self.pose_yi.value, self.pose_psi.value]]).T, t)

        k = self.k_des(np.array([[self.pose_xi.value,
                          self.pose_yi.value,
                          self.pose_psi.value]]).T, t)
        
        # theta_des     = k["theta_des"]
        # theta_dot_des = k["theta_d_dot"]
        # ex, ey        = k["ex"], k["ey"]

        # tracker velocity and Jacobians
        vp      = k["vp"]
        dvp_dx  = k["dvp_dx"]
        dvp_dy  = k["dvp_dy"]
        dvp_dt  = k["dvp_dt"]
        
        # rho2  = ex**2 + ey**2 + 1e-8               
        # u_ang = wrap_to_pi(theta_des - self.pose_psi.value)         
        
        # h0, dhdx, dhdy = get_h0(self.hvalues, self.pose_xi.value, self.pose_yi.value)
        h0, dhdx, dhdy, d2hdx2, d2hdy2, d2hdxdy = get_h0(self.hvalues, self.pose_xi.value, self.pose_yi.value)

        a0      = self.delta_obs
        alpha_f = self.alpha
        alpha_q = 0.1

        grad_h0 = np.array([dhdx, dhdy])
        Hh0     = np.array([[d2hdx2, d2hdxdy],
                            [d2hdxdy, d2hdy2]])

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
        theta_s = np.arctan2(vs[1], vs[0])
        ths_x, ths_y, ths_t = dtheta(dvs_dx), dtheta(dvs_dy), dtheta(dvs_dt)

        # Lie derivatives 
        delta_th = wrap_to_pi(self.pose_psi.value - theta_s)
        sinDel   = np.sin(delta_th)
        hx = dhdx + a0*sinDel*ths_x

        # print(f"Hh0: {Hh0[:,1]}")
        # print(f"ths_y: {ths_y:.4f}")

        hy = dhdy + a0*sinDel*ths_y
        hpsi = -a0*sinDel
        ht =  a0*sinDel*ths_t

        # print(f"hx: {hx:.4f}")
        # print(f"hy: {hy:.4f}")

        Lgv     = hx*self.cos_psi.value + hy*self.sin_psi.value
        Lgomega = hpsi
        Lt      = ht

        self.conspar.value             = Lgv
        self.conspar_ang.value         = Lgomega
        self.conspar_timevarying.value = Lt
        self.cbf_alpha.value           = self.alpha * (h0 - a0*(1 - np.cos(delta_th)))

        # THIS IS A MAGIC TRICK: nominal input based on vs
        self.nominal_linear.value = np.linalg.norm(vs)
        # vs_direction = np.arctan2(vs[1],vs[0]) 
        theta_s = np.arctan2(vs[1], vs[0])
        self.nominal_angular.value = -self.Kom * wrap_to_pi(psi - theta_s)
        # ======================================================================

        # if we want t o use k_des directly
        # k_out = self.k_des(np.array([[self.pose_xi.value,
        #                           self.pose_yi.value,
        #                           self.pose_psi.value]]).T,
        #                t)                     # same time variable
        # self.nominal_linear.value  = k_out["linear_cmd"]
        # self.nominal_angular.value = k_out["angular_cmd"]

        # self.nominal_linear.value, self.nominal_angular.value = u_nominal 
        
        # rho2  = ex**2 + ey**2 + 1e-8               
        # u_ang = wrap_to_pi(theta_des - self.pose_psi.value)         
        
        # h0, dhdx, dhdy = get_h0(self.hvalues, self.pose_xi.value, self.pose_yi.value)

        # self.conspar.value = ((dhdx) * (self.cos_psi.value ) + (dhdy) * (self.sin_psi.value) 
        #                       - self.delta_obs * np.sin(u_ang) * (ey * self.cos_psi.value - ex * self.sin_psi.value) / rho2)

        # # self.conspar.value = - 2 * (self.pose_yi.value) * (self.sin_psi.value )

        # self.conspar_ang.value =  (self.delta_obs) * (np.sin(wrap_to_pi(theta_des - psi)))  

        # self.conspar_timevarying.value = - (self.delta_obs) * theta_dot_des * (np.sin(wrap_to_pi(theta_des - psi)))

        # self.cbf_alpha.value = self.alpha * (h0 - self.delta_obs * (1 - np.cos(wrap_to_pi(theta_des - psi))))  # 

        # self.cbf_alpha.value = self.alpha * ((self.d)**2 - (self.pose_yi.value)**2 )  # update this

        # record h values
        h1 = self.cbf_alpha.value / self.alpha

        # parameters for grid search
        # gamma_vals = np.linspace(0.0, 1, 100)
        # best_cost = np.inf
        # best_gamma1, best_gamma2 = None, None

        # Kalman filter provided delta (variance)
        # delta = self.delta_x  # Kalman filter provided delta (variance)
        delta = 1 * 0.2  # Kalman filter provided delta (variance)


        # sample N points around the current state
        # N = 100
        # np.random.seed(0)  

        # start_time = time.time()

        # t_now  = SSF.counter * self.dt
        x_nom  = np.array([self.pose_xi.value,
                           self.pose_yi.value,
                           self.pose_psi.value])

        # derivatives that do NOT depend on gamma
        h0, dhdx, dhdy, *_ = get_h0(self.hvalues, x_nom[0], x_nom[1])
        grad_h0   = np.array([dhdx, dhdy])
        Lf_h      = 0.0
        Lg_v      = grad_h0 @ np.array([np.cos(x_nom[2]),
                                        np.sin(x_nom[2])])
        Lg_vec    = np.array([Lg_v, 0.0])              
        g_norm    = np.linalg.norm(Lg_vec) + 1e-12     
        alpha_hs        = self.alpha * h0

        
        k_out     = self.k_des(x_nom.reshape(3, 1), t)
        u_nom     = np.array([k_out["linear_cmd"], k_out["angular_cmd"]])
        proj_nom  = Lg_vec @ u_nom

        gammas_vec   = np.linspace(1e-3, 1.0, 60)           # 60 × 60 grid
        gammas1, gammas2  = np.meshgrid(gammas_vec, gammas_vec)
        gammas1_flat = gammas1.ravel()                           # shape (3600,)
        gammas2_flat = gammas2.ravel()

        rho      = -Lf_h - alpha_hs + gammas1_flat*g_norm + (gammas2_flat**2) * (g_norm**2)
        gain     = np.maximum(rho - proj_nom, 0.0) / g_norm**2
        U_nom    = u_nom + gain[:, None] * Lg_vec      # (3600, 2)

        # delta    = self.delta_x
        N        = 80
        X_pert   = x_nom + np.random.uniform(-delta, delta, size=(N, 3))

        #   compute safe controls for all N states *vectorised*
        psi_p    = X_pert[:, 2]                        # (N,)
        cos_p    = np.cos(psi_p);  sin_p = np.sin(psi_p)
        Lg_v_p   = dhdx*cos_p + dhdy*sin_p             # (N,)
        g_vec_p  = np.stack([Lg_v_p, np.zeros_like(Lg_v_p)], axis=1)
        g_norm_p = np.linalg.norm(g_vec_p, axis=1) + 1e-12

        rho_p    = (-Lf_h - alpha_hs
                    + gammas1_flat[None, :]*g_norm_p[:, None]
                    + gammas2_flat[None, :]*(g_norm_p**2)[:, None])
        proj_p   = (g_vec_p @ u_nom)[:, None]          # (N,1)
        gain_p   = np.maximum(rho_p - proj_p, 0.0) / (g_norm_p**2)[:,None]
        U_pert   = u_nom + gain_p[..., None]*g_vec_p[:,None,:]   # (N,3600,2)

        diff     = U_nom[None, :, :] - U_pert          # (N,3600,2)
        sigma_max= np.linalg.norm(diff, axis=2).max(axis=0)      # (3600,)

        # cost     = np.maximum(delta*sigma_max - gammas1_flat, 0.0) \
        #            / (2.0*gammas2_flat + 1e-9)
        cost = np.maximum(delta*sigma_max - gammas1_flat, 0.0) / (2.0*gammas2_flat + 1e-9)
        cost[~np.isfinite(cost)] = 1e6
        best     = np.nanargmin(cost)

        self.k_1 = 1 * float(gammas1_flat[best])
        self.k_2 = 1 * float(gammas2_flat[best])

        # for gamma1 in gamma_vals:
        #     for gamma2 in gamma_vals:
        #         sigma_max = 0
        #         for _ in range(N):
        #             # small random perturbations within delta
        #             perturbation = np.random.uniform(-delta, delta, size=3)
        #             perturbed_state = state_nominal + perturbation

        #             # nominal and perturbed control inputs
        #             nominal_control = self.k_des(state_nominal.reshape(3, 1), SSF.counter * self.dt)
        #             perturbed_control = self.k_des(perturbed_state.reshape(3, 1), SSF.counter * self.dt)

        #             u_nom = np.array([nominal_control["linear_cmd"], nominal_control["angular_cmd"]])
        #             u_pert = np.array([perturbed_control["linear_cmd"], perturbed_control["angular_cmd"]])

        #             control_diff_norm = np.linalg.norm(u_nom - u_pert)
        #             sigma_max = max(sigma_max, control_diff_norm)

        #         # cost function from Eq (16)
        #         cost = max(delta * sigma_max - gamma1, 0) / (2 * gamma2 + 1e-6)  # Avoid division by zero

        #         if cost < best_cost:
        #             best_cost = cost
        #             best_gamma1, best_gamma2 = gamma1, gamma2

        # solution_time = time.time() - start_time
        # print(f"Grid search time: {solution_time:.4f} s")

        # if solution_time > self.dt:
        #     rospy.logwarn(f"Optimization took {solution_time:.4f}s, exceeding sampling interval!")

        # self.k_1 = best_gamma1
        # self.k_2 = best_gamma2
        print(f"Optimized gamma1: {self.k_1}, gamma2: {self.k_2}")


        def rob_term(k1, k2, Lg_vec):
            nrm = np.linalg.norm(Lg_vec)
            # w   = np.exp(self.k_tune * h_val)
            return k1 * nrm  + (k2 **2 ) * (nrm**2)  # we use gamma_2^2 as in the paper
        
        # tunable robust CBF term
        self.tunable.value = rob_term(self.k_1, self.k_2, np.array([self.conspar.value, self.conspar_ang.value]))

        print(  # f"alpha {self.alpha:6.2f} -> {alpha_safe:6.2}, "
            # f"Roll: {phi * 180 / 3.1415926:6.2f}, "
            # f"Pitch: {theta * 180 / 3.1415926:6.2f}, "
            f"Yaw: {psi * 180 / 3.1415926 :6.2f},"
            f"x: {self.pose_xi.value}, "
            f"y: {self.pose_yi.value}")

        alpha_safe = self.alpha

        # Solve QP
        self.prob.solve(solver="OSQP")
        print(self.prob.status)

        # if self.prob.status == 'infeasible' and u_nominal[0] == 0 and u_nominal[1] == 0:
        #     print('infeasible since Lgh(x) = 0')

        linear_safe = self.track_safe_acc_lin.value
        angular_safe = self.track_safe_acc_ang.value

        print("h_1(x) = ", self.cbf_alpha.value / self.alpha)
        # print("h_obs1(x) = ", h_obs1)
        # print("h_obs2(x) = ", h_obs2)

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
            "gamma_1": self.k_1,
            "gamma_2": self.k_2,
            "delta": delta,
            "h1": h1,
            "u_baseline1": self.nominal_linear.value,
            "u_baseline2": self.nominal_angular.value,
            "u_filtered1": u_safe[0],
            "u_filtered2": u_safe[1],
        }

        return u_safe
