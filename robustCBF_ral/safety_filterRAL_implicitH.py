# tests with the Poisson-based safety funtion
#!/usr/bin/env python3
# import rospy
import cvxpy as cp
import csv
import os
import numpy as np
# from std_msgs.msg import Float64

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

    imax = 220.0 # Grid I Dimension (Negative Y-Direction)
    jmax = 1000.0 # Grid J Dimension (Positive X-Direction)
    ds = 0.01 # Grid Resolution

    # Define X-Y Coordinate for Bottom Left Corner of Grid
    x0, y0 = -1.00, -1.10 

    # Fractional Index Corresponding to Current Position
    ir = (imax-1.0) - (y - y0) / ds
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

    # Compute Gradients
    i_eps = 1.0
    j_eps = 1.0
    ip = np.fmin(np.fmax(0.0, ic - i_eps), imax-1.0)
    im = np.fmin(np.fmax(0.0, ic + i_eps), imax-1.0)
    jp = np.fmin(np.fmax(0.0, jc + j_eps), jmax-1.0)
    jm = np.fmin(np.fmax(0.0, jc - j_eps), jmax-1.0)
    dhdx = (bilinear_interpolation(h, ic, jp) - bilinear_interpolation(h, ic, jm)) / ((jp-jm)*ds)
    dhdy = (bilinear_interpolation(h, ip, jc) - bilinear_interpolation(h, im, jc)) / ((im-ip)*ds)

    return h0, dhdx, dhdy


def obstacle_cbf(p_x, p_y, psi, o_x, o_y, R, delta):
    """
    returns:  h,   A (coeff of v),   B (coeff of omega)
     """
    dx, dy   = p_x - o_x, p_y - o_y
    D        = np.hypot(dx, dy)
    n_x, n_y = dx / D, dy / D
    c        = n_x * np.cos(psi) + n_y * np.sin(psi)          
    h        = D - R + delta * c
    A        = c + delta * (1 - c**2) / D                         
    B        = delta * (n_x * -np.sin(psi) + n_y * np.cos(psi))   
    return h, A, B


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
            h_battery=0.5,
            h_non_battery=0.5,
            ulimit_enable=True,
            v_limit=1.5,  
            w_limit=1.5,  
            robot_width=0.4,  
            robot_length=0.7,
            track_length=0.1,
            track_radius=0.1,
            cg_distance=0.2,  
            robot_mass=24,  
            # robot_inertia=np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]),
            robot_inertia=np.array([[0.63, 0, 0], [0, 1.11, 0], [0, 0, 1.52]]),
            dt=0.1,  # stamp real time
            alpha=1,  # class-K function for CBF
            sigma_v=1,
            delta_l=0.5,
            k_b=2,  # it should be greater than alpha
            b_h=2,  #
            tau=1.0,
            ksi_theta=0.1,
            v_ref = 0.3,
            delta_obs = 0.2,
            obs_x = None, 
            obs_y = None,
            data_dict=None

    ):
        """
        Limitations on tracks, linear and angular velocities (control inputs) are enforced
        """
        # robot geometry and mass
        if data_dict is None:
            data_dict = {"uncertainty_data_1": 0,
                         "uncertainty_data_2": 0,
                         "uncertainty_data_3": 0,
                         "uncertainty_data_4": 0,
                         "h1": 0,
                         "h2": 0,
                         "h3": 0,
                         "h4": 0,
                         "dh1": 0,
                         "dh2": 0,
                         "dh3": 0,
                         "dh4": 0,
                         "u_baseline1": 0,
                         "u_baseline2": 0,
                         "u_filtered1": 0,
                         "u_filtered2": 0,
                         }
        self.data_dict = data_dict
        self.robot_width = robot_width
        self.robot_length = robot_length
        self.track_length = track_length
        self.track_radius = track_radius
        self.cg_distance = cg_distance
        self.robot_mass = robot_mass
        self.robot_inertia = robot_inertia

        self.tip_risk = 0.0  # 0: no tip over risk, 1: tipped over
        self.tip_angle = 25.0  # tip over angle

        self.x_init = None
        self.y_init = None
        self.psi_init = None

        # self.xgoal = np.array([[0, -1.0]]).T
        # self.xO = np.array([[1.5, 0], [3, -1.5]]).T
        scale = 0.1
        self.Kv = 10 * scale
        self.Kom = 15 * scale

        # a forward speed to use for x_des(t)
        self.v_ref = v_ref

        self.c = 2 * np.pi / 6 # frequency parameter for the narrow passage
        self.b = 0.0       # phase shift for the narrow passage
        self.a = 0.0       # vertical offset for the narrow passage

        # # lane keeping
        # self.d = (3.03) / 2 - 0.4 # the robot body shoud stay in the set, a circle around the robot

        # # obstacle-1  and obstacle-2
        # self.obs_x  = np.array([3.0,  6.0])   
        # self.obs_y  = np.array([0.0,  0.0])   
        # self.obs_R  = np.array([0.25,  0.25])   # radius

        # tunable heading-weight 
        self.delta_obs = delta_obs

        # enable/disable safety filter and/or control input limits
        self.ulimit_enable = ulimit_enable
        # velocity and acceleration limits
        self.v_limit = v_limit  
        self.w_limit = w_limit  
       
        self.dt = dt  # sampling time (fixed)

        # CBF paramters
        self.alpha = alpha  # cbf class-k function parameter

        # robustness parameters
        self.k_1 = 1 * 0.1
        self.k_2 = 1 * 0.1

        # self.k_tune = -0.5 # if is tunable
        self.k_tune = 0.0 # if is not tunable

        # Read the CSV files
        imax = 220.0 # Grid I Dimension (Negative Y-Direction)
        jmax = 1000.0 # Grid J Dimension (Positive X-Direction)
        # ds = 0.01 # Grid Resolution

        script_dir = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(script_dir, 'poisson_safety_grid.csv')
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            # data = list(reader)
            data = [[float(val) for val in row] for row in reader]


        self.hvalues = np.reshape(data, (int(imax),int(jmax)))

        # Tunable-ISSf-ICBF parameter
        self.tunable = cp.Parameter()

        # uncertainty estimator parameters
        self.sigma_v = sigma_v
        self.delta_l = delta_l
        self.k_b = k_b
        self.b_h = b_h

        self.tau = tau
        self.ksi_theta = ksi_theta

        # increase values to make more conservative
        self.h_battery = np.clip(h_battery, a_min=0.5, a_max=2)  # ladder side 0.5<h_battery<2
        self.h_non_battery = np.clip(h_non_battery, a_min=0.3, a_max=2)  # non-battery 0.3<h_non_battery<2

        # decision variables
        self.track_safe_acc_lin = cp.Variable()  # v
        self.track_safe_acc_ang = cp.Variable()  # omega

        self.cons_constant1 = cp.Parameter()
        self.cons_constant2 = cp.Parameter()
        self.cons_constant3 = cp.Parameter()
        self.cons_constant4 = cp.Parameter()

        self.centrifugal = cp.Parameter()  # v*\omega

        # DOB-ICBF parameter
        self.unc_est_1 = cp.Parameter()  
        self.unc_est_2 = cp.Parameter()  
        self.unc_est_3 = cp.Parameter()  
        self.unc_est_4 = cp.Parameter()  #

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

        self.parameter_4eps = cp.Parameter()

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
       
        # # control input bounds
        # if self.ulimit_enable:
        #     constr.append(cp.abs(self.track_safe_acc_lin) <= self.v_limit)
        #     constr.append(cp.abs(self.track_safe_acc_ang) <= self.w_limit)

        self.prob = cp.Problem(cp.Minimize(cost), constr)

        # to check if whether the problem is DCP and/or DPP
        print(self.prob.is_dcp(dpp=True))  # is the problem DPP, constraints should be affine in parameters
        print(self.prob.is_dcp())  # is the problem Disciplined Convex Programming (DCP)

    def uncertainty_estimator(self, h_value, hdot_value, gamma, unc_est):
        gamma += self.k_b * (hdot_value + unc_est) * self.dt
        b_hat = self.k_b * h_value - gamma
        rob_term = 1 * b_hat - 0 * self.b_h / self.k_b

        return rob_term, gamma


    def k_des(self, state, t,y_mag=1.5, c= None,y_shift = 0.0):
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
        y_des = y_mag * np.sin(self.c * x_des + y_shift)

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

        return linear, angular, theta_d, theta_d_dot, ex, ey

    def control_ssf(self, state, u_nominal, phi, psi, theta, dot_phi, dot_theta, dot_psi, ddot_psi, ddot_theta,
                    ddot_phi, acc_z, acc_y, dot_v):
        #
        # u_nominal: [ v, w ]
        # states: x, y, \theta

        self.pose_xi.value = state[0]  # x-axis in the world frame
        self.pose_yi.value = state[1]  # y-axis in the world frame

        x_temp = state[0]
        y_temp = state[1]
        psi_temp = state[2]
        if SSF.counter == 0: # initial pose is always 0
            print(SSF.counter)
            self.x_init = x_temp
            self.y_init = y_temp
            self.psi_init = psi_temp
        else:
            self.pose_xi.value = x_temp - self.x_init  # x-axis in the world frame
            self.pose_yi.value = y_temp - self.y_init  # y-axis in the world frame
            # psi = psi_temp - self.psi_init
            psi = wrap_to_pi(psi_temp - self.psi_init)

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

        self.nominal_linear.value, self.nominal_angular.value, theta_des, theta_dot_des, ex, ey = self.k_des(
            np.array([[self.pose_xi.value, self.pose_yi.value, self.pose_psi.value]]).T, t)
        
        rho2  = ex**2 + ey**2 + 1e-8               
        u_ang = wrap_to_pi(theta_des - self.pose_psi.value)         
        
        h0, dhdx, dhdy = get_h0(self.hvalues, self.pose_xi.value, self.pose_yi.value)

        self.conspar.value = ((dhdx) * (self.cos_psi.value ) + (dhdy) * (self.sin_psi.value) 
                              - self.delta_obs * np.sin(u_ang) * (ey * self.cos_psi.value - ex * self.sin_psi.value) / rho2)

        # self.conspar.value = - 2 * (self.pose_yi.value) * (self.sin_psi.value )

        self.conspar_ang.value =  (self.delta_obs) * (np.sin(wrap_to_pi(theta_des - psi)))  

        self.conspar_timevarying.value = - (self.delta_obs) * theta_dot_des * (np.sin(wrap_to_pi(theta_des - psi)))

        self.cbf_alpha.value = self.alpha * (h0 - self.delta_obs * (1 - np.cos(wrap_to_pi(theta_des - psi))))  # 

        # self.cbf_alpha.value = self.alpha * ((self.d)**2 - (self.pose_yi.value)**2 )  # update this

        # record h values
        h1 = self.cbf_alpha.value / self.alpha

        def rob_term(k1, k2, Lg_vec, h_val):
            nrm = np.linalg.norm(Lg_vec)
            w   = np.exp(self.k_tune * h_val)
            return k1 * nrm * w + k2 * (nrm**2) * w
        
        # tunable robust CBF term
        self.tunable.value = rob_term(self.k_1, self.k_2, np.array([self.conspar.value, self.conspar_ang.value]), h1)

        print(  # f"alpha {self.alpha:6.2f} -> {alpha_safe:6.2}, "
            # f"Roll: {phi * 180 / 3.1415926:6.2f}, "
            # f"Pitch: {theta * 180 / 3.1415926:6.2f}, "
            f"Yaw: {psi * 180 / 3.1415926 :6.2f},"
            f"x: {self.pose_xi.value}, "
            f"y: {self.pose_yi.value}")

        g_B_x = np.cos(phi) * np.sin(theta - 3.1415926) * (-9.81)
        g_B_y = -np.sin(phi - 3.1415926) * (-9.81)
        g_B_z = np.cos(phi) * np.cos(theta) * (-9.81)

        alpha_safe = self.alpha

        # measurements for y_zmp
        c1 = -self.cg_distance / acc_z
        c2 = (self.track_length + self.robot_width) / 2
        c3 = (-self.cg_distance * self.robot_mass * g_B_y - self.robot_inertia[
            0, 0] * ddot_phi + self.robot_inertia[1, 1] *
              dot_theta * dot_psi - self.robot_inertia[2, 2] * dot_theta * dot_psi) / (acc_z * self.robot_mass)

        self.cons_constant1.value = c3 / c1 + c2 / c1
        self.cons_constant2.value = -c3 / c1 + c2 / c1

        # measurements for x_zmp
        c4 = self.robot_length / 2
        c5 = (-self.cg_distance * self.robot_mass * g_B_x + self.robot_inertia[
            1, 1] * ddot_theta + self.robot_inertia[1, 1] *
              dot_theta * dot_phi - self.robot_inertia[2, 2] * dot_theta * dot_phi) / (acc_z * self.robot_mass)

        self.cons_constant3.value = c5 / c1 + c4 / c1
        self.cons_constant4.value = -c5 / c1 + c4 / c1

        # if we want to use human operator:
        # self.nominal_linear.value, self.nominal_angular.value = u_nominal 

        self.centrifugal.value = self.nominal_linear.value * self.nominal_angular.value  # v*\omega

        self.nominal_linear_acc.value = -dot_v - g_B_x  # du_input is = (u_k - u_(k-1))/dt or acc_lin
        self.nominal_angular_acc.value = dot_psi        # du_input_w is = (u_k - u_(k-1))/dt or acc_ang


        # h for the tipover, not for the position
        h2 = -self.nominal_linear.value * self.nominal_angular.value + self.cons_constant2.value
        h3 = self.nominal_linear_acc.value + self.cons_constant3.value
        h4 = -self.nominal_linear_acc.value + self.cons_constant4.value

        # record \dot{h} values
        dh1 = (self.conspar.value * self.nominal_linear_acc.value)

        dh2 = (
            -self.nominal_linear.value * self.nominal_angular_acc.value -
            self.nominal_angular.value * self.nominal_linear_acc.value
        )
        dh3 = (0 + alpha_safe * self.nominal_linear_acc.value)
        dh4 = (0 - alpha_safe * self.nominal_linear_acc.value)

        if SSF.counter == 0:
            self.gamma1_init = self.k_b * self.pose_xi.value
            self.gamma2_init = self.k_b * self.pose_yi.value
            self.gamma3_init = self.k_b * psi
            self.gamma4_init = self.k_b * h4
            self.u_enc1_init = 0.0
            self.u_enc2_init = 0.0
            self.u_enc3_init = 0.0

        self.unc_est_1.value, gamma1 = self.uncertainty_estimator(
            self.pose_xi.value,
            self.cos_psi.value * self.nominal_linear.value,
            self.gamma1_init,
            self.u_enc1_init
        )
        self.unc_est_2.value, gamma2 = self.uncertainty_estimator(
            self.pose_yi.value,
            self.sin_psi.value * self.nominal_linear.value,
            self.gamma2_init,
            self.u_enc2_init
        )
        self.unc_est_3.value, gamma3 = self.uncertainty_estimator(
            self.pose_psi.value,
            self.nominal_angular.value,
            self.gamma3_init,
            self.u_enc3_init
        )
        self.unc_est_4.value, gamma4 = self.uncertainty_estimator(
            h4, dh4, self.gamma4_init, self.u_enc3_init
        )

        self.cons_pose1.value = (
            self.sin_psi.value * self.cos_psi.value * self.unc_est_1.value
            + (self.sin_psi.value ** 2) * self.unc_est_2.value
            + self.ksi_theta * self.cos_psi.value * self.unc_est_3.value
        )
        self.cons_pose2.value = (
            self.unc_est_2.value + self.ksi_theta * self.cos_psi.value * self.unc_est_3.value
        )
        self.parameter_4eps.value = 1 + (self.ksi_theta * self.cos_psi.value) ** 2

        self.gamma1_init = gamma1
        self.gamma2_init = gamma2
        self.gamma3_init = gamma3
        self.gamma4_init = gamma4

        self.u_enc1_init = self.unc_est_1.value
        self.u_enc2_init = self.unc_est_2.value
        self.u_enc3_init = self.unc_est_3.value

        # Solve QP
        self.prob.solve(solver="OSQP")
        print(self.prob.status)

        if self.prob.status == 'infeasible' and u_nominal[0] == 0 and u_nominal[1] == 0:
            print('infeasible since Lgh(x) = 0')

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
            "uncertainty_data_1": self.unc_est_1.value,
            "uncertainty_data_2": self.unc_est_2.value,
            "uncertainty_data_3": self.unc_est_3.value,
            "uncertainty_data_4": self.unc_est_4.value,
            "h1": h1,
            "h2": h2,
            "h3": h3,
            "h4": h4,
            "dh1": dh1,
            "dh2": dh2,
            "dh3": dh3,
            "dh4": dh4,
            "u_baseline1": self.nominal_linear.value,
            "u_baseline2": self.nominal_angular.value,
            "u_filtered1": u_safe[0],
            "u_filtered2": u_safe[1],
        }

        return u_safe

