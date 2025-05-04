# import rospy
import cvxpy as cp
import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

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
            v_limit=1.5,  
            w_limit=1.5,    
            dt=0.1,  # stamp real time
            alpha=1,  # class-K function for CBF
            v_ref = 0.3,
            delta_obs = 0.2,
            obs_x = np.array([3.0,  6.0]),
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

        self.c = 2 * np.pi / 6 # frequency parameter for the narrow passage
        self.b = 0.0       # phase shift for the narrow passage
        self.a = 0.0       # vertical offset for the narrow passage

        # lane keeping
        self.d = (3.03) / 2 - 0.4 # the robot body shoud stay in the set, a circle around the robot

        # obstacle-1  and obstacle-2
        # self.obs_x  = np.array([3.0,  6.0])   
        # self.obs_y  = np.array([0.0,  0.0])  
        self.obs_x  = obs_x
        self.obs_y  = obs_y
        self.obs_R  = np.array([0.65,  0.65])   # radius

        # tunable heading-weight 
        self.delta_obs = delta_obs

        # obtacle, 1
        self.conspar_lin_obs1   = cp.Parameter()      # for v
        self.conspar_ang_obs1   = cp.Parameter()      # for omega
        self.cbf_alpha_obs1     = cp.Parameter()
        self.tunable_obs1       = cp.Parameter()
        # obtacle, 2
        self.conspar_lin_obs2   = cp.Parameter()      
        self.conspar_ang_obs2   = cp.Parameter()      
        self.cbf_alpha_obs2     = cp.Parameter()
        self.tunable_obs2       = cp.Parameter()
        
        # velocity and acceleration limits
        self.v_limit = v_limit  
        self.w_limit = w_limit  
    
        self.dt = dt  # sampling time (fixed)

        # CBF paramters
        self.alpha = alpha  # cbf class-k function parameter

        # robustness parameters
        self.k_1 = 0 * 0.1
        self.k_2 = 0 * 0.1

        # self.k_tune = -0.5 # if is tunable
        self.k_tune = 0.0 # if is not tunable

        # T-ISSf-ICBF parameter
        self.tunable = cp.Parameter()
        # decision variables
        self.track_safe_acc_lin = cp.Variable()  # derivative of control inputs, dv
        self.track_safe_acc_ang = cp.Variable()  # derivative of control inputs, dw
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
        self.cbf_alpha = cp.Parameter()  

        # cost function
        cost = cp.square(self.track_safe_acc_lin - self.nominal_linear) + cp.square(
            self.track_safe_acc_ang - self.nominal_angular)
        # print(cost.is_dcp(dpp=True))  # is the cost Disciplined Parametrized Programming (DPP)
       
        # constraint function
        # nominal CBF
        # constr = [self.track_safe_acc_lin * self.conspar + self.cbf_alpha >= 0]

        # robust CBF, as Lh = 0 we have conspar = Lgh
        # constr = [self.track_safe_acc_lin * self.conspar - self.k_1 * cp.norm(self.conspar) - self.k_2 * cp.square(cp.norm(self.conspar)) 
        #           + self.cbf_alpha >= 0]
        
        # tunable robust CBF
        constr = [self.track_safe_acc_lin * self.conspar - self.tunable + self.cbf_alpha >= 0,
                  # obstacle-1 CBF (A₁ v + B₁ ω)
                    self.track_safe_acc_lin * self.conspar_lin_obs1 +
                    self.track_safe_acc_ang * self.conspar_ang_obs1
                    - self.tunable_obs1 + self.cbf_alpha_obs1 >= 0,
                    # obstacle-2 CBF (A₂ v + B₂ ω)
                    self.track_safe_acc_lin * self.conspar_lin_obs2 +
                    self.track_safe_acc_ang * self.conspar_ang_obs2
                    - self.tunable_obs2 + self.cbf_alpha_obs2 >= 0, 
                    cp.abs(self.track_safe_acc_lin) <= self.v_limit,
                    cp.abs(self.track_safe_acc_ang) <= self.w_limit]

        self.prob = cp.Problem(cp.Minimize(cost), constr)

        # to check if whether the problem is DCP and/or DPP
        print(self.prob.is_dcp(dpp=True))  # is the problem DPP, constraints should be affine in parameters
        print(self.prob.is_dcp())  # is the problem Disciplined Convex Programming (DCP)


    def k_des(self, state, t,y_mag=1.5,c = 3.0 * np.pi / 6):
            """
            state = [x, y, psi]^T
            We define a reference x_des(t) that moves forward in x at speed self.v_ref,
            and then compute y_des from the sinusoid.
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
            y_des = y_mag* np.sin(self.c * x_des)

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

            return linear, angular

    def control_ssf(self, state, u_nominal):
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
            self.pose_xi.value, self.pose_yi.value, psi = [0,0,0]
        else:
            self.pose_xi.value = x_temp - self.x_init  # x-axis in the world frame
            self.pose_yi.value = y_temp - self.y_init  # y-axis in the world frame
            # psi = psi_temp - self.psi_init
            psi = wrap_to_pi(psi_temp - self.psi_init)

        self.sinydx.value = np.sin(self.c * self.pose_xi.value + self.b)
        self.cosydx.value = np.cos(self.c * self.pose_xi.value + self.b)

        self.sin_psi.value = np.sin(psi)
        self.cos_psi.value = np.cos(psi)

        #  for the safety constraint d^2- (y - sin(cx)^2
        # self.conspar.value = - 2 * (self.pose_yi.value - self.sinydx.value + self.a) * (
        #         self.sin_psi.value - self.c * self.cos_psi.value * self.cosydx.value)
        
        # self.cbf_alpha.value = self.alpha * (
        #             (self.d)**2 - (self.pose_yi.value - (self.sinydx.value + self.a))**2 )
        
        #  for the safety constraint d^2- y^2
        self.conspar.value = - 2 * (self.pose_yi.value) * (self.sin_psi.value )
        
        self.cbf_alpha.value = self.alpha * ((self.d)**2 - (self.pose_yi.value)**2 )

        # record h values
        h1 = self.cbf_alpha.value / self.alpha

        # self.tunable.value = (
        #     self.k_1 * np.linalg.norm(self.conspar.value) * np.exp(-0.5 * h1) 
        #     + self.k_2 * np.exp(-0.5 * h1) * (np.linalg.norm(self.conspar.value)) ** 2
        # )

        h_obs1, A1, B1 = obstacle_cbf(
        self.pose_xi.value, self.pose_yi.value, psi,
        self.obs_x[0],      self.obs_y[0],      self.obs_R[0],
        self.delta_obs)
        self.conspar_lin_obs1.value = A1
        self.conspar_ang_obs1.value = B1
        self.cbf_alpha_obs1.value   = self.alpha * h_obs1

        h_obs2, A2, B2 = obstacle_cbf(
        self.pose_xi.value, self.pose_yi.value, psi,
        self.obs_x[1],      self.obs_y[1],      self.obs_R[1],
        self.delta_obs)
        self.conspar_lin_obs2.value = A2
        self.conspar_ang_obs2.value = B2
        self.cbf_alpha_obs2.value   = self.alpha * h_obs2

        def rob_term(k1, k2, Lg_vec, h_val):
            nrm = np.linalg.norm(Lg_vec)
            w   = np.exp(self.k_tune * h_val)
            return k1 * nrm * w + k2 * (nrm**2) * w
        
        # lane
        self.tunable.value = rob_term(self.k_1, self.k_2, np.array([self.conspar.value]), h1)
        
        # obstacle-1
        self.tunable_obs1.value = rob_term(self.k_1, self.k_2, np.array([A1, B1]), h_obs1)

        # obstacle-2
        self.tunable_obs2.value = rob_term(self.k_1, self.k_2, np.array([A2, B2]), h_obs2)

        self.pose_psi.value = psi
        self.sin_psi.value = np.sin(psi)
        self.cos_psi.value = np.cos(psi)

        # time = number_of_calls * dt
        t = SSF.counter * self.dt

        self.nominal_linear.value, self.nominal_angular.value = self.k_des(
            np.array([[self.pose_xi.value, self.pose_yi.value, self.pose_psi.value]]).T, t)

        # Solve QP
        self.prob.solve(solver="OSQP")
        print(self.prob.status)

        if self.prob.status == 'infeasible' and u_nominal[0] == 0 and u_nominal[1] == 0:
            print('infeasible since Lgh(x) = 0')

        linear_safe = self.track_safe_acc_lin.value
        angular_safe = self.track_safe_acc_ang.value

        print("h_1(x) = ", self.cbf_alpha.value / self.alpha)
        print("h_obs1(x) = ", h_obs1)
        print("h_obs2(x) = ", h_obs2)

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

        self.data_dict = {"h1": h1}

        return u_safe
