import numpy as np
from typing import List
import itertools
import logging
from safety_filterRAL import SSF, get_h0
import time as timer
from types import SimpleNamespace
import matplotlib.pyplot as plt
import csv
import os
import pathlib, datetime

out_dir = pathlib.Path("data"); out_dir.mkdir(exist_ok=True)
stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cbf_csv = out_dir / f"cbf_traj_{stamp}.csv"

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(script_dir, 'poisson_safety_grid_2_5.csv')
file_path = os.path.join(script_dir, 'poisson_safety_grid_1_5.csv')


class UnicycleModel:
    '''
    Unicycle ODE of the form:
    
    model inputs - u, w
    states - x, y, theta
    measured outputs - x, y

    \dot{x} = u*cos(theta)
    \dot{y} = u*sin(theta)
    \dot{theta} = w
    '''

    def __init__(self,parameters):
        self.parameters = parameters
        

    # Runge-Kutta (4th-order)
    def rk4(self, rhs, t_span, s0):
        t0, tf = t_span
        dt = tf - t0
        s = s0
        k1 = rhs(t0, s)
        k2 = rhs(t0 + dt / 2, s + dt * k1 / 2)
        k3 = rhs(t0 + dt / 2, s + dt * k2 / 2)
        k4 = rhs(t0 + dt, s + dt * k3)
        s_next = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return [s, s_next]
    
    def forward_euler(self,rhs,t_span,s0):
        t0,tf = t_span
        dt = tf - t0
        s_next = s0 + rhs(t0,s0)*dt
        return [s0, s_next]

    def atan(self, y, x, y_b, x_b):
        theta_hat = np.arctan2(y, x)
        theta_low = np.zeros_like(theta_hat)
        theta_high = np.zeros_like(theta_hat)

        x_l = x - x_b
        x_h = x + x_b
        y_l = y - y_b
        y_h = y + y_b

        theta_low[(x_h>0) & (y_l>0)] = np.arctan2(y_l, x_h)[(x_h>0) & (y_l>0)]
        theta_low[(x_h<0) & (y_h>0)] = np.arctan2(y_h, x_h)[(x_h<0) & (y_h>0)]
        theta_low[(x_l<0) & (y_h<0)] = np.arctan2(y_h, x_l)[(x_l<0) & (y_h<0)]
        theta_low[(x_l>0) & (y_l<0)] = np.arctan2(y_l, x_l)[(x_l>0) & (y_l<0)]

        theta_high[(x_l>0) & (y_h>0)] = np.arctan2(y_h, x_l)[(x_l>0) & (y_h>0)]
        theta_high[(x_l<0) & (y_l>0)] = np.arctan2(y_l, x_l)[(x_l<0) & (y_l>0)]
        theta_high[(x_h<0) & (y_l<0)] = np.arctan2(y_l, x_h)[(x_h<0) & (y_l<0)]
        theta_high[(x_h>0) & (y_h<0)] = np.arctan2(y_h, x_h)[(x_h>0) & (y_h<0)]
        
        theta_low[(x_l<0) & (y_l<0) & (x_h>0) & (y_h>0)] = -10*np.pi
        theta_high[(x_l<0) & (y_l<0) & (x_h>0) & (y_h>0)] = 10*np.pi
        theta_high[theta_high<theta_hat] += 2*np.pi
        theta_low[theta_low>theta_hat] -= 2*np.pi

        return theta_hat, theta_low, theta_high


    def ode(self, x, u):
        '''
        RHS of ode
            x - state
            v - forward velocity
            w - angular velocity
        '''
        v, w = u[0], u[1]
        dxdt = np.zeros(3)
        dxdt[0] = v*np.cos(x[2])
        dxdt[1] = v*np.sin(x[2])
        dxdt[2] = w
        return dxdt
    
    def dt_dynamics(self,t,s,u):
        rhs = lambda t,x: self.ode(x,u)
        t_span = [t, t+self.parameters.sample_time]
        [s, s_next] = self.rk4(rhs,t_span,s)
        # [s, s_next] = self.forward_euler(rhs,t_span,s)
        s_next[2] = self.wrap_to_pi(s_next[2])
        return s_next

    # wrap to (-pi, pi]
    def wrap_to_pi(self,angle):
        wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        if wrapped == -np.pi: wrapped  = np.pi 
        return wrapped

class SingleIntegratorModel:
    '''
    Single integrator ODE of the form:
    
    model inputs - vx, vy
    states - x, y
    measured outputs - x, y

    \dot{x} = vx
    \dot{y} = vy
    '''

    def __init__(self,parameters):
        self.parameters = parameters
        

    # Runge-Kutta (4th-order)
    def rk4(self, rhs, t_span, s0):
        t0, tf = t_span
        dt = tf - t0
        s = s0
        k1 = rhs(t0, s)
        k2 = rhs(t0 + dt / 2, s + dt * k1 / 2)
        k3 = rhs(t0 + dt / 2, s + dt * k2 / 2)
        k4 = rhs(t0 + dt, s + dt * k3)
        s_next = s + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return [s, s_next]
    
    def forward_euler(self,rhs,t_span,s0):
        t0,tf = t_span
        dt = tf - t0
        s_next = s0 + rhs(t0,s0)*dt
        return [s0, s_next]


    def ode(self, x, u):
        '''
        RHS of ode
            x - state
            vx - x-axis velocity
            vy - y-axis velocity
        '''
        vx, vy = u[0], u[1]
        dxdt = np.zeros(2)
        dxdt[0] = vx
        dxdt[1] = vy
        return dxdt
    
    def dt_dynamics(self,t,s,u):
        rhs = lambda t,x: self.ode(x,u)
        t_span = [t, t+self.parameters.sample_time]
        [s, s_next] = self.rk4(rhs,t_span,s)
        # [s, s_next] = self.forward_euler(rhs,t_span,s)
        return s_next
    
# tunable parameters 
y_mag = 1.5
c = 2.9 * np.pi / 6 
v_ref = 0.25
delta_obs = 0.3
y_0_init = -0.35

obs_x  = np.array([2.5,  6.9])   
obs_y  = np.array([0.0,  0.0])  
obs_R  = np.array([1,  1])  

# parameters
control_type = 'unicycle'
if control_type=='unicycle':
    x0 = np.array([0,y_0_init,0])
if control_type == 'single_integrator':
    x0 = np.array([0,0])

total_time = 40
T = 0.02
use_filter = True

# close the loop
# sim with safety filter
state = x0
states = [state]
ref_state = x0
ref_states = [ref_state]

nom_controls = []
act_controls = []
controls_filtered = []

h_values = []
gamma1_values = []
gamma2_values = []

u_nom_uni1_values = []
u_nom_uni2_values = []

timesteps = int(total_time / T)

print('\n--------------------start closed-loop simulation------------------')
print(f'safety filter: {use_filter}, total iteration: {timesteps}')
print('------------------------------------------------------------------\n')

tic = timer.time()
input_bounds = np.array([[-5, 5], [-2, 2]])
model_param = SimpleNamespace(sample_time = T,input_bounds = input_bounds, integration_method = 'RK45')
unicyle = UnicycleModel(parameters=model_param)
single_inte = SingleIntegratorModel(parameters=model_param)

ssf_controller = SSF(v_ref = v_ref,delta_obs = delta_obs, obs_x = obs_x, obs_y = obs_y)
ssf_controller.x_init = x0

for i in range(timesteps):
    time = i*T

    # unicycle model #
    # generate nominal input in different cases
    # x_ref = reference_path(time,reference_param)
    state_2d = state.reshape(-1,1)
    # u_nom = ssf_controller.k_des(state_2d,time,y_mag = y_mag, c = c)
    # v_cmd, w_cmd, *_ = ssf_controller.k_des(state_2d, time, y_mag=y_mag, c=c)

    k = ssf_controller.k_des(state_2d, time, y_mag = y_mag, c = c)
    # k      = ssf_controller._k_safe(state_2d, time, 0, 0, t)

    v_cmd  = k["linear_cmd"]
    w_cmd  = k["angular_cmd"]
    u_nom  = [v_cmd, w_cmd]

    # u_nom = [v_cmd, w_cmd]

    # generate actual input in different cases
    if use_filter:
        # safety filter
        # control_actual = ssf_controller.control_ssf(state,u_nom, phi=0.0, psi=0.0, theta=0.0, dot_phi=0.0, dot_theta=0.0, dot_psi=0.0, ddot_psi=0.0, 
        #             ddot_theta =0.0, ddot_phi=0.0, acc_z=1.0, acc_y=0.0, dot_v=0.0)
        # for the safety filter
        control_actual = ssf_controller.control_ssf(state,u_nom)
    else:
        control_actual = u_nom
    
    state_next = unicyle.dt_dynamics(time,state,control_actual)

    # single integrator model #
    # state_2d = state.reshape(-1,1)
    # u_nom = ssf_controller.single_integrator_tracking_controller(state_2d,time,y_mag=y_mag, c= c)

    # if use_filter:
    #     control_actual = ssf_controller.single_integrator_safety_filter(state_2d,time,y_mag=y_mag, c= c)
    # else:
    #     control_actual = u_nom
    # state_next = single_inte.dt_dynamics(time,state,control_actual)

    # logging
    ref_state_x, ref_state_y = ssf_controller.xy_ref(state_2d,time,y_mag=y_mag, c= c)
    # ref_state = [ref_state_x,ref_state_y]
    h_val = ssf_controller.data_dict['h1'] 
    gamma1_val = ssf_controller.data_dict['gamma_1'] 
    gamma2_val = ssf_controller.data_dict['gamma_2'] 
    u_nom_uni1 = ssf_controller.data_dict['u_baseline1'] 
    u_nom_uni2 = ssf_controller.data_dict['u_baseline2'] 
    states.append(state_next)
    ref_states.append(ref_state)
    nom_controls.append(u_nom)
    act_controls.append(control_actual)
    h_values.append(float(h_val)) 
    gamma1_values.append(float(gamma1_val)) 
    gamma2_values.append(float(gamma2_val)) 

    u_nom_uni1_values.append(float(u_nom_uni1)) 
    u_nom_uni2_values.append(float(u_nom_uni2)) 

    # ground-true
    state = state_next

    if (i + 1) % (timesteps/10) == 0:
        toc = timer.time()
        print(f"\nProgress: {i + 1}/{timesteps} iterations completed ({(i + 1)/timesteps*100:.1f}%) Elapsed time: {toc-tic:.1f} seconds.")


# convert lists to numpy arrays 
states = np.array(states) 
ref_states = np.array(ref_states)  
nom_controls = np.array(nom_controls)  
act_controls = np.array(act_controls)  
controls_filtered = np.array(controls_filtered)  
h_values = np.array(h_values) 
gamma1_values = np.array(gamma1_values)  
gamma2_values = np.array(gamma2_values)  

u_nom_uni1_values = np.array(u_nom_uni1_values)  
u_nom_uni2_values = np.array(u_nom_uni2_values) 

time_axis = np.linspace(0, timesteps * T, timesteps)  

t_grid  = np.arange(states.shape[0]) * T           

traj_mat  = np.column_stack([t_grid,          
                             states])         

ctrl_pad  = np.vstack([act_controls,
                       act_controls[-1]])     
data_mat  = np.hstack([traj_mat, ctrl_pad])  

h_pad  = np.pad(h_values.astype(float),       (0, 1), mode='edge') 
g1_pad = np.pad(gamma1_values.astype(float),  (0, 1), mode='edge')  
g2_pad = np.pad(gamma2_values.astype(float),  (0, 1), mode='edge')  

data_mat = np.hstack([data_mat,
                      h_pad[:,  None],
                      g1_pad[:, None],
                      g2_pad[:, None]])

csv_file  = out_dir / f"cbf_log_{stamp}.csv"  
np.savetxt(csv_file,
           data_mat,
           delimiter=",",
           header="t,x,y,theta,v,omega,h,gamma1,gamma2",
           fmt="%.6f",
           comments="")

print(f"[INFO]  CBF log saved to: {csv_file.resolve()}")


def plt_levelset(func, grid_region,ax=None):
    if ax is None:
        ax = plt.gca()

    xlim = grid_region[0]
    ylim = grid_region[1]
    
    x_vals = np.linspace(xlim[0], xlim[1], 200)
    y_vals = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])
    
    # level set
    ax.contour(X, Y, Z, levels=[0], colors='r', linewidths=2, linestyles='dashed')

    ax.text(xlim[0] + 0.3* (xlim[1] - xlim[0]), 3, 'Safety Boundary', 
             color='r', fontsize=14, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


# Plot Trajectory
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(states[:, 0], states[:, 1], label='Robot Trajectory')

ax = plt.gca()
plt.ylim(-7,4)

x_limits = ax.get_xlim()
y_limits = ax.get_ylim()
# grid_region = [x_limits, y_limits]
grid_region = [[-1,10],[-2,2]]

# Read the CSV files
imax = 230.0 # Grid I Dimension (Negative Y-Direction)
jmax = 1000.0 # Grid J Dimension (Positive X-Direction)
# ds = 0.01 # Grid Resolution
with open(file_path, 'r') as file:
    reader = csv.reader(file)
    data = [[float(val) for val in row] for row in reader]

htable = np.reshape(data, (int(imax),int(jmax)))
print(get_h0(htable,3,0))

plt_levelset(lambda x,y: get_h0(htable,x,y)[0],grid_region)

plt.scatter(states[0, 0], states[0, 1], color='green', marker='o', label='Start')
plt.scatter(states[-1, 0], states[-1, 1], color='red', marker='x', label='End')

# reference traj
x_refs = np.linspace(0, 9, 500)
#y values
y_refs = y_mag * np.sin(c * x_refs) + y_0_init
plt.plot(x_refs, y_refs, label='Reference trajectory')

# obstacles
circle0 = plt.Circle((obs_x[0], obs_y[0]), obs_R[0], fill=False, edgecolor='blue', linewidth=2)
ax.add_patch(circle0)

circle1 = plt.Circle((obs_x[1], obs_y[1]), obs_R[1], fill=False, edgecolor='blue', linewidth=2)
ax.add_patch(circle1)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Unicycle Trajectory")
plt.legend()
plt.grid()

# Inputs
plt.subplot(1, 2, 2)
plt.plot(time_axis, u_nom_uni1_values, label='Nominal $u_1$')
plt.plot(time_axis, u_nom_uni2_values, label='Nominal $u_2$')
if use_filter:
    plt.plot(time_axis, act_controls[:, 0], label='Filtered $u_1$', linestyle='dashed')
    plt.plot(time_axis, act_controls[:, 1], label='Filtered $u_2$', linestyle='dashed')
plt.xlabel("Time (s)")
plt.ylabel("Control Inputs")
plt.title("Control Inputs vs Filtered Inputs")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# h-values
plt.figure(figsize=(8, 4))
plt.plot(time_axis, h_values, label='h-values')
plt.axhline(y=0, color='r', linestyle='--', label='h = 0 boundary')
plt.xlabel("Time (s)")
plt.ylabel("h-value")
plt.title("Safety Constraint h over Time")
plt.legend()
plt.grid()
plt.show()


# h-values
plt.figure(figsize=(8, 4))
plt.plot(time_axis, gamma1_values, label='gamma1')
plt.plot(time_axis, gamma2_values, label='gamma2', color='r')
plt.xlabel("Time (s)")
plt.ylabel("gamma-values")
plt.title("gamma1, 2 over Time")
plt.legend()
plt.grid()
plt.show()