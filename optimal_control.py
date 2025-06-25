import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

dt        = 0.1          
T_horizon = 40.0         
N         = int(T_horizon / dt)

# box bounds
xmin, xmax = -1.0, 9.0
ymin, ymax = -1.5, 0.8   

# obstacles 
obstacles = np.array([[2.5, 0.0, 1.0],
                      [6.9, 0.0, 1.0]])

# dynamic / cost parameters
v_max = 1.5
w_max = 1.5
Q = 1.5 * np.diag([1.0, 1.0, 1.5])
R = np.diag([1.0, 1.0])

# reference trajectory parameters
v_ref   = 0.25
y_mag   = 1.5
c_freq  = 2.9 * np.pi / 6
y_shift = -0.5

def ref_state(t):
    x_d = v_ref * t
    y_d = y_mag*np.sin(c_freq * x_d) + y_shift
    theta_d = np.arctan2(y_mag * c_freq * np.cos(c_freq * x_d), v_ref)
    return np.array([x_d, y_d, theta_d])

def step(state, control):
    v, w = control
    x, y, theta = state
    return np.array([
        x + v*np.cos(theta)*dt,
        y + v*np.sin(theta)*dt,
        theta + w*dt
    ])

# objective and constraints (vectorised trajectory rollout) 
def simulate_traj(u):
    """Return trajectory array (N+1,3) given control sequence u(N,2)."""
    x = np.zeros((N+1, 3))
    x[0] = np.array([0.0, y_shift, 0.0])   
    for k in range(N):
        x[k+1] = step(x[k], u[k])
    # wrap heading into (-pi,pi]
    x[:,2] = (x[:,2] + np.pi) % (2*np.pi) - np.pi
    return x

def objective(u_flat):
    u = u_flat.reshape(N,2)
    x = simulate_traj(u)
    t = np.arange(N+1)*dt
    ref = np.array([ref_state(tt) for tt in t])
    e = x - ref
    J = np.sum((e @ Q * e).sum(axis=1)) + np.sum((u @ R * u).sum(axis=1))
    return J

def ineq(u_flat):
    u = u_flat.reshape(N,2)
    x = simulate_traj(u)
    obs_c = np.min(np.hypot(x[:,0,None]-obstacles[:,0],
                            x[:,1,None]-obstacles[:,1]) - obstacles[:,2],
                   axis=1)
    ebox_c = np.vstack([x[:,0]-xmin,
                        xmax - x[:,0],
                        x[:,1]-ymin,
                        ymax - x[:,1]]).min(axis=0)  # shape (N+1,)
    return np.concatenate([obs_c[1:], ebox_c[1:]])

# initial guess: small forward speed, zero turn
u0 = np.zeros((N,2))
u0[:,0] = v_ref
u0_flat = u0.flatten()

bounds = [(-v_max, v_max), (-w_max, w_max)] * N
constr = {'type':'ineq', 'fun': ineq}

res = minimize(objective, u0_flat, method='SLSQP', bounds=bounds,
               constraints=constr, options={'maxiter':300, 'ftol':1e-3, 'disp':True})

print("success:", res.success, res.message)
u_opt = res.x.reshape(N,2)
x_traj = simulate_traj(u_opt)
t_grid = np.arange(N+1)*dt
ref_traj = np.array([ref_state(t) for t in t_grid])

plt.figure(figsize=(9,6))
plt.plot(ref_traj[:,0], ref_traj[:,1],'k--',label='reference')
plt.plot(x_traj[:,0], x_traj[:,1],'b',label='robot path')
for ox,oy,r in obstacles: plt.gca().add_patch(plt.Circle((ox,oy),r,
                                                         color='red',alpha=0.3))
plt.axhline(ymin,color='grey',ls=':'); plt.axhline(ymax,color='grey',ls=':')
plt.axvline(xmax,color='grey',ls=':'); plt.axvline(xmin,color='grey',ls=':')
plt.xlim(xmin-0.5,xmax+0.5); plt.ylim(ymin-0.5,ymax+0.5)
plt.xlabel('x [m]'); plt.ylabel('y [m]'); # plt.title('Trajectory (50 s horizon)')
plt.grid(); plt.legend(); plt.axis('equal')
plt.show()

fig,(a1,a2)=plt.subplots(2,1,figsize=(9,5),sharex=True)
a1.plot(t_grid[:-1],u_opt[:,0]); a1.set_ylabel('v [m/s]'); a1.grid(True)
a2.plot(t_grid[:-1],u_opt[:,1]); a2.set_ylabel('ω [rad/s]'); a2.set_xlabel('t [s]'); a2.grid(True)
# plt.suptitle('Optimal controls (dt=0.1, horizon=50 s)')
plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()
