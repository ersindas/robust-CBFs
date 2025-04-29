clear; clc; close all;

alpha       = 1.0;
k1          = 0.0;
k2          = 0.0;

d_lane      = 1 - 0.4;                      
delta_obs   = 0.1;                          

obs = [ 0.0  0.0  0.4;      % [x  y  R] obstacle-1
        2.0  0.0  0.4 ];    %          obstacle-2

v_max = 1.5;   w_max = 1.5;                % control limits

% gains used in k_des (KV, Kpsi)   (10*0.1, 15*0.1)
Kv   = 1.0;     
Kpsi = 1.5;

wrapToPi = @(ang) mod(ang+pi,2*pi)-pi;

% h = d^2 - y^2
lane_h   = @(y)      d_lane^2 - y.^2;
lane_dh  = @(psi,y) -2*y.*sin(psi);                     
lane_Lg  = @(psi,y) -2*y.*sin(psi);                      

obst_cbf = @(px,py,psi,ox,oy,R) ...
    cbf_obstacle(px,py,psi,ox,oy,R,delta_obs);           

robust   = @(Lg,h) 0*k1*norm(Lg)*exp(-0.0*h) + ...
                   0*k2*(norm(Lg)^2)*exp(-0.0*h);

dt   = 0.1;          T = 100;            N = round(T/dt);
x    = zeros(3,N+1);  
u_d  = zeros(2,N+1);  
u_f  = zeros(2,N+1);  

x(:,1) = [-2; 0; 0];                      
x_goal = [3; 0.0];                          

H = zeros(3,N+1);     
for k = 1:N
    p  = x(1:2,k);   psi = x(3,k);

    err   = x_goal - p;
    dist  = norm(err);
    if dist < 0.1
        v_des = 0;  w_des = 0;
    else
        theta_d = atan2(err(2),err(1));
        v_des   = Kv   * dist;
        w_des   = -Kpsi*wrapToPi(psi - theta_d);
    end
    u_d(:,k) = [v_des; w_des];

    h_lane       = lane_h(p(2));
    A_lane       = lane_Lg(psi,p(2));      
    Lg_lane_vec  = A_lane;                 
    r_lane       = robust(Lg_lane_vec,h_lane);

    [h1,A1,B1]   = obst_cbf(p(1),p(2),psi,obs(1,1),obs(1,2),obs(1,3));
    Lg1_vec      = [A1 B1];
    r_obs1       = robust(Lg1_vec,h1);

    [h2,A2,B2]   = obst_cbf(p(1),p(2),psi,obs(2,1),obs(2,2),obs(2,3));
    Lg2_vec      = [A2 B2];
    r_obs2       = robust(Lg2_vec,h2);

    H(:,k) = [h_lane; h1; h2];

    Hqp = 2*diag([1 1]);  fqp = -2*u_d(:,k);
    Aqp = [ -A_lane 0 ;           
            -A1     -B1;          
            -A2     -B2];
    bqp = [ -(r_lane - alpha*h_lane);
            -(r_obs1 - alpha*h1);
            -(r_obs2 - alpha*h2)];

    A_bound = [ 1 0;-1 0; 0 1; 0 -1];
    b_bound = [ v_max; v_max; w_max; w_max];

    Aqp = [Aqp; A_bound];  bqp = [bqp; b_bound];

    opts = optimoptions('quadprog','Display','off');
    [u_safe,~,exitflag] = quadprog(Hqp,fqp,Aqp,bqp,[],[],[],[],[],opts);

    if exitflag ~= 1            
        u_safe = u_d(:,k);
    end
    u_f(:,k) = u_safe;

    v = u_safe(1);  w = u_safe(2);
    x(:,k+1) = x(:,k) + dt*[v*cos(psi);
                            v*sin(psi);
                            w ];
end
u_d(:,end) = u_d(:,end-1);  u_f(:,end) = u_f(:,end-1);
H(:,end)   = H(:,end-1);

figure; hold on; grid on; axis equal;
fill([-4 4 4 -4],[-d_lane -d_lane d_lane d_lane],[0.9 0.95 1],'EdgeColor','none');
plot([-4 4],[ d_lane d_lane],'k--');  plot([-4 4],[-d_lane -d_lane],'k--');

th = linspace(0,2*pi,80);
for i=1:2
    plot(obs(i,1)+obs(i,3)*cos(th), obs(i,2)+obs(i,3)*sin(th),'r','LineWidth',1.5);
    scatter(obs(i,1),obs(i,2),30,'r','filled');
end

plot(x(1,:),x(2,:),'b','LineWidth',2);
scatter(x(1,1),x(2,1),50,'g','filled'); text(x(1,1),x(2,1),'  start');
scatter(x_goal(1),x_goal(2),50,'k','filled'); text(x_goal(1),x_goal(2),'  goal');

xlabel('\xi  (m)'); ylabel('\eta (m)');
title('Safe set, obstacles and robot trajectory');
legend({'lane corridor','','obstacle','traj.'},'Location','NorthEast');

t = 0:dt:T;
figure;
subplot(2,1,1);
plot(t,u_d(1,:),'--',t,u_f(1,:),'LineWidth',1.2);
ylabel('v  (m/s)'); grid on; legend('desired','safe');
title('Control inputs');
subplot(2,1,2);
plot(t,u_d(2,:),'--',t,u_f(2,:),'LineWidth',1.2);
ylabel('\omega  (rad/s)'); xlabel('time (s)'); grid on; legend('desired','safe');

function [h,A,B] = cbf_obstacle(px,py,psi,ox,oy,R,delta)
dx = px-ox; dy = py-oy;
D  = max(hypot(dx,dy),0);        
nx = dx/D;  ny = dy/D;
c  = nx*cos(psi) + ny*sin(psi);     
h  = D - R + delta*c;
A  = c + delta*(1-c^2)/D;           
B  = delta*(nx*(-sin(psi)) + ny*cos(psi));   
end