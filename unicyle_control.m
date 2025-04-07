function unicyle_control()
    dt          = 0.1;  
    total_time  = 8;   
    steps       = round(total_time / dt);

    Kv   = 1.0;
    Kom  = 1.5;

    v_ref = 1.2;       
    c  = 2*pi / 6;  

    state = [0; 0; 0];  
    x_init = state(1);  

    time_data = 0:dt:total_time;  

    state_history   = zeros(3, steps+1);
    state_history(:,1) = state;
    control_history = zeros(2, steps);  
    
    for k = 1:steps
       
        t = (k-1) * dt;

        x_des = x_init + v_ref * t;
        y_des = sin(c * x_des);

        x    = state(1);
        y    = state(2);
        psi  = wrap_to_pi(state(3));

        dx = x_des - x;
        dy = y_des - y;
        dist = sqrt(dx^2 + dy^2);

        theta_d = atan2(dy, dx);
        theta_d = wrap_to_pi(theta_d);

        v = Kv * dist;

        heading_err = wrap_to_pi(psi - theta_d);
        w = -Kom * heading_err;

        state = unicycle_dynamics(state, [v; w], dt);

        state_history(:,k+1)   = state;
        control_history(:,k)   = [v; w];
    end

    figure;

    subplot(2,1,1); hold on; box on;

    plot(state_history(1,:), state_history(2,:), 'b-', ...
         'LineWidth',1.5, 'DisplayName','robot path');

    x_ref_vals = x_init + v_ref .* time_data;
    y_ref_vals = sin(c * x_ref_vals);
    plot(x_ref_vals, y_ref_vals, 'k--', 'LineWidth',1.5, 'DisplayName','reference');

    xlabel('x_p');
    ylabel('y_p');
    legend('Location','best');

    subplot(2,1,2); hold on; box on;

    control_time = 0:dt:(steps-1)*dt;

    plot(control_time, control_history(1,:), 'r-', 'LineWidth',1.5, 'DisplayName','v(t)');
    plot(control_time, control_history(2,:), 'b-', 'LineWidth',1.5, 'DisplayName','w(t)');

    xlabel('Time [s]');
    ylabel('v and w');
    legend('Location','best');
    % title('velocities');
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2*pi) - pi;
end

function new_state = unicycle_dynamics(state, control_input, dt)
    x     = state(1);
    y     = state(2);
    theta = wrap_to_pi(state(3));

    v     = control_input(1);
    omega = control_input(2);

    x_dot     = v * cos(theta);
    y_dot     = v * sin(theta);
    theta_dot = omega;

    x     = x + x_dot * dt;
    y     = y + y_dot * dt;
    theta = wrap_to_pi(theta + theta_dot * dt);

    new_state = [x; y; theta];
end
