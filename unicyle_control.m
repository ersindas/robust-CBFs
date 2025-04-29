function unicycle_control()
    dt          = 0.1;           % sample time
    total_time  = 40;            % [s]
    steps       = round(total_time / dt);

    Kv   = 1.0;                  % speed gain
    Kom  = 1.5;                  % heading gain

    v_ref = 0.3;                 % forward ref speed for x_des(t)
    c     = 2*pi/6;              % sinusoid frequency

    % safe-set corridor  h_lane = d^2 - y^2 >= 0
    d_lane = 1;            % => corridor |y| <= 0.6 m

    % two obstacles (same as ROS code)
    obs   = [ 3.0 0.0 0.4 ;      % [x y R] obstacle-1
              6.0 0.0 0.4 ];     % obstacle-2

    %% ---------------- initial state & data storage -------------------
    state = [0; 0; 0];          % start  2 m before first obstacle
    x_init = state(1);

    state_history   = zeros(3, steps+1);
    state_history(:,1) = state;
    control_history = zeros(2, steps);

    %% ---------------- simulation loop --------------------------------
    for k = 1:steps
        t = (k-1) * dt;

        % reference trajectory (straight + sinusoid)
        x_des = x_init + v_ref * t;
        y_des = 1.2 * sin(c * x_des);

        % current state
        x   = state(1);
        y   = state(2);
        psi = wrap_to_pi(state(3));

        % pure-pursuit style controller
        dx = x_des - x;  dy = y_des - y;
        dist     = hypot(dx,dy);
        theta_d  = wrap_to_pi(atan2(dy,dx));

        v = Kv  * dist;
        w = -Kom * wrap_to_pi(psi - theta_d);

        % integrate unicycle dynamics
        state = unicycle_dynamics(state, [v; w], dt);

        % log
        state_history(:,k+1) = state;
        control_history(:,k) = [v; w];
    end

    time = 0:dt:total_time;

    figure('Name','Trajectory & Control');
    subplot(2,1,1); hold on; box on; grid on;
    title('Safe set, obstacles and robot path');
    xlabel('\xi  (m)'); ylabel('\eta  (m)'); axis equal;

    % lane corridor (filled light-blue band)
    fill([0 12 12 0],[-d_lane -d_lane d_lane d_lane], ...
         [0.9 0.95 1],'EdgeColor','none','DisplayName','safe corridor');

    % corridor boundary dotted
    plot([0 12],[ d_lane d_lane],'k--','HandleVisibility','off');
    plot([0 12],[-d_lane -d_lane],'k--','HandleVisibility','off');

    % obstacles
    th = linspace(0,2*pi,100);
    for i = 1:size(obs,1)
        xc = obs(i,1) + obs(i,3)*cos(th);
        yc = obs(i,2) + obs(i,3)*sin(th);
        plot(xc,yc,'r','LineWidth',1.6,'DisplayName','obstacle');
        scatter(obs(i,1),obs(i,2),25,'k','filled','HandleVisibility','off');
    end

    % robot path
    plot(state_history(1,:), state_history(2,:), 'b-', 'LineWidth',1.6,...
         'DisplayName','robot path');

    % reference trajectory
    x_ref_vals = x_init + v_ref.*time;
    y_ref_vals = 1.2 * sin(c * x_ref_vals);
    plot(x_ref_vals, y_ref_vals, 'k--', 'LineWidth',1.2, ...
         'DisplayName','reference');

    legend('Location','NorthEast');

    subplot(2,1,2); hold on; box on; grid on;
    title('Desired control inputs (no safety filter)');
    control_time = 0:dt:(steps-1)*dt;
    plot(control_time, control_history(1,:), 'r-', 'LineWidth',1.5, ...
         'DisplayName','v(t)');
    plot(control_time, control_history(2,:), 'b-', 'LineWidth',1.5, ...
         'DisplayName','\omega(t)');
    xlabel('time  (s)'); ylabel('[m/s]  &  [rad/s]');
    legend('Location','best');
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

    x     = x + v*cos(theta)*dt;
    y     = y + v*sin(theta)*dt;
    theta = wrap_to_pi(theta + omega*dt);

    new_state = [x; y; theta];
end