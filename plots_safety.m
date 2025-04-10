clear all; clc; close all; warning off;

set(0,'defaultLineLineWidth',2);
set(0,'defaultLineMarkerSize',2);
set(0,'DefaultAxesLinewidth',2,'DefaultLineLineWidth',2);
set(0,'defaultAxesFontSize',16);
set(0,'defaultTextInterpreter','latex');
set(0,'defaultTextFontSize',14);
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLegendFontSize',14);

dataSets = [
    struct('controlFile','robot_control_2025-02-28-19-02-25.bag', ...
           'stateFile','robot_state_2025-02-28-19-02-25.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-41-02.bag', ...
           'stateFile','robot_state_2025-02-28-19-41-02.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-10-45.bag', ...
           'stateFile','robot_state_2025-02-28-19-10-45.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-15-59.bag', ...
           'stateFile','robot_state_2025-02-28-19-15-59.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-18-10.bag', ...
           'stateFile','robot_state_2025-02-28-19-18-10.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-19-41.bag', ...
           'stateFile','robot_state_2025-02-28-19-19-40.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-21-00.bag', ...
           'stateFile','robot_state_2025-02-28-19-20-59.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-23-46.bag', ...
           'stateFile','robot_state_2025-02-28-19-23-46.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-41-02.bag', ...
           'stateFile','robot_state_2025-02-28-19-41-02.bag'), ...
    struct('controlFile','robot_control_2025-02-28-19-57-12.bag', ...
           'stateFile','robot_state_2025-02-28-19-57-12.bag'), ...
    struct('controlFile','robot_control_2025-02-28-20-01-08.bag', ...
           'stateFile','robot_state_2025-02-28-20-01-08.bag')
];

u_max = 1.5;
Ts = 0.1;   % 10 Hz
% star = 1;

for iData = 1:length(dataSets)

    [~, controlFileName, ~] = fileparts(dataSets(iData).controlFile);
    disp(['processing dataset ' num2str(iData) ' -> ' controlFileName]);

    bag       = rosbag(dataSets(iData).controlFile);
    bagInfo   = rosbag('info', dataSets(iData).controlFile);
    bag_p     = rosbag(dataSets(iData).stateFile);
    bagInfo_p = rosbag('info', dataSets(iData).stateFile);

    bSel_des        = select(bag,'Topic','/cmd_vel/controller/filter_data');
    bSel            = select(bag,'Topic','/real/cmd_vel');
    msgStructs_des  = readMessages(bSel_des,'DataFormat','struct');
    msgStructs      = readMessages(bSel,'DataFormat','struct');

    h_value         = cellfun(@(m) double(m.H1), msgStructs_des);   

    bSel_real       = select(bag,'Topic','/cmd_vel/controller/drive');
    bSel_pose       = select(bag_p,'Topic','/state/pose');
    msgStructs_real = readMessages(bSel_real,'DataFormat','struct');
    msgStructs_pose = readMessages(bSel_pose,'DataFormat','struct');

    x_p = cellfun(@(m) double(m.Pose.Pose.Position.X), msgStructs_pose); 
    x_p = x_p - x_p(1); 
    y_p = cellfun(@(m) double(m.Pose.Pose.Position.Y), msgStructs_pose); 
    y_p = y_p - y_p(1);
    z_p = cellfun(@(m) double(m.Pose.Pose.Position.Z), msgStructs_pose); 
    z_p = z_p - z_p(1);

    q_1 = cellfun(@(m) double(m.Pose.Pose.Orientation.X), msgStructs_pose);
    q_2 = cellfun(@(m) double(m.Pose.Pose.Orientation.Y), msgStructs_pose);
    q_3 = cellfun(@(m) double(m.Pose.Pose.Orientation.Z), msgStructs_pose);
    q_4 = cellfun(@(m) double(m.Pose.Pose.Orientation.W), msgStructs_pose);

    theta = atan2(2.*(q_4.*q_3 + q_1.*q_2), 1 - 2.*(q_2.*q_2 + q_3.*q_3));
    theta = theta - theta(1);

    var_y = cellfun(@(m) double(m.Pose.Covariance(8)), msgStructs_pose);

    v_r = cellfun(@(m) double(m.ULinearVelocityFiltered),  msgStructs_des);
    w_r = cellfun(@(m) double(m.UAngularVelocityFiltered), msgStructs_des);

    time_s = zeros(length(x_p), 1);
    for k = 1:length(x_p) - 1
        time_s(k) = double(msgStructs_pose{k+1}.Header.Stamp.Sec) ...
                  + 1e-9*double(msgStructs_pose{k+1}.Header.Stamp.Nsec) ...
                  - ( double(msgStructs_pose{1}.Header.Stamp.Sec) ...
                  + 1e-9*double(msgStructs_pose{1}.Header.Stamp.Nsec) );
    end
    time_s(end) = time_s(end-1) + 0.001;  
    time_sn = time_s(1:end) - time_s(1);

    t = 0:Ts:(length(w_r)*Ts - Ts);

    c = 2*pi/6;
    b0 = 0;
    a = 0;
    d = 0.5;  
    h_values = d^2 - (y_p - sin(c*x_p + b0) - a).^2;   

    fig1 = figure('Position',[80 80 800 400]);
    hold on;
    plot(x_p, h_values, 'k', 'DisplayName','CBF $h(x,y)$');
    xlabel('$x_p \ [m]$');
    ylabel('$h(x,y)$');
    xlim([min(x_p), max(x_p)]);
    legend('Location','best');
    box on;
    hold off;

    saveas(fig1, [controlFileName '_CBF_h_vs_x.png']);

    fig2 = figure('Position',[100 100 800 600]);
    hold on;

    upper_bound = y_p + sqrt(var_y);
    lower_bound = y_p - sqrt(var_y);

    x_safe = linspace(min(x_p), max(x_p), 500);
    y_safe_upper = sin(c*x_safe + b0) + a + d;
    y_safe_lower = sin(c*x_safe + b0) + a - d;

    fill([x_safe, fliplr(x_safe)], ...
         [y_safe_upper, fliplr(y_safe_lower)], ...
         [0.8, 1.0, 0.8], 'EdgeColor','none', ...
         'DisplayName','safe set');

    y_safe_upper_d1 = sin(c*x_safe + b0) + a + 1;
    y_safe_lower_d1 = sin(c*x_safe + b0) + a - 1;
    plot(x_safe, y_safe_upper_d1, '--r', 'LineWidth',1.5, ...
         'DisplayName','safe boundary d=1');
    plot(x_safe, y_safe_lower_d1, '--r', 'LineWidth',1.5, ...
         'HandleVisibility','off');

    fill([x_p; flipud(x_p)], [upper_bound; flipud(lower_bound)], ...
         [0.7 0.7 0.7], 'EdgeColor','none', ...
         'DisplayName','uncertainty');

    x_ref_vals = linspace(min(x_p), max(x_p), 500);
    y_ref_vals = sin(c*x_ref_vals);
    plot(x_ref_vals, y_ref_vals, 'k--', 'LineWidth',1.5, ...
         'DisplayName','reference path');

    plot(x_p, y_p, 'b', 'LineWidth',2, 'DisplayName','robot path');

    text(mean(x_safe)+1.2, max(y_safe_upper_d1)-0.3, ...
         '$h(x,y) = d^2 - (y - \sin(cx + b) - a)^2$', ...
         'Interpreter','latex','FontSize',14,'HorizontalAlignment','center');

    xlabel('$x_p \ [m]$');
    ylabel('$y_p \ [m]$');
    legend('Location','northoutside','Orientation','horizontal','NumColumns',3);
    axis equal;
    box on;
    hold off;

    saveas(fig2, [controlFileName '_Safety_vs_Path.png']);

    disp(['Finished dataset ' num2str(iData) ': ' controlFileName]);
    fprintf('\n');
end