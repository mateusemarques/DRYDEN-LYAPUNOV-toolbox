%EXAMPLE_MLST  Open-loop disturbance susceptibility of a generic longitudinal model.
%   Same case as python/examples/mlst_example.py. Values are ILLUSTRATIVE.
V = 150; g = 9.81;
Xu = -0.015; Xa = 4.0; Zu = -0.0009; Za = -0.8; Mu = 0; Ma = -2.0; Mq = -0.9;
A = [Xu Xa 0 -g; Zu Za 1 0; Mu Ma Mq 0; 0 0 1 0];
B = [A(:,1), A(:,2)/V]; B(4,:) = 0;            % inputs: u_g, w_g (m/s)
C = [0 0 1 0; 0 0 0 1];                          % outputs: q, theta
T = mlst_modal(A, B, C, {'u','alpha','q','theta'});
fprintf('%6s %8s %7s %8s %8s %9s  dominant\n', 'mode', 'wn', 'zeta', 'Wc^', 'Wo^', 'index');
for i = 1:numel(T)
    if T(i).oscillatory, t = 'osc'; else, t = 'real'; end
    fprintf('%6s %8.4f %7.4f %8.2e %8.2e %9.3e  %s\n', t, T(i).wn, T(i).zeta, ...
        T(i).wc_norm, T(i).wo_norm, T(i).idx, T(i).dominant);
end
r = mlst_norms(A, B, C, [0.5 10]);
fprintf('Hankel singular values: %s\n', mat2str(r.hsv', 4));
fprintf('H2 = %.4f\n', r.H2);
fprintf('Hinf (all)          = %.3f at %.4f rad/s\n', r.Hinf, r.w_Hinf);
fprintf('Hinf (0.5-10 rad/s) = %.3f at %.3f rad/s\n', r.Hinf_band, r.w_Hinf_band);
