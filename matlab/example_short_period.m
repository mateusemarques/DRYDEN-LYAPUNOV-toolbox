%EXAMPLE_SHORT_PERIOD  Generic short-period aircraft in vertical Dryden turbulence.
%   Plant values are ILLUSTRATIVE (not any specific aircraft). Same case as
%   python/examples/short_period_example.py, so both versions can be compared.
V = 80; h = 300; g = 9.81;
Za = -1.0; Ma = -3.0; Mq = -1.2;
A  = [Za 1; Ma Mq];
Bg = [0 0 Za/V; 0 0 Ma/V];                 % columns: u_g v_g w_g
C  = [1 0; 0 1; -(V/g)*Za 0];              % alpha, q, nz
Dg = [0 0 0; 0 0 0; 0 0 -(V/g)*Za/V];
names = {'alpha [deg]', 'q [deg/s]', 'nz [g]'};
sc = [180/pi; 180/pi; 1];
lev = {'light', 'moderate', 'severe'};
fprintf('V = %g m/s, h = %g m\n', V, h);
for i = 1:3
    p = dryden_params(V, h, lev{i});
    [Aa, Ba, Ca] = dl_augment(A, Bg, C, p, Dg);
    rl = lyap_rms(Aa, Ba, Ca).*sc;
    [mm, ms] = mc_rms(Aa, Ba, Ca, 0.01, 600, 20, i);
    mm = mm.*sc; ms = ms.*sc;
    fprintf('\n%s: sigma_w = %.3f m/s, L_w = %.1f m\n', lev{i}, p.sigma(3), p.L(3));
    for j = 1:3
        fprintf('  %-12s Lyapunov %8.4f | Monte Carlo %8.4f +- %.4f | diff %+.1f%%\n', ...
            names{j}, rl(j), mm(j), ms(j), 100*(mm(j)-rl(j))/rl(j));
    end
end
