%RUN_TESTS  Self-checks for the MATLAB/Octave version. Prints PASS/FAIL.
ok = true;
% 1) each filter reproduces sigma^2 with noise intensity pi
cases = {50, []; 300, []; 450, 1.5; 3000, 2.0};
lev = {'light', 'moderate', 'severe'};
for c = 1:size(cases,1)
    for i = 1:3
        p = dryden_params(100, cases{c,1}, lev{i}, cases{c,2});
        [A, B, C] = dryden_ss(p);
        r = lyap_rms(A, B, C);
        ok = dl_check(ok, max(abs(r(:)' - p.sigma)./p.sigma) < 1e-9, ...
            sprintf('filter sigma h=%g %s', cases{c,1}, lev{i}));
    end
end
% 2) high altitude needs sigma_high
try, dryden_params(200, 5000); ok = dl_check(ok, false, 'sigma_high required');
catch, ok = dl_check(ok, true, 'sigma_high required'); end
% 3) Monte Carlo vs Lyapunov on a generic plant
V = 80; A = [-1 1; -3 -1.2]; Bg = [0 0 -1/V; 0 0 -3/V];
p = dryden_params(V, 300, 'moderate');
[Aa, Ba, Ca] = dl_augment(A, Bg, eye(2), p);
rl = lyap_rms(Aa, Ba, Ca); mm = mc_rms(Aa, Ba, Ca, 0.01, 400, 16, 3);
ok = dl_check(ok, all(abs(mm - rl)./rl < 0.05), 'Monte Carlo within 5% of Lyapunov');
% 4) MLST: resonant peak, H2 and first-order Gramians
wn = 2; z = 0.2; A2 = [0 1; -wn^2 -2*z*wn]; B2 = [0; wn^2]; C2 = [1 0];
r = mlst_norms(A2, B2, C2, [5 50]); r0 = mlst_norms(A2, B2, C2, [1e-2 1e2]);
ok = dl_check(ok, abs(r.Hinf - 1/(2*z*sqrt(1-z^2)))/r.Hinf < 1e-5, 'Hinf resonant peak');
ok = dl_check(ok, abs(r.w_Hinf_band - 5) < 5e-3, 'Hinf band-limited search');
ok = dl_check(ok, abs(r.H2 - sqrt(wn/(4*z))) < 1e-9, 'H2 closed form');
[Wc, Wo] = mlst_gramians(-2, 3, 0.5);
ok = dl_check(ok, abs(Wc - 9/4) < 1e-12 && abs(Wo - 0.25/4) < 1e-12, 'first-order Gramians');
T = mlst_modal(A2, B2, C2);
ok = dl_check(ok, numel(T) == 1 && abs(T.zeta - z) < 1e-9, 'complex pair merged');
[~, ~, e] = mlst_gramians([0 1; 0 -1], [0; 1], [1 0]);
ok = dl_check(ok, e > 0, 'marginal mode shifted');
if ok, disp('ALL TESTS PASSED'); else, error('some tests failed'); end

