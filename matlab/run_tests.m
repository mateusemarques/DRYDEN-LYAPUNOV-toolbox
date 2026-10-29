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
if ok, disp('ALL TESTS PASSED'); else, error('some tests failed'); end

