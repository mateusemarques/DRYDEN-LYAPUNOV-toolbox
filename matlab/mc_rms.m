function [mu, sd] = mc_rms(Aa, Ba, Ca, dt, t_final, n_runs, seed, W)
%MC_RMS  Monte Carlo RMS with exact (Van Loan) discretization of the noise.
%   Each run starts from the stationary covariance. Returns mean and std over runs.
if nargin < 4 || isempty(dt), dt = 0.01; end
if nargin < 5 || isempty(t_final), t_final = 600; end
if nargin < 6 || isempty(n_runs), n_runs = 20; end
if nargin < 7 || isempty(seed), seed = 0; end
if nargin < 8 || isempty(W), W = pi*eye(size(Ba,2)); end
if isscalar(W), W = W*eye(size(Ba,2)); end
randn('state', seed); %#ok<RAND>  (works in MATLAB and Octave)
n = size(Aa,1);
E = expm([-Aa, Ba*W*Ba'; zeros(n), Aa']*dt);
Phi = E(n+1:end, n+1:end)'; Qd = Phi*E(1:n, n+1:end); Qd = (Qd+Qd')/2;
[~, P0] = lyap_rms(Aa, Ba, Ca, W);   % exact sampling keeps the continuous stationary covariance
Lq = chol(Qd + 1e-30*eye(n), 'lower'); L0 = chol(P0 + 1e-30*eye(n), 'lower');
N = round(t_final/dt);
X = L0*randn(n, n_runs); acc = zeros(size(Ca,1), n_runs);
for k = 1:N
    Y = Ca*X; acc = acc + Y.^2;
    X = Phi*X + Lq*randn(n, n_runs);
end
r = sqrt(acc/N);
mu = mean(r, 2); sd = std(r, 0, 2);
end
