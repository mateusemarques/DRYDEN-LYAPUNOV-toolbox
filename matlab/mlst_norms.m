function r = mlst_norms(A, B, C, band, n_grid, shift)
%MLST_NORMS  H2, Hinf (global and in a band) and Hankel singular values.
%   r = mlst_norms(A, B, C, band, n_grid, shift)
%   band [rad/s] = [w_lo w_hi] for the band-limited Hinf (default [0.5 10]).
%   Hinf is found on a log grid (n_grid points, default 2000) and refined with fminbnd.
if nargin < 4 || isempty(band), band = [0.5 10]; end
if nargin < 5 || isempty(n_grid), n_grid = 2000; end
if nargin < 6, shift = []; end
[Wc, Wo, eps] = mlst_gramians(A, B, C, shift);
As = A - eps*eye(size(A,1));
r.eps = eps;
r.H2 = sqrt(trace(C*Wc*C'));
r.hsv = sort(sqrt(abs(real(eig(Wc*Wo)))), 'descend');
[r.Hinf, r.w_Hinf] = peak(As, B, C, [1e-3 1e2], n_grid);
[r.Hinf_band, r.w_Hinf_band] = peak(As, B, C, band, n_grid);
end

function [pk, wp] = peak(A, B, C, band, n)
w = logspace(log10(band(1)), log10(band(2)), n);
s = arrayfun(@(x) smax(A, B, C, x), w);
[pk, k] = max(s); wp = w(k);
lo = w(max(k-1,1)); hi = w(min(k+1,n));
if hi > lo
    [x, f] = fminbnd(@(x) -smax(A, B, C, x), lo, hi);
    if -f > pk, pk = -f; wp = x; end
end
end

function s = smax(A, B, C, w)
s = max(svd(C*((1i*w*eye(size(A,1)) - A)\B)));
end
