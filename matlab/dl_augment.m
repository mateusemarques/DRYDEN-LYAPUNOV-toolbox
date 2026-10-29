function [Aa, Ba, Ca] = dl_augment(A, Bg, C, p, Dg)
%DL_AUGMENT  Append Dryden filters to x' = A x + Bg g, y = C x + Dg g.
%   Bg (n x 3) and Dg (ny x 3) map gust velocities [u_g v_g w_g] (m/s).
if nargin < 5 || isempty(Dg), Dg = zeros(size(C,1), 3); end
[Af, Bf, Cf] = dryden_ss(p);
n = size(A,1); m = size(Af,1);
Aa = [A, Bg*Cf; zeros(m,n), Af];
Ba = [zeros(n,3); Bf];
Ca = [C, Dg*Cf];
end
