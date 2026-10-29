function [A, B, C, D] = dryden_ss(p)
%DRYDEN_SS  Block-diagonal state space of the three Dryden filters (5 states).
%   Inputs: 3 white noises of intensity pi (see README). Outputs: [u_g v_g w_g] m/s.
%   H_u = su*sqrt(2Lu/(pi V)) / (1 + Lu/V s)
%   H_v = sv*sqrt(Lv/(pi V)) (1 + sqrt(3) Lv/V s) / (1 + Lv/V s)^2   (w alike)
V = p.V;
tau = p.L(1)/V; k = p.sigma(1)*sqrt(2*p.L(1)/(pi*V));
A1 = -1/tau; B1 = k/tau; C1 = 1;
[A2, B2, C2] = second(p.L(2), p.sigma(2), V);
[A3, B3, C3] = second(p.L(3), p.sigma(3), V);
A = blkdiag(A1, A2, A3); B = blkdiag(B1, B2, B3); C = blkdiag(C1, C2, C3); D = zeros(3);
end

function [A, B, C] = second(L, s, V)
tau = L/V; k = s*sqrt(L/(pi*V));
A = [0 1; -1/tau^2 -2/tau]; B = [0; 1]; C = k*[1/tau^2 sqrt(3)/tau];
end
