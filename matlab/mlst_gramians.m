function [Wc, Wo, eps] = mlst_gramians(A, B, C, shift)
%MLST_GRAMIANS  Controllability/observability Gramians of (A - eps I, B, C).
%   eps defaults to 0 when A is Hurwitz, else max(real(eig)) + 5e-3
%   (marginal modes such as integrators have no finite Gramians).
if nargin < 4 || isempty(shift)
    mr = max(real(eig(A)));
    if mr < -1e-6, eps = 0; else, eps = mr + 5e-3; end
else
    eps = shift;
end
As = A - eps*eye(size(A,1));
Wc = sylvester(As, As', -B*B');  Wc = (Wc + Wc')/2;
Wo = sylvester(As', As, -C'*C);  Wo = (Wo + Wo')/2;
end
