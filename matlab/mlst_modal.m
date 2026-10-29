function T = mlst_modal(A, B, C, state_names, shift)
%MLST_MODAL  Modal susceptibility table (complex pairs merged, sorted by wn).
%   Fields per mode: eig, wn, zeta, wc, wo (modal Gramian diagonals),
%   wc_norm, wo_norm (normalised by the largest mode), idx = sqrt(wc*wo),
%   dominant (state with the largest participation factor).
if nargin < 4, state_names = {}; end
if nargin < 5, shift = []; end
[Wc, Wo] = mlst_gramians(A, B, C, shift);
[Phi, L] = eig(A); lam = diag(L); Pinv = inv(Phi);
wck = real(diag(Pinv*Wc*Pinv')); wok = real(diag(Phi'*Wo*Phi));
part = abs(Phi .* Pinv.');
tol = 1e-8; used = false(size(lam)); T = struct([]);
for k = 1:numel(lam)
    if used(k) || imag(lam(k)) < -tol, continue; end
    g = k;
    if abs(imag(lam(k))) > tol
        j = find(~used & abs(lam - conj(lam(k))) < 1e-6*max(1,abs(lam(k))));
        j = j(j ~= k); if ~isempty(j), g = [k j(1)]; end
    end
    used(g) = true;
    m.eig = lam(k); m.wn = abs(lam(k)); m.zeta = -real(lam(k))/abs(lam(k));
    m.oscillatory = abs(imag(lam(k))) > tol;
    m.wc = sum(wck(g)); m.wo = sum(wok(g));
    [~, d] = max(part(:,k));
    if isempty(state_names), m.dominant = d; else, m.dominant = state_names{d}; end
    if isempty(T), T = m; else, T(end+1) = m; end %#ok<AGROW>
end
wcm = max(abs([T.wc])); wom = max(abs([T.wo]));
for i = 1:numel(T)
    T(i).wc_norm = T(i).wc/wcm; T(i).wo_norm = T(i).wo/wom;
    T(i).idx = sqrt(abs(T(i).wc*T(i).wo));
end
[~, o] = sort([T.wn]); T = T(o);
end
