function [rms, P] = lyap_rms(Aa, Ba, Ca, W)
%LYAP_RMS  Steady-state output RMS from Aa P + P Aa' + Ba W Ba' = 0.
%   W defaults to pi*eye (Dryden convention). Uses sylvester (core MATLAB/Octave).
if nargin < 4 || isempty(W), W = pi*eye(size(Ba,2)); end
if isscalar(W), W = W*eye(size(Ba,2)); end
if max(real(eig(Aa))) >= 0, error('augmented system is not asymptotically stable'); end
P = sylvester(Aa, Aa', -Ba*W*Ba');
P = (P + P')/2;
rms = sqrt(diag(Ca*P*Ca'));
end
