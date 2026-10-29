function p = dryden_params(V, h, intensity, sigma_high)
%DRYDEN_PARAMS  Dryden scale lengths and intensities (MIL-F-8785C / MIL-HDBK-1797).
%   p = dryden_params(V, h, intensity, sigma_high)
%   V [m/s], h [m], intensity = 'light' | 'moderate' | 'severe' (W20 = 15/30/45 kt).
%   sigma_high [m/s] is required above 1000 ft: read it from the handbook's
%   intensity-vs-altitude chart for the exceedance probability you need.
%   Returns p.V, p.h, p.L = [Lu Lv Lw] (m), p.sigma = [su sv sw] (m/s).
if nargin < 3 || isempty(intensity), intensity = 'moderate'; end
if nargin < 4, sigma_high = []; end
ft = 0.3048; kt = 0.514444;
switch lower(intensity)
    case 'light',    W20 = 15*kt;
    case 'moderate', W20 = 30*kt;
    case 'severe',   W20 = 45*kt;
    otherwise, error('intensity must be light, moderate or severe');
end
if V <= 0, error('V must be positive'); end
hf = max(h/ft, 10);
if hf <= 1000
    [L, s] = low_alt(hf, W20, ft);
else
    if isempty(sigma_high)
        error('sigma_high is required above 1000 ft (MIL-HDBK-1797, intensity vs altitude)');
    end
    Lhi = 1750*ft*[1 1 1]; shi = sigma_high*[1 1 1];
    if hf >= 2000
        L = Lhi; s = shi;
    else
        [L1, s1] = low_alt(1000, W20, ft); f = (hf - 1000)/1000;
        L = L1 + f*(Lhi - L1); s = s1 + f*(shi - s1);
    end
end
p = struct('V', V, 'h', h, 'L', L, 'sigma', s);
end

function [L, s] = low_alt(hf, W20, ft)
den = 0.177 + 0.000823*hf;
Lu = hf/den^1.2*ft; Lw = hf*ft;
sw = 0.1*W20; su = sw/den^0.4;
L = [Lu Lu Lw]; s = [su su sw];
end
