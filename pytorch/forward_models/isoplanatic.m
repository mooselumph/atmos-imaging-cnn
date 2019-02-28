function [y phi]=produce_data(in,seed,dr0,sig,speckle_flg,padsize);

energy = sum(in(:));
in=double(in);
rng(round(10000*seed));

padsize = double(padsize);


padvec = [padsize padsize];
in = padarray(in,padvec,'replicate');

N = length(in);
D = 1*N;
r0 = D/dr0;
delta = 1;

x = (-N/2 : N/2-1) *delta;
[X Y] = meshgrid(x);
ap = circ(X,Y,D);
ap = ones(size(ap));

r = in;
if speckle_flg ==1
    g = sqrt(r/2).*randn(size(r))+1i*sqrt(r/2).*randn(size(r));
else
    g = sqrt(r/2)+1i*sqrt(r/2);
end

phi = ft_phase_screen(r0, N, delta, inf, 0);

phi = angle(exp(1i*phi));

G = ft2(g,1);
s = ap.*G.*exp(1i*phi);

y = ift2(s,1/N);
y = abs(y).^2+sig*randn(size(y));

y = y(padsize+1:end-padsize,padsize+1:end-padsize);
y = y/sum(y(:))*energy;
phi = phi+pi;
phi = 255*phi/(2*pi);

end


function phz = ft_phase_screen(r0, N, delta, L0, l0)
% setup the PSD
del_f = 1/(N*delta);   % frequency grid spacing [1/m]
fx = (-N/2 : N/2-1) * del_f;
% frequency grid [1/m]
[fx fy] = meshgrid(fx);
[th f] = cart2pol(fx, fy);  % polar grid
fm = 5.92/l0/(2*pi); % inner scale frequency [1/m]
f0 = 1/L0;           % outer scale frequency [1/m]
% modified von Karman atmospheric phase PSD
PSD_phi = 0.023*r0^(-5/3) * exp(-(f/fm).^2) ...
    ./ (f.^2 + f0^2).^(11/6);
PSD_phi(N/2+1,N/2+1) = 0;
% random draws of Fourier coefficients
cn = (randn(N) + i*randn(N)) .* sqrt(PSD_phi)*del_f;
% synthesize the phase screen
phz = real(ift2(cn, 1));
end

function G = ft2(g, delta)
% function G = ft2(g, delta)
G = fftshift(fft2(fftshift(g))) * delta^2;
end

function g = ift2(G, delta_f)
N = size(G, 1);
g = ifftshift(ifft2(ifftshift(G))) * (N * delta_f)^2;
end

function z = circ(x, y, D)
r = sqrt(x.^2+y.^2);
z = double(r<D/2);
z(r==D/2) = 0.5;
end

