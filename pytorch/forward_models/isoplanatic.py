
import numpy as np

def apply_aberration(input,dr0,sig,speckle_flg,padsize):

    energy = np.sum(input)

    padvec = ((padsize,padsize),(padsize,padsize))
    input = np.pad(input,padvec,'edge')

    N = input.shape[0]
    D = 1*N
    r0 = D/dr0
    delta = 1

    # Define Aperture Function
    x = np.arange(-N/2, N/2) * delta
    X,Y = np.meshgrid(x,x)
    ap = circ(X,Y,D)
    ap = np.ones(ap.shape)

    # Apply Speckle Phase
    r = input
    if speckle_flg:
        g = np.sqrt(r/2)*np.random.randn(r.shape[0],r.shape[1]) + 1j*np.sqrt(r/2)*np.random.randn(r.shape[0],r.shape[1])
    else:
        g = np.sqrt(r/2)+1j*np.sqrt(r/2) 

    # Create Phase Screen
    phi = ft_phase_screen(r0, N, delta, 0, np.inf)
    phi = np.angle(np.exp(1j*phi))

    # Apply Phase Screen in Fourier Domain 
    G = ft2(g,1)
    s = ap*G*np.exp(1j*phi)
    y = ift2(s,1/N)

    # Additive Noise
    y = abs(y)**2 + sig*np.random.randn(y.shape[0],y.shape[1])

    # Prepare Outputs
    y = y[padsize:-padsize,padsize:-padsize]
    y = y/np.sum(y)*energy
    phi = phi+np.pi
    phi = 255*phi/(2*np.pi)

    return y,phi


def ft_phase_screen(r0, N, delta, f0, fm):
    # Setup the PSD
    delta_f = 1/(N*delta)
    fx = np.arange(-N/2,N/2)*delta_f
    # Frequency grid [1/m]
    fx,fy = np.meshgrid(fx,fx)
    #th = np.arctan2(fx,fy)
    f = np.sqrt(fx**2+fy**2) # Polar Grid
    f[N//2,N//2] = 1
    #fm = 5.92/l0/(2*np.pi)  # inner scale frequency [1/m]
    #f0 = 1/L0               # outer scale frequency [1/m]
    # modified von Karman atmospheric phase PSD
    PSD_phi = 0.023*r0**(-5/3)*np.exp(-(f/fm)**2)/(f**2+f0**2)**(11/6)
    PSD_phi[N//2,N//2] = 0
    # random draws of Fourier coefficients
    cn = (np.random.randn(N,N)+1j*np.random.randn(N,N))*np.sqrt(PSD_phi)*delta_f
    phz = np.real(ift2(cn,1))
    return phz

def ft2(g,delta):
    G = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))*delta**2
    return G

def ift2(G,delta_f):
    N = G.shape[0]
    g = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G)))*(N*delta_f)**2
    return g

def circ(x,y,D):
    r = np.sqrt(x**2+y**2)
    z = r < D/2
    z[r==D/2] = 0.5
    return z