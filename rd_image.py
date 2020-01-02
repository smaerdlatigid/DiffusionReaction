import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files (x86)\ffmpeg\bin'
import matplotlib.animation as anim

from scipy.sparse import spdiags 
from scipy.ndimage.morphology import binary_dilation

from skimage.io import imread
from skimage.transform import resize

from shapes import superEllipse

class AnimatedGif():
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, figsize=(6,6), facecolor=(1,1,1) )
        plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95, wspace=0.2, hspace=0.2)
        self.ax.get_yaxis().set_visible(False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.images = []
 
    def add(self, image):
        ax = self.ax.imshow(image,cmap='binary_r')
        self.images.append([ax])
 
    def save(self, filename, fps=10):
        import matplotlib.animation as anim

        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, fps=fps,
            savefig_kwargs={'facecolor':'black'}
            #progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
        )

class GrayScott():
    """Class to solve Gray-Scott Reaction-Diffusion equation"""
    def __init__(self, N):
        self.N = N
        self.u = np.ones((N, N))
        self.v = np.zeros((N, N))
    
    
    def laplacian(self):
        """Construct a sparse matrix that applies the 5-point discretization"""
        N = self.N
        e=np.ones(N**2)
        e2=([1]*(N-1)+[0])*N
        e3=([0]+[1]*(N-1))*N
        A=spdiags([-4*e,e2,e3,e,e],[0,-1,1,-N,N],N**2,N**2)
        return A


    def initialise(self):
        """Setting up the initial condition"""
        N, N2, r = self.N, np.int(self.N/2), 16
        
        self.u += 0.02*np.random.random((N,N))
        self.v += 0.02*np.random.random((N,N))

    
    def integrate(self, Nt, Du, Dv, F, K, L):
        """Integrate the resulting system of equations using the Euler method"""
        u = self.u.reshape((N*N))
        v = self.v.reshape((N*N))

        #evolve in time using Euler method
        for i in range(Nt):
            uvv = u*v*v
            u += (Du*L.dot(u) - uvv +  F *(1-u))
            v += (Dv*L.dot(v) + uvv - (F+K)*v  )
        
        self.u = u
        self.v = v
        
    
    def configPlot(self):
        """Plotting business"""
        u = self.u
        v = self.v
        N = self.N

        f,ax = plt.subplots(1, figsize=(9,9))
        ax.imshow(u.reshape((N, N)), cmap=plt.cm.binary)    
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axis('equal')
        plt.tight_layout()
        plt.show()

        f,ax = plt.subplots(1, figsize=(9,9))
        ax.imshow(v.reshape((N, N)), cmap=plt.cm.binary)    
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axis('equal')
        plt.tight_layout()
        plt.show()

def draw_shape(img,x,y):
    xi = np.clip(x, 0,N-1).astype(int)
    yi = np.clip(y, 0,N-1).astype(int)
    img[yi,xi] = 1

    for yy in np.unique(yi):
        ymask = yi == yy
        xx = np.arange(min(xi[ymask]), max(xi[ymask]))
        img[yy,xx] =1
    return img

if __name__ == "__main__":
    N = 256

    Du, Dv, F, K = 0.16, 0.08, 0.060, 0.062
    Nt = 3
    NSTEPS = 100
    anim = AnimatedGif()

    img = imread("DigitalDreams3.png",as_gray=True).astype(np.float)/255.
    img = resize(img, (N,N))
    img /= np.max(img)*1.01
    
    #img = imread("Lena512.png",as_gray=True).astype(np.float)/255.
    init = np.copy(img)

    rdSolver = GrayScott(N)
    L = rdSolver.laplacian()
    rdSolver.initialise()
    rdSolver.u += init*0.5
    rdSolver.v += init*0.2
    
    for i in range(NSTEPS):    
        print(i)
        F = 0.05 + 0.075*i/NSTEPS
        FM = F -  0.03*init
        #FM = F*np.ones((rdSolver.N,rdSolver.N))
        rdSolver.integrate( int(i* Nt), Du, Dv, FM.reshape(-1), K, L)
        anim.add(rdSolver.v.reshape((rdSolver.N,rdSolver.N)) )

    anim.save('test_picture.gif')
