import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files (x86)\ffmpeg\bin'

import matplotlib.animation as anim

from scipy.sparse import spdiags

from skimage.io import imread
from skimage.transform import resize


class AnimatedGif:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, figsize=(6, 6), facecolor=(1, 1, 1))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
        self.ax.get_yaxis().set_visible(False)
        self.ax.get_xaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.images = []

    def add(self, image):
        ax = self.ax.imshow(image, cmap='binary_r')
        self.images.append([ax])

    def save(self, filename, fps=10):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(
            filename,
            fps=fps,
            savefig_kwargs={'facecolor': 'black'}
            # progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
        )


class GrayScott:
    """Class to solve Gray-Scott Reaction-Diffusion equation"""

    def __init__(self, N, Du, Dv, K):
        """

        :param N:   Matrix size
        :param Du:  Region of integration over U
        :param Dv:  Region of integration over V
        :param K:
        """

        self.N = N
        self.u = np.ones((N * N))
        self.v = np.zeros((N * N))
        self.L = self.laplacian()
        self.Du = Du
        self.Dv = Dv
        self.K = K

    def laplacian(self):
        """Construct a sparse matrix that applies the 5-point discretization"""
        N = self.N
        e = np.ones(N * N)
        e2 = ([1] * (N - 1) + [0]) * N
        e3 = ([0] + [1] * (N - 1)) * N
        A = spdiags([-4 * e, e2, e3, e, e], [0, -1, 1, -N, N], N ** 2, N ** 2)
        return A

    def initialise(self, seed=np.random.randint(0, 2**31)):
        """Setting up the initial condition"""
        N, N2, r = self.N, np.int(self.N / 2), 16
        np.random.seed(seed)
        self.u += 0.02 * np.random.random((N * N))
        self.v += 0.02 * np.random.random((N * N))

    def integrate(self, Nt, F):
        """Integrate the resulting system of equations using the Euler method

        :param Nt:  Number of integration steps.
        :param F:
        :return:
        """
        # evolve in time using Euler method
        u = self.u
        v = self.v
        L = self.L
        K = self.K
        Du = self.Du
        Dv = self.Dv

        for i in range(Nt):
            uvv = u * v * v
            u += (Du * L.dot(u) - uvv + F * (1 - u))
            v += (Dv * L.dot(v) + uvv - (F + K) * v)

        self.u = u
        self.v = v

    def config_plot(self):
        """Plotting business"""
        u = self.u
        v = self.v
        N = self.N

        f, ax = plt.subplots(1, figsize=(9, 9))
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

        f, ax = plt.subplots(1, figsize=(9, 9))
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


def draw_shape(img, x, y):
    xi = np.clip(x, 0, N - 1).astype(int)
    yi = np.clip(y, 0, N - 1).astype(int)
    img[yi, xi] = 1

    for yy in np.unique(yi):
        ymask = yi == yy
        xx = np.arange(min(xi[ymask]), max(xi[ymask]))
        img[yy, xx] = 1
    return img


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="file", default="rickandmorty.jpg",
                        help="File to turn into a diffusion gif")
    parser.add_argument("-o", "--output_name", dest="output_name", default="output.gif",
                        help="Output file name.")
    parser.add_argument("-n", "--num_steps", dest="num_steps", type=int, default=100)
    parser.add_argument("-d", "--dimensions", dest="dimensions", type=int, default=256,
                        help="Output image size in px.")
    parser.add_argument("--du", type=float, default=0.16,
                        help="")
    parser.add_argument("--dv", type=float, default=0.08,
                        help="")
    parser.add_argument("--F", type=float, default=0.060,
                        help="")
    parser.add_argument("--K", type=float, default=0.062,
                        help="")
    parser.add_argument("--Nt", type=float, default=3,
                        help="")
    parser.add_argument("--constant_rate", action="store_true",
                        help="")
    parser.add_argument("-s", "--seed", type=int, default=np.random.randint(0, 2**31),
                        help="")

    args = parser.parse_args()
    if not args.output_name.endswith(".gif"):
        args.output_name += ".gif"
    return args


if __name__ == "__main__":
    args = parse_args()
    N = args.dimensions

    Du, Dv, F, K = args.du, args.dv, args.F, args.K
    Nt = args.Nt
    num_steps = args.num_steps

    img = imread(args.file, as_gray=True).astype(np.float) / 255.
    img = resize(img, (N, N))
    img = img.ravel()
    img *= (1 / np.max(img) * 1.01)

    rdSolver = GrayScott(N, Du, Dv, K)
    rdSolver.initialise(args.seed)
    rdSolver.u += img * 0.5
    rdSolver.v += img * 0.2

    animation = AnimatedGif()
    if args.constant_rate:
        step_fn = lambda x: int(50 * Nt)
    else:
        step_fn = lambda x: int(x * Nt)

    for i in range(num_steps):
        F = 0.05 + 0.075 * i / num_steps
        FM = F - 0.03 * img
        # FM = F*np.ones((rdSolver.N,rdSolver.N))

        rdSolver.integrate(step_fn(i * Nt), FM)
        animation.add(rdSolver.v.reshape((rdSolver.N, rdSolver.N)))
        print(f"\rStep {i+1} out of {num_steps} complete. ", flush=True, end="")
    print()

    animation.save(args.output_name)
