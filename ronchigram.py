import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.ndimage import zoom


# calc probe shape: https://www.sciencedirect.com/science/article/pii/S0304399199001941?via%3Dihub
# Krivanek notation: https://www.sciencedirect.com/science/article/pii/S0304399199000133
# http://ronchigram.com/introduction_to_the_ronchigram_and_its_calculation_with_ronchigramcom.pdf
# https://www.sciencedirect.com/science/article/pii/S0304399111002129?via%3Dihub


class Aberrations():
    def __init__(self):
        self.n = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 4,
                           5, 5, 5, 5])  # aberration order
        # degree (rotational symmetry)
        self.m = np.array([0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6])
        # amp
        self.Cnm = np.array([-7e-9,  # defocus, O2
                             -.849e-9,  # 2-fold Stig, A2
                             1.38e-9,  # Axial Coma, P3
                             1.59e-9,  # 3-fold Stig, A3
                             195.8e-9,  # 3rd Order Spherical, O4
                             -118e-9,  # 3rd Order Axial Star, Q4
                             65.1e-9,  # 4-fold Stig, A4
                             2.23e-6,  # 4th Order Axial Coma, P5
                             -6.13e-6,  # 3-Lobe Aberration, R5,
                             -7.44e-6,  # 5-fold Stig, A5
                             -0.201e-3,  # 5th Order Spherical, O6
                             0.131e-3,  # 5th Order Axial Star
                             0.001e-3,  # 5th Order Rosette
                             0.223e-3,  # 6-fold Stig, A6
                             ])  # all in m

        # angle
        self.phinm = np.array([0,  # defocus
                               -13.56,  # 2-fold Stig
                               115.94,  # Axial Coma
                               6.86,  # 3-fold Stig
                               0.,  # 3rd Order Spherical
                               65.87,  # 3rd Order Axial Star
                               10.87,  # 4-fold Stig
                               -62.01,  # 4th Order Axial Coma
                               -50.62,  # 3-Lobe Aberration
                               10.46,  # 5-fold Stig
                               0.,  # 5th Order Spherical
                               0.,  # 5th Order Axial Star
                               0.,  # 5th Order Rosette
                               -2.71,  # 6-fold Stig
                               ])  # all in degrees
        self.phinm = np.radians(self.phinm)  # convert to radians


class Sample():
    def __init__(self):
        s = 1


class Ronchigram:
    def __init__(self, acc):
        self.acc = acc
        # Simulation conditions
        self.imdim = 256  # number of pixels in simulation
        self.simdim = 150e-3  # max radius to simulate in rad
        self.cl_rad = 26e-3

        # Sample Conditions
        self.sigma = 0.0065  # V-1nm-1
        self.V = 10.8  # V
        self.t = 10  # nm
        self.factor = 8

        max_alpha = self.simdim
        self.xy = np.linspace(-max_alpha, max_alpha, self.imdim)
        [xx, yy] = np.meshgrid(self.xy, self.xy, indexing='ij')
        self.rho = np.sqrt(xx ** 2 + yy ** 2)
        self.phi = np.arctan2(yy, xx)
        self.cl_apt = list()
        self.cl_apt = [[1 if j <= self.cl_rad else 0 for j in i] for i in self.rho]

        self.ronchigram = 0
        self.chi = 0
        self.probe = 0
        self.aberrations = Aberrations()

        self.calc_wav(acc)
        self.calc_chi()
        self.calc_probe()
        self.calc_ronchigram()

    def setup(self):
        # determine the calc space based on the im and sim dims
        max_alpha = self.simdim
        self.xy = np.linspace(-max_alpha, max_alpha, self.imdim)
        [xx, yy] = np.meshgrid(self.xy, self.xy, indexing='ij')
        self.rho = np.sqrt(xx ** 2 + yy ** 2)
        self.phi = np.arctan2(yy, xx)
        self.cl_apt = list()
        self.cl_apt = [[1 if j <= self.cl_rad else 0 for j in i] for i in self.rho]

    def rand_aberrations(self):
        # defocus zero
        self.aberrations.Cnm[0] = 0
        # A2
        self.aberrations.Cnm[1] = np.random.uniform(-50e-9, 50e-9)
        self.aberrations.phinm[1] = np.random.uniform(0, 360)
        # P3
        self.aberrations.Cnm[2] = np.random.uniform(-500e-9, 500e-9)
        self.aberrations.phinm[2] = np.random.uniform(0, 360)
        # A3
        self.aberrations.Cnm[3] = np.random.uniform(-500e-9, 500e-9)
        self.aberrations.phinm[3] = np.random.uniform(0, 360)
        # O4
        self.aberrations.Cnm[4] = np.random.uniform(-5e-6, 5e-6)
        # Q4
        self.aberrations.Cnm[5] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[5] = np.random.uniform(0, 360)
        # A4
        self.aberrations.Cnm[6] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[6] = np.random.uniform(0, 360)
        # P5
        self.aberrations.Cnm[7] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[7] = np.random.uniform(0, 360)
        # R5
        self.aberrations.Cnm[8] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[8] = np.random.uniform(0, 360)
        # A5
        self.aberrations.Cnm[9] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[9] = np.random.uniform(0, 360)
        # O6
        self.aberrations.Cnm[10] = np.random.uniform(-5e-6, 5e-6)
        # R6
        self.aberrations.Cnm[11] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[11] = np.random.uniform(0, 360)
        # A6
        self.aberrations.Cnm[12] = np.random.uniform(-5e-6, 5e-6)
        self.aberrations.phinm[12] = np.random.uniform(0, 360)

    def calc_wav(self, kev):
        """Returns the wavelength for a given accelerating voltage

        Parameters
        ----------
        kev : float
            Accelerating voltage in keV

        Returns
        -------
        float
            Wavelegth in m
        """
        self.wav = 1.23986e-9 / (np.sqrt(kev * (2 * 510.998 + kev)))

    def calc_chi(self):
        """ Returns the aberation function
        from https://www.sciencedirect.com/science/article/pii/S0304399111002129

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.cl_apt = [[1 if j <= self.cl_rad else 0 for j in i] for i in self.rho]

        numAbs = len(self.aberrations.n)
        chi = np.zeros([self.imdim, self.imdim])  # change to list to speed up?
        for k in range(0, numAbs, 1):
            n = self.aberrations.n[k] + 1
            m = self.aberrations.m[k]
            Cnm = self.aberrations.Cnm[k]
            phinm = self.aberrations.phinm[k]
            chi = chi + Cnm * np.cos(m * (self.phi - phinm)) * self.rho ** (n) / (n)
        self.chi = 2 / 10 * np.pi / self.wav * chi

    def calc_probe(self):
        """ Returns the probe function for a given aberation fn and CL Apt

        Parameters
        ----------
        chi - this aberration fn

        Returns
        -------
        the probe
        """
        exp = np.exp(-1j * self.chi) * self.cl_apt
        psi_p = ifft2(exp)
        self.probe = np.abs(ifftshift(psi_p)) ** 2

    def calc_ronchigram(self):
        exp = np.exp(-1j * self.chi)
        psi_r = ifft2(exp)
        psi_r = ifftshift(psi_r)

        rand = np.random.rand(int(self.imdim / self.factor), int(self.imdim / self.factor))
        # plt.imshow(rand)

        rand = zoom(rand, self.factor, order=0)
        # plt.imshow(rand)

        V_x = self.sigma * self.t * self.V * rand

        T_x = np.exp(-1j * V_x)
        I_q = fft2(psi_r * T_x)
        # I_q = fftshift(I_q)
        I_q = np.abs(I_q) ** 2
        self.ronchigram = I_q  # * self.cl_apt

    def plot_phase(self):
        e = self.simdim * 1e3
        fig, ax = plt.subplots(1, 1)
        ax.imshow(np.mod(self.chi, 2 * np.pi),
                  extent=[-e, e, -e, e]
                  )  # cmap=cm.bwr)
        plt.show()

    def plot_probe(self):
        e = self.imdim / self.simdim * self.wav * (1 / (np.pi))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(self.probe,
                  extent=[-e, e, -e, e],
                  )  # cmap=cm.bwr)
        plt.show()

    def plot_ronchigram(self):
        e = self.simdim * 1e3
        fig, ax = plt.subplots(1, 1)
        ax.imshow(self.ronchigram, extent=[-e, e, -e, e], cmap=cm.Greys)
        plt.show()


if __name__ == "__main__":
    # print(f'Wavelength = \t {calc_wav(acc):.4} m')
    ronch = Ronchigram(300)
    ronch.plot_phase()
    # plot_probe(calc_probe(calc_chi(aber)))
    ronch.plot_ronchigram()
