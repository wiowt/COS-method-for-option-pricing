import numpy as np
import numpy.typing as npt
from typing import Tuple


class iCOS:
    """
    Implied COS implementation

    ...

    Methods
    ----------
    __init__(self, S: float, T: float, r: float)
        class initialization
    fit(self, K: float, C: float, P: float, N: int = 32)
        estimates parameters from option prices
    __calc_psi(self, x: npt.NDArray)
        auxiliary computations
    __calc_H(self, x: npt.NDArray)
        auxiliary computations
    __calc_factors(self, x: npt.NDArray)
        auxiliary computations
    call_price(self, x: npt.NDArray)
        computes call prices for a given array of strikes
    rnd(self, n: int = 1000)
        computes risk neutral density for a given number of points
    """

    def __init__(self, S: float, T: float, r: float) -> None:
        """
        :param S: initial spot price
        :param T: time to maturity
        :param r: risk-neutral rate
        """
        self.S = S
        self.T = T
        self.r = r
        self.F = S * np.exp(r * T)


    def fit(self, K: npt.NDArray, C: npt.NDArray, P: npt.NDArray, N: int = 32) -> None:
        """
        :param K: np.array of strikes
        :param C: np.array of call prices
        :param P: np.array of put prices
        """
        self.C = C
        self.P = P

        self.O = (K > self.F) * C + (K <= self.F) * P
        
        self.alpha = K.min()
        self.beta = K.max()

        self.N = N
        self.m = np.arange(self.N)
        self.u = self.m * np.pi / np.log(self.beta / self.alpha)

        psi = self.__calc_psi(K)

        self.D = np.sum((K[1:] - K[:-1]) * (self.O[1:] * psi[:, 1:] + self.O[:-1] * psi[:, :-1]) / 2, axis=1) + \
                    np.exp(-self.r * self.T) * np.cos(self.u * np.log(self.F / self.alpha))
        
        C_bar, Z_c, Z_p = self.__calc_factors(K)
        X = np.vstack((np.ones_like(Z_c), Z_c, Z_p))

        self.theta, self.theta_c, self.theta_p = np.linalg.inv(X @ X.T) @ X @ (C - C_bar - C[-1])
        
        

    
    def __calc_psi(self, x: npt.NDArray) -> npt.NDArray:
        psi = self.u[:, None] / x[None, :]**2 * (np.sin(self.u[:, None] * np.log(x[None, :] / self.alpha))
                                        - self.u[:, None] * np.cos(self.u[:, None] * np.log(x[None, :] / self.alpha)))
        return psi
    

    def __calc_H(self, x: npt.NDArray) -> npt.NDArray:
        H = np.zeros((self.N, len(x)))
        H[1:] = 2 * x[None, :] / (self.u[1:, None] * (1 + self.u[1:, None] ** 2) * np.log(self.beta / self.alpha)) * \
            ((-1)**self.m[1:, None] * self.u[1:, None] * self.beta / x[None, :] - \
            self.u[1:, None] * np.cos(self.u[1:, None] * np.log(self.alpha / x[None, :])) - \
            np.sin(self.u[1:, None] * np.log(self.alpha / x[None, :])))
        H[0] = 2 / np.log(self.beta / self.alpha) * x * (self.beta / x - np.log(self.beta / x) - 1)

        return H
    

    def __calc_factors(self, x: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        H = self.__calc_H(x)

        C_bar = H * self.D[:, None]
        C_bar[0] /= 2
        C_bar = C_bar.sum(0)

        Z_c = (-1)**self.m[:, None] * H
        Z_c[0] /= 2
        Z_c = Z_c.sum(0) + x - self.beta

        Z_p = -H
        Z_p[0] /= 2
        Z_p = Z_p.sum(0)

        return C_bar, Z_c, Z_p


    def call_price(self, x: npt.NDArray) -> npt.NDArray:
        """
        :param x: np.array of strikes
        """
        C_bar, Z_c, Z_p = self.__calc_factors(x)
        return C_bar + self.C[-1] + Z_c * self.theta_c + Z_p * self.theta_p + self.theta


    def rnd(self, n: int = 1000) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        :param n: number of points in the interval [log(alpha), log(beta)]
        """
        x = np.linspace(np.log(self.alpha), np.log(self.beta), n)
        v = 2 * np.exp(self.r * self.T) / np.log(self.beta / self.alpha)
        f = v * (self.D[:, None] + (-1)**self.m[:, None] * self.theta_c - self.theta_p) \
            * np.cos(self.u[:, None] * (x[None, :] - np.log(self.alpha)))
        f[0] /= 2
        return x, f.sum(0)
