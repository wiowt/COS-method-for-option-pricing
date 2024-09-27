import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from scipy.special import gamma
from typing import Tuple


def _psi(a: float, b: float, c: float, d: float, k: int) -> npt.NDArray:
    u = np.arange(1, k) * np.pi / (b - a)
    res = np.zeros(k)
    res[1:] = (np.sin(u * (d - a)) - np.sin(u * (c - a))) / u
    res[0] = d - c
    return res


def _chi(a: float, b: float, c: float, d: float, k: int) -> npt.NDArray:
    u = np.arange(k) * np.pi / (b - a)
    res = (
          np.cos(u * (d - a)) * np.exp(d) \
          - np.cos(u * (c - a)) * np.exp(c) \
          + u * np.sin(u * (d - a)) * np.exp(d) \
          - u * np.sin(u * (c - a)) * np.exp(c)
          ) / (1 + u ** 2)
    return res



def _U_call(K: float, a: float, b: float, k: int):
    return 2 / (b - a) * K (_chi(a, b, 0, b, k) - _psi(a, b, 0, b, k))


def _U_put(K: float, a: float, b: float, k: int):
    return 2 / (b - a) * K (_chi(a, b, a, 0, k) + _psi(a, b, a, 0, k))


def _trancation_range(c1: float, c2: float, c4: float) -> Tuple[float, float]:
    a = c1 - 10 * np.sqrt(c2 + np.sqrt(c4))
    b = c1 + 10 * np.sqrt(c2 + np.sqrt(c4))
    return a, b


class PricingModel(ABC):
    """
    Abstract class for COS method models

    ...

    Methods
    ----------
    __init__(self)
        class initialization with main params (i.e. sigma for BGM)
    _char_func(self, w: npt.NDArray, r: float, T: float)
        computes characteristic function for ln(S_T)
    def _comulants(self, r: float, T: float)
        computes comulants (for better trancation)
    call_price(self, S: float, K: npt.NDArray, r: float, T: float, k: int = 32)
        computes call prices for a given array of strikes
    rnd(self, S: float, K: npt.NDArray, r: float, T: float, k: int = 32)
        computes risk neutral density for a given array of strikes
    """
    @abstractmethod
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def _char_func(self, w: npt.NDArray, r: float, T: float) -> npt.NDArray:
        pass

    @abstractmethod
    def _comulants(self, r: float, T: float) -> Tuple[float, float, float]:
        pass

    def call_price(self, S: float, K: npt.NDArray, r: float, T: float, k: int = 32) -> npt.NDArray:
        c1, c2, c4 = self._comulants(r, T)
        a, b = _trancation_range(c1, c2, c4)

        u = np.arange(k) * np.pi / (b - a)
        U = _U_call(a, b, k)
        x = np.log(S / K) 
        terms = np.real(self._char_func(u, r, T)[:, None] * np.exp(1.j * u[:, None] *(x[None, :] - a)) * U[:, None])
        terms[0] /= 2
        V = K * np.exp(-r * T) * terms.sum(0)
        return V

    def rnd(self, S: float, K: npt.NDArray, r: float, T: float, k: int = 32) -> npt.NDArray:
        c1, c2, c4 = self._comulants(r, T)
        a, b = _trancation_range(c1, c2, c4)

        u = np.arange(k) * np.pi / (b - a)
        x = np.log(S / K)

        F = 2 / (b - a) * np.real(self._char_func(u, r, T) * np.exp(-1.j * a * u))
        f = F[:, None] * np.cos((x[None, :] - a) * u[:, None])
        f[0] /= 2
        return f.sum(0)


class GBMModel(PricingModel):

    def __init__(self, sigma: float) -> None:
        """
        :param sigma: vol in BS model
        """
        self.sigma = sigma
        assert sigma > 0

    def _char_func(self, w: npt.NDArray, r: float, T: float) -> npt.NDArray:
        return np.exp(1.j * w * (r - 0.5*self.sigma**2) * T - 0.5 * w**2 * self.sigma**2 * T)

    def _comulants(self, r: float, T: float) -> Tuple[float, float, float]:
        c1 = r * T
        c2 = self.sigma**2 * T
        c4 = 0
        return c1, c2, c4


class CGMYModel(PricingModel):

    def __init__(self, C: float, G: float, M: float, Y: float, sigma: float = 0) -> None:
        """
        :param C: _
        :param G: _
        :param M: _
        :param Y: _
        :param sigma: volatility of brownian part in extended model
        """
        self.C = C
        self.G = G
        self.M = M
        self.Y = Y
        self.sigma = sigma


    def _char_func(self, w: npt.NDArray, r: float, T: float) -> npt.NDArray:
        drift_corr = - self.G * gamma(-self.Y) * \
                       ((self.M - 1.j * w) ** self.Y - self.M ** self.Y + \
                       (self.G + 1.j * w) ** self.Y - self.G ** self.Y) - \
                        0.5 * self.sigma**2

        return np.exp(1.j * (w + drift_corr) * r * T - 0.5 * w**2 * self.sigma**2 * T) * \
               np.exp(T * self.C * gamma(-self.Y) * \
               ((self.M - 1.j * w) ** self.Y - self.M ** self.Y + \
               (self.G + 1.j * w) ** self.Y - self.G ** self.Y))


    def _comulants(self, r: float, T: float) -> Tuple[float, float, float]:
        c1 = r * T + self.C * T * gamma(1 - self.Y) * \
             (self.M**(self.Y - 1) - self.G**(self.Y - 1))
        c2 = self.sigma**2 * T + self.C * T * gamma(2 - self.Y) * \
             (self.M**(self.Y - 2) + self.G**(self.Y - 2))
        c4 = self.C * T * gamma(4 - self.Y) * \
             (self.M**(self.Y - 4) + self.G**(self.Y - 4))
        return c1, c2, c4


class HestonModel(PricingModel):

    def __init__(self, kappa: float, theta: float,
                 dzeta: float, rho: float, v0: float) -> None:
        """
        Volatility process: dv_t = kappa*(theta - v_t) + dzeta*sqrt(v)dW_t
        :param v0: initial value for v_t
        :param rho: correlation between brownian motions
        """
        self.kappa = kappa
        self.theta = theta
        self.dzeta = dzeta
        self.rho = rho
        self.v0 = v0
        assert 2 * kappa * theta > dzeta**2


    def _char_func(self, w: npt.NDArray, r: float, T: float) -> npt.NDArray:
        D = np.sqrt((self.kappa - 1.j * self.rho * self.dzeta * w)**2 + \
        (w**2 + 1.j * w) * self.dzeta**2)
        A = (self.kappa - 1.j * self.rho * self.dzeta * w - D)
        G = A / (self.kappa - 1.j * self.rho * self.dzeta * w + D)
        B = (1 - G * np.exp(-D * T))
        

        return np.exp(1.j * w * r * T + \
               self.v0 / self.dzeta**2 * (1 - np.exp(-D * T)) / B * A) * \
               np.exp(self.kappa * self.theta / self.dzeta**2 * (T * A - 2 * np.log(B / (1 - G))))


    def _comulants(self, r: float, T: float) -> Tuple[float, float, float]:
        c1 = r * T + (1 - np.exp(-self.kappa * T)) * \
             (self.theta - self.v0) / (2 * self.kappa) - 0.5 * self.theta * T

        c2 = 1 / (8 * self.kappa**3) * (self.dzeta * T * self.kappa * np.exp(-self.kappa * T) * \
             (self.v0 - self.theta) * (8 * self.kappa * self.rho - 4 * self.dzeta) + \
             self.kappa * self.rho * self.dzeta * (1 - np.exp(-self.kappa * T)) * \
             (16 * self.theta - 8 * self.v0) + 2 * self.theta * self.kappa * T * \
             (-4 * self.kappa * self.rho * self.dzeta + self.dzeta**2 + 4 * self.kappa**2) + \
             self.dzeta**2 * ((self.theta - 2 * self.v0) * np.exp(-2 * self.kappa * T) + \
             self.theta * (6 * np.exp(-self.kappa * T) - 7) + 2 * self.v0) + \
             8 * self.kappa**2 * (self.v0 - self.theta) * (1 - np.exp(-self.kappa * T)))

        c4 = 0
        return c1, c2, c4
