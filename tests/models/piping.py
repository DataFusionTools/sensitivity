import math


class SellmeijerRule:
    @staticmethod
    def calculate_critical_head(*argv):
        L, D, kappa, d70, rho_s, rho_l, eta, theta = (
            argv[0][0],
            argv[0][1],
            argv[0][2],
            argv[0][3],
            argv[0][4],
            argv[0][5],
            argv[0][6],
            argv[0][7],
        )
        Hc = (
            L
            * SellmeijerRule.calculate_Fg(L, D)
            * SellmeijerRule.calculate_Fs(d70, kappa, L)
            * SellmeijerRule.calculate_Fr(rho_s, rho_l, eta, theta)
        )
        return Hc

    @staticmethod
    def calculate_geometry_factor(L, D):
        if D / L == 1:
            return 1.08
        else:
            return 0.24 / ((D / L) ** 2.8 - 1)

    @staticmethod
    def calculate_Fg(L, D):
        return 0.91 * (D / L) ** SellmeijerRule.calculate_geometry_factor(L, D)

    @staticmethod
    def calculate_Fs(d_70, kappa, L):
        return d_70 / (kappa * 1e-10 * L) ** (1 / 3)

    @staticmethod
    def calculate_Fr(rho_s, rho_l, eta, theta):
        tan_theta = math.tan(theta * math.pi / 180)
        return (rho_s - rho_l) * eta * tan_theta / rho_l


def calculate_k_from_Ic(Ic):
    kappa = []
    for ic_value in Ic:
        if 1.0 < ic_value < 3.27:
            a = 0.952
            b = -3.04
        else:
            a = -4.52
            b = -1.37
        kappa.append(10 ** (a + b * ic_value))
    return kappa
