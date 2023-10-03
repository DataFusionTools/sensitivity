import os
import shutil
import pytest
import numpy as np

from models.piping import SellmeijerRule
from sensitivity.sensitivity import Sensitivity, SensitivityMethods


TOL = 1e-3


class TestSensitivity:
    @pytest.mark.unittest
    def test_morris(self):
        param_piping = {
            "names": ["L", "D", "kappa", "d70", "rho_s", "rho_l", "eta", "theta"],
            "bounds": [
                [5, 20],
                [10, 100],
                [1e-10, 1e-2],
                [2e-7, 2e-4],
                [2000, 3000],
                [900, 1100],
                [2.5e-2, 2.5],
                [30, 50],
            ],
        }

        s = Sensitivity()
        s.define_model(SellmeijerRule.calculate_critical_head, param_piping)
        s.run_sensitivity(SensitivityMethods.Morris)

        mu_start = [
            915.9254001644193,
            99.52218844798767,
            2796.866806146651,
            2150.2271814858223,
            660.8008008618482,
            323.1977472767453,
            1967.1992051873128,
            707.168950042373,
        ]

        assert all(np.abs(s.sensitivity_results["mu_star"] - mu_start) <= TOL)

        s.plot("./sensitivity", "Morris.png")
        assert os.path.isfile(r"./sensitivity/Morris.png")
        shutil.rmtree("./sensitivity")

    @pytest.mark.unittest
    def test_morris_2(self):

        params = {
            "names": ["a", "b"],
            "bounds": [[0, 1], [0, 1]],
        }

        fixed_args = 0.0

        s = Sensitivity()
        s.define_model(parabola, params, fixed_args)
        s.run_sensitivity(SensitivityMethods.Morris)

        assert all(np.abs(s.sensitivity_results["mu_star"] - np.array([1, 0])) <= TOL)

        s.plot("./sensitivity", "Morris.png")
        assert os.path.isfile(r"./sensitivity/Morris.png")
        shutil.rmtree("./sensitivity")

    @pytest.mark.unittest
    def test_sobol(self):
        param_piping = {
            "names": ["L", "D", "kappa", "d70", "rho_s", "rho_l", "eta", "theta"],
            "bounds": [
                [5, 20],
                [10, 100],
                [1e-10, 1e-2],
                [2e-7, 2e-4],
                [2000, 3000],
                [900, 1100],
                [2.5e-2, 2.5],
                [30, 50],
            ],
        }

        s = Sensitivity()
        s.define_model(SellmeijerRule.calculate_critical_head, param_piping)
        s.run_sensitivity(SensitivityMethods.Sobol)

        S1 = [
            0.02524764,
            0.00189894,
            0.28866622,
            0.21257942,
            0.02024224,
            0.00423152,
            0.18361218,
            0.03374495,
        ]

        assert all(np.abs(s.sensitivity_results["S1"] - S1) <= TOL)

        s.plot("./sensitivity", "Sobol.png")
        assert os.path.isfile(r"./sensitivity/Sobol.png")
        shutil.rmtree("./sensitivity")

    @pytest.mark.unittest
    def test_sobol_2(self):

        params = {
            "names": ["a", "b"],
            "bounds": [[0, 1], [0, 1]],
        }

        fixed_args = 0.0

        s = Sensitivity()
        s.define_model(parabola, params, fixed_args)
        s.run_sensitivity(SensitivityMethods.Sobol)

        assert all(np.abs(s.sensitivity_results["S1"] - np.array([1, 0])) <= TOL)

        s.plot("./sensitivity", "Sobol.png")
        assert os.path.isfile(r"./sensitivity/Sobol.png")
        shutil.rmtree("./sensitivity")


def parabola(*argv):
    return argv[0][1] + argv[0][2] * argv[0][0] ** 2
