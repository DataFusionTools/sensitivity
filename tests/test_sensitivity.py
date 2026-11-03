import os
import shutil
import pytest
import numpy as np

from models.piping import SellmeijerRule
from sensitivity.sensitivity import Sensitivity, SensitivityMethods


TOL = 1e-3


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    # plots are written to ./sensitivity, which must not be the source package
    monkeypatch.chdir(tmp_path)


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
            981.6714757094445,
            105.68694346889463,
            3025.4928572997146,
            2002.8338452586038,
            671.4740502480046,
            294.39830170154084,
            1896.4432709396765,
            616.7476032489969,
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
