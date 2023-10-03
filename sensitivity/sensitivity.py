import os
import sys
from dataclasses import dataclass, field
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from SALib.sample import saltelli
from SALib.analyze import sobol
from typing import Callable, Union
import numpy as np
import matplotlib.pylab as plt
from enum import Enum
import types
from core.base_class import BaseClass


class SensitivityMethods(Enum):
    Morris = "morris"
    Sobol = "sobol"


@dataclass
class Sensitivity(BaseClass):
    sensitivity_function: Union[Callable, None] = None
    N: int = 1024
    problem: dict = field(default_factory=dict)
    sensitivity_results: dict = field(default_factory=dict)
    fixed_arguments: list = field(default_factory=list)
    morris_nb_levels: int = 4
    seed: int = 1
    method: str = ""
    plot_data: dict = field(default_factory=dict)

    def define_model(self, sensitivity_function, parameters, *args):
        """
        Define the sensitivity analysis model

        :param sensitivity_function: function that calls the model
        :param parameters: model parameters for the sensitivity function
        """
        if not isinstance(sensitivity_function, types.FunctionType):
            sys.exit(
                "Error in Sensitivity.define_model:\n\tsensitivity_function needs to be a function"
            )

        self.sensitivity_function = sensitivity_function
        self.problem = parameters
        self.fixed_arguments = args
        self.problem["num_vars"] = len(parameters["names"])

    def run_sensitivity(self, method):
        """
        Run the sensitivity analysis for the specified method

        :param method: Method for the sensitivity study (morris or cobol)
        """

        # assign method
        self.method = method.value

        # sample the variables
        if method.value == "morris":
            self.morris_method()
        elif method.value == "sobol":
            self.sobol_method()

    def morris_method(self):
        """
        Morris method for sensitivity analysis
        """
        # sample the variables
        data = morris_sample.sample(
            self.problem, self.N, num_levels=self.morris_nb_levels, seed=self.seed
        )
        # run the sensitivity function for the sample variables
        results = np.array(
            [
                self.sensitivity_function(np.hstack([self.fixed_arguments, d]))
                for d in data
            ]
        )

        # determine the sensitivity
        self.sensitivity_results = morris_analyze.analyze(
            self.problem,
            data,
            results,
            num_levels=self.morris_nb_levels,
            seed=self.seed,
        )

        # sort data
        idx = np.argsort(self.sensitivity_results["mu_star"])
        self.plot_data = {
            "x_label": r"$\mu^{*}$",
            "x_data": np.array(self.sensitivity_results["names"])[idx],
            "y_data": self.sensitivity_results["mu_star"][idx],
        }

    def sobol_method(self):
        """
        Sobol method for sensitivity analysis
        """
        # sample the variables
        data = saltelli.sample(self.problem, self.N)
        # run the sensitivity function for the sample variables
        results = np.array(
            [
                self.sensitivity_function(np.hstack([self.fixed_arguments, d]))
                for d in data
            ]
        )

        # determine the sensitivity
        self.sensitivity_results = sobol.analyze(self.problem, results, seed=self.seed)

        # sort data
        idx = np.argsort(self.sensitivity_results["S1"])
        self.plot_data = {
            "x_label": r"$S1$",
            "x_data": np.array(self.sensitivity_results.problem["names"])[idx],
            "y_data": self.sensitivity_results["S1"][idx],
        }

    def plot(self, output_folder="./", name="sensitivity.png"):
        """
        Create plot

        :param output_folder: output file location
        :param name: name of the figure
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        fig, ax = plt.subplots()
        ax.barh(self.plot_data["x_data"], self.plot_data["y_data"])
        ax.set_xlabel(self.plot_data["x_label"])
        ax.set_ylabel("model parameter")
        ax.set_title(f"Sensitivity model: {self.method}")
        ax.grid()
        plt.savefig(os.path.join(output_folder, name))
        plt.close()
