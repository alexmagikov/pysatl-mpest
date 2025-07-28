"""A module that provides a class that save data about how the metric error is distributed"""

from math import isnan
from pathlib import Path

import numpy as np
import yaml

from experimental_env.analysis.analyze_summarizers.analysis_summarizer import (
    AnalysisSummarizer,
)
from experimental_env.analysis.metrics import AMetric
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.utils import round_sig


class ErrorSummarizer(AnalysisSummarizer):
    """
    A class that calculates the average error for a dataset using the selected metric.
    """

    def __init__(self, metric: AMetric):
        super().__init__()
        self._metric = metric

    def calculate(self, results: list[ExperimentDescription]) -> tuple:
        """
        Helper function for calculating mean and standard deviation
        """
        errors = []
        for result in results:
            base_mixture = result.base_mixture
            result_mixture = result.steps[-1].result_mixture
            error = self._metric.error(base_mixture, result_mixture)

            if isnan(error):
                continue

            errors.append(error)

        if not errors:
            return 0, 0, 0

        mean = np.mean(errors)
        std = np.std(errors)
        median = np.median(errors)
        percentile_90 = np.percentile(errors, 90)
        percentile_95 = np.percentile(errors, 95)

        return float(mean), float(std), float(median), float(percentile_90), float(percentile_95)

    def _create_metrics_dict(self, mean, deviation, median, percentile_90, percentile_95, prefix="") -> dict:
        """
        Creates a dictionary of metrics, including only those whose value is between 0 and 1.
        """
        potential_metrics = {
            "mean": mean,
            "standard_deviation": deviation,
            "median": median,
            "percentile_90": percentile_90,
            "percentile_95": percentile_95,
        }

        info_dict = {}
        for name, value in potential_metrics.items():
            if value <= 1:
                key = f"{prefix}{name}"
                info_dict[key] = round_sig(value, 3)

        return info_dict

    def analyze_method(self, results: list[ExperimentDescription], method: str):
        mean, deviation, median, percentile_90, percentile_95 = self.calculate(results)

        info_dict = self._create_metrics_dict(mean, deviation, median, percentile_90, percentile_95)

        yaml_path: Path = self._out_dir.joinpath("metric_info.yaml")

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(info_dict, file)

    def compare_methods(
        self,
        results_1: list[ExperimentDescription],
        results_2: list[ExperimentDescription],
        method_1: str,
        method_2: str,
    ):
        mean_1, deviation_1, median_1, percentile_90_1, percentile_95_1 = self.calculate(results_1)
        mean_2, deviation_2, median_2, percentile_90_2, percentile_95_2 = self.calculate(results_2)

        info_dict_1 = self._create_metrics_dict(
            mean_1, deviation_1, median_1, percentile_90_1, percentile_95_1, prefix=f"{method_1}_"
        )

        info_dict_2 = self._create_metrics_dict(
            mean_2, deviation_2, median_2, percentile_90_2, percentile_95_2, prefix=f"{method_2}_"
        )

        info_dict = {**info_dict_1, **info_dict_2}

        yaml_path: Path = self._out_dir.joinpath("metric_info.yaml")

        with open(yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(info_dict, file)
