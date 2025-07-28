"""The script implements the second step of the experiment"""

from pathlib import Path

from experimental_env.experiment.experiment_executors.random_executor import (
    RandomExperimentExecutor,
)
from experimental_env.experiment.estimators import (
    MomentsEstimator,
    LMomentsEstimator,
)
from experimental_env.preparation.dataset_parser import SamplesDatasetParser
from mpest.em.breakpointers import StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)

SOURCE_DIR = Path("D:\mpest\stage_1")
WORKING_DIR = Path("D:\mpest\stage_2")

if __name__ == '__main__':
    # Parse stage 1
    parser = SamplesDatasetParser()
    datasets = parser.parse(SOURCE_DIR)

    # Execute stage 2
    executor = RandomExperimentExecutor(WORKING_DIR, 5  , 43)
    executor.execute(
        datasets,
        LMomentsEstimator(
            StepCountBreakpointer(max_step=16),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
        ),
    )

    executor = RandomExperimentExecutor(WORKING_DIR, 5, 43)
    executor.execute(
        datasets,
        MomentsEstimator(
            StepCountBreakpointer(max_step=16),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
        ),
    )
