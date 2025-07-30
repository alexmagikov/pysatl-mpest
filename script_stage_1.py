"""The script implements the first step of the experiment"""

from pathlib import Path

from experimental_env.preparation.dataset_generator import (
    RandomDatasetGenerator,
)
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp

WORKING_DIR = Path("D:\mpest\stage_1")
SAMPLES_SIZE = 1000

r_generator = RandomDatasetGenerator(42)
mixtures = [
    [GaussianModel, GaussianModel],
]
for models in mixtures:
    r_generator.generate(SAMPLES_SIZE, models, Path(WORKING_DIR), exp_count=100)
