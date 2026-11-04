"""
backend.training

LexiRep Training Workbench core package.
Provides:
- Strict dataset validation (validate_dataset, DatasetValidationResult, DatasetValidationError)
- Defensible, train-set-only anchor scanning (AnchorScanner, AnchorCandidate, AnchorScanResult)
- Production-compatible model trainer (LexiRepTrainer, TrainingConfig, TrainingProgress, TrainingResult, TrainingCancelled)
- Subprocess job management and sandboxing (TrainingJobManager, JobManagerError)
"""

from backend.training.validator import (
    validate_dataset,
    DatasetValidationResult,
    DatasetValidationError,
)
from backend.training.anchor_scanner import (
    AnchorScanner,
    AnchorCandidate,
    AnchorScanResult,
)
from backend.training.trainer import (
    LexiRepTrainer,
    TrainingConfig,
    TrainingProgress,
    TrainingResult,
    TrainingCancelled,
)
from backend.training.job_manager import (
    TrainingJobManager,
    JobManagerError,
)
from backend.training.dataset_loader import (
    load_dataset_npz,
    DatasetRegistry,
    DatasetLoaderError,
    DatasetTooLargeError,
)
from backend.training.evaluator import (
    evaluate_lexirep_benchmark,
    evaluate_bundle_on_dataset,
    BenchmarkResult,
)
from backend.training.api import router as training_router

__all__ = [
    "validate_dataset",
    "DatasetValidationResult",
    "DatasetValidationError",
    "AnchorScanner",
    "AnchorCandidate",
    "AnchorScanResult",
    "LexiRepTrainer",
    "TrainingConfig",
    "TrainingProgress",
    "TrainingResult",
    "TrainingCancelled",
    "TrainingJobManager",
    "JobManagerError",
    "load_dataset_npz",
    "DatasetRegistry",
    "DatasetLoaderError",
    "DatasetTooLargeError",
    "evaluate_lexirep_benchmark",
    "evaluate_bundle_on_dataset",
    "BenchmarkResult",
    "training_router",
]
