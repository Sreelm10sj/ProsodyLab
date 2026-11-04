"""
backend.training.worker

Isolated subprocess worker for long-running LexiRep training jobs.
Executed as:
    python -m backend.training.worker --job-id <UUID> --dataset <path_to_npz> --output-dir <path> --config <path_to_config_json>

Lifecycle:
1. Parse CLI arguments and establish job sandbox
2. Atomically write initial job_state.json ("queued" -> "validating")
3. Validate feature matrix, labels, and word groupings via validate_dataset
4. Run AnchorScanner if explicit anchor is not specified in config
5. Launch LexiRepTrainer with real-time atomic job_state.json updates
6. Handle graceful cooperative cancellation (SIGTERM / flag)
7. Finalize state as "completed" or "failed" with structured diagnostics
"""

import os
import sys
import json
import time
import signal
import argparse
from typing import Dict, Any, Optional

import numpy as np

# Ensure backend root is on sys.path
_TESTS_OR_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TESTS_OR_TRAIN_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from backend.training.validator import validate_dataset
from backend.training.anchor_scanner import AnchorScanner
from backend.training.trainer import (
    LexiRepTrainer,
    TrainingConfig,
    TrainingProgress,
    TrainingCancelled,
)

# Global cancellation flag for graceful signal handling
_CANCEL_REQUESTED = False


def _signal_handler(signum, frame):
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True


def is_cancelled() -> bool:
    return _CANCEL_REQUESTED


def write_job_state_atomically(state_file: str, state_dict: Dict[str, Any]):
    """Write job_state.json using an atomic temporary file rename."""
    temp_file = f"{state_file}.tmp_{os.getpid()}"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        os.replace(temp_file, state_file)
    except Exception as exc:
        # If atomic replace fails (e.g. temporary Windows lock), attempt retry
        time.sleep(0.05)
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=2)
            os.replace(temp_file, state_file)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="LexiRep Subprocess Training Worker")
    parser.add_argument("--job-id", required=True, help="Unique UUID for this training job")
    parser.add_argument("--dataset", required=True, help="Path to input .npz dataset file")
    parser.add_argument("--output-dir", required=True, help="Directory to save artifacts and job_state.json")
    parser.add_argument("--config", required=True, help="Path to training config JSON file")
    args = parser.parse_args()

    # Register OS signals for graceful termination
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    job_id = args.job_id
    dataset_path = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output_dir)
    config_path = os.path.abspath(args.config)
    os.makedirs(output_dir, exist_ok=True)

    state_file = os.path.join(output_dir, "job_state.json")
    t0 = time.time()

    state = {
        "job_id": job_id,
        "status": "queued",
        "current_step": "Worker initialized",
        "progress_percent": 0.0,
        "loop": 0,
        "total_loops": 13,
        "elapsed_seconds": 0.0,
        "estimated_remaining_seconds": None,
        "pseudo_label_changes": 0,
        "loss_history": [],
        "error": None,
    }
    write_job_state_atomically(state_file, state)

    try:
        # 1. Load training config
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file missing: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        language = raw_cfg.get("language", "GER")
        outer_loops = raw_cfg.get("outer_loops", 13)
        state["total_loops"] = outer_loops

        trainer_cfg = TrainingConfig(
            language=language,
            seed=raw_cfg.get("seed", 42),
            device=raw_cfg.get("device", "auto"),
            warmup_epochs=raw_cfg.get("warmup_epochs", 5),
            idec_pretrain_epochs=raw_cfg.get("idec_pretrain_epochs", 10),
            outer_loops=outer_loops,
            idec_steps_per_loop=raw_cfg.get("idec_steps_per_loop", 100),
            refinement_epochs=raw_cfg.get("refinement_epochs", 3),
            supcon_temperature=raw_cfg.get("supcon_temperature", 0.08),
            batch_words=raw_cfg.get("batch_words", None),
            idec_batch_size=raw_cfg.get("idec_batch_size", 128),
            top_k_rep=raw_cfg.get("top_k_rep", None),
            learning_rate=raw_cfg.get("learning_rate", 0.001),
        )

        # 2. Load dataset
        state["status"] = "validating"
        state["current_step"] = "Loading and validating dataset"
        state["progress_percent"] = 1.0
        state["elapsed_seconds"] = round(time.time() - t0, 2)
        write_job_state_atomically(state_file, state)

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file missing: {dataset_path}")

        data = np.load(dataset_path)
        if "X_tr" in data and "Y_tr" in data and "W_tr" in data:
            X_tr, Y_tr, W_tr = data["X_tr"], data["Y_tr"], data["W_tr"]
        elif "X" in data and "Y" in data and "W" in data:
            X_tr, Y_tr, W_tr = data["X"], data["Y"], data["W"]
        else:
            raise KeyError("Dataset .npz must contain either ('X_tr', 'Y_tr', 'W_tr') or ('X', 'Y', 'W').")

        # Held-out test set if present in archive
        X_te = data.get("X_te")
        Y_te = data.get("Y_te")
        W_te = data.get("W_te")
        if X_te is None:
            X_te = data.get("X_test")
            Y_te = data.get("Y_test")
            W_te = data.get("W_test")

        # Validate dataset
        val_res = validate_dataset(X_tr, Y_tr, W_tr)
        if not val_res.valid:
            error_msg = "; ".join(val_res.errors)
            raise ValueError(f"Dataset validation failed: {error_msg}")

        # 3. Anchor Selection
        explicit_anchor = raw_cfg.get("anchor")
        if explicit_anchor is None:
            state["status"] = "anchor_search"
            state["current_step"] = "Running train-set anchor scan"
            state["progress_percent"] = 2.0
            state["elapsed_seconds"] = round(time.time() - t0, 2)
            write_job_state_atomically(state_file, state)

            scanner = AnchorScanner(max_candidates=10)
            scan_res = scanner.scan(X_tr, Y_tr, W_tr, language=language)
            if scan_res.selected_anchor is None:
                raise RuntimeError("Anchor search did not identify any viable anchor pairs.")
            selected_anchor = scan_res.selected_anchor.to_dict()
        else:
            selected_anchor = explicit_anchor

        # 4. Configure Trainer with real-time state persistence
        loss_history = []

        def on_trainer_progress(p: TrainingProgress):
            idec_val = None
            if p.metrics and "idec_loss" in p.metrics:
                idec_val = p.metrics["idec_loss"]
            elif p.loss is not None and p.status in ("idec_pretrain", "training"):
                idec_val = p.loss

            if idec_val is not None:
                state["idec_loss"] = float(idec_val)

            if p.metrics:
                state["current_metrics"] = p.metrics

            if p.loss is not None or idec_val is not None:
                loss_history.append({
                    "loop": p.loop,
                    "epoch": p.epoch,
                    "phase": p.status,
                    "loss": p.loss,
                    "idec_loss": idec_val,
                    "label_changes": p.pseudo_label_changes,
                    "elapsed": round(time.time() - t0, 2),
                })
                if len(loss_history) > 100:
                    del loss_history[0]

            state["status"] = p.status
            state["current_step"] = p.current_step
            state["progress_percent"] = p.progress_percent
            state["loop"] = p.loop
            state["total_loops"] = p.total_loops
            state["pseudo_label_changes"] = p.pseudo_label_changes
            state["loss_history"] = loss_history
            state["elapsed_seconds"] = round(time.time() - t0, 2)

            # Estimate remaining time if in training loops
            if p.loop > 0 and p.total_loops > 0:
                fraction_done = max(0.01, p.progress_percent / 100.0)
                elapsed = time.time() - t0
                state["estimated_remaining_seconds"] = round((elapsed / fraction_done) - elapsed, 1)

            write_job_state_atomically(state_file, state)

        trainer = LexiRepTrainer(
            config=trainer_cfg,
            callback=on_trainer_progress,
            cancellation_check=is_cancelled,
        )

        train_result = trainer.train(
            X_train=X_tr,
            Y_train=Y_tr,
            W_train=W_tr,
            anchor=selected_anchor,
            output_dir=output_dir,
            X_test=X_te,
            Y_test=Y_te,
            W_test=W_te,
        )

        # 5. Success
        state["status"] = "completed"
        state["current_step"] = "Training and verification complete"
        state["progress_percent"] = 100.0
        state["elapsed_seconds"] = round(time.time() - t0, 2)
        state["estimated_remaining_seconds"] = 0.0
        state["error"] = None
        state["final_metrics"] = train_result.final_metrics
        if "training_accuracy" in train_result.final_metrics:
            state["training_accuracy"] = train_result.final_metrics["training_accuracy"]
        if "final_idec_loss" in train_result.final_metrics:
            state["idec_loss"] = train_result.final_metrics["final_idec_loss"]
        if "benchmark" in train_result.final_metrics:
            state["benchmark"] = train_result.final_metrics["benchmark"]
        write_job_state_atomically(state_file, state)
        sys.exit(0)

    except TrainingCancelled:
        state["status"] = "cancelled"
        state["current_step"] = "Training cancelled by user request"
        state["elapsed_seconds"] = round(time.time() - t0, 2)
        state["estimated_remaining_seconds"] = None
        write_job_state_atomically(state_file, state)
        sys.exit(0)

    except Exception as exc:
        state["status"] = "failed"
        state["current_step"] = "Training failed"
        state["error"] = str(exc)
        state["elapsed_seconds"] = round(time.time() - t0, 2)
        state["estimated_remaining_seconds"] = None
        write_job_state_atomically(state_file, state)
        sys.exit(1)


if __name__ == "__main__":
    main()
