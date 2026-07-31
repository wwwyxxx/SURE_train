# 3.4 Validation and State Management

A fully automated adapter agent must not only generate integration code but also verify that every intermediate artifact is correct and recover from transient or deterministic failures. This section describes the validation framework, failure-handling strategy, retry mechanism, and persistent pipeline state used by the ms-swift adapter agent.

## 3.4.1 Validator Design

Validators are deterministic, standalone Python scripts that exercise the adapter code produced at each stage. They are organized into two layers:

* **Core validators** (`validators/core/`) check generic invariants that every ms-swift integration must satisfy, such as dataset registration (`validate_dataset_registration.py`), model registration (`validate_model_registration.py`), template registration (`validate_template_registration.py`), forward pass correctness (`validate_forward_pass.py`), single-step training (`validate_single_step_training.py`), loss computation (`validate_loss_computation.py`), freeze/unfreeze logic (`validate_freeze_unfreeze.py`), inference (`validate_inference.py`), checkpoint save/load (`validate_checkpoint_save_load.py`), distributed launch (`validate_distributed_launch.py`), and batch-size scaling (`validate_batch_size_scaling.py`).
* **Model-specific validators** (`validators/model_specific/`) encode knowledge about known failure modes of particular architectures. For example, `validators/model_specific/kimi_audio/validate_label_shift.py` verifies that `text_input_ids`, `labels`, and `text_loss_mask` have identical shapes, that the mask is binary, that padding positions are excluded from loss, and that label-shift alignment matches the assistant-token count. Similar targeted checks exist for MiMo-Audio (`validate_mimo_audio_forward.py`, `validate_mimo_audio_template.py`) to catch RVQ tokenization and audio-text interleaving errors early.

All validators share a uniform execution contract: they accept `--custom-register-path` plus stage-specific arguments (e.g., `--model-type`, `--dataset-name`, `--model`), import the generated registration module dynamically, call ms-swift APIs such as `load_dataset`, `get_model_tokenizer`, `get_template`, and `data_collator`, and exit with code `0` only if every assertion passes.

Because validators may hang when loading large models or compiling CUDA kernels, an executor wrapper `executors/run_validator.py` runs each validator in a subprocess with a configurable timeout (default 600 s). The wrapper produces a structured JSON report:

```json
{
  "passed": true,
  "validator": "validators/core/validate_dataset_registration.py",
  "report": "...",
  "stdout": "...",
  "stderr": "...",
  "metrics": {}
}
```

If the validator script is missing, the process times out, or the subprocess returns a non-zero exit code, `passed` is set to `false` and the captured output is preserved for diagnosis. A validator must be executed and pass before the corresponding pipeline stage can be marked as completed; skipping validation is prohibited by the agent harness.

## 3.4.2 Failure Handling

Failures are handled at two levels: the state machine enforces stage ordering and records errors, while the agent applies a staged debugging policy.

**State-machine enforcement.** The pipeline state manager (`pipeline.py`) only allows a stage to start if the immediately preceding stage has status `completed` (`check_stage_order`). When a stage fails, the agent calls:

```bash
python pipeline.py --fail-stage {stage_id} \
  --run-id {run_id} \
  --error-file outputs/{run_id}/error_{stage_id}.json
```

This updates the stage status to `failed`, increments `retry_count`, stores a structured `error` object (`{type, message, traceback}`), and sets the run-level `overall_status` to `failed` and `error_summary` to the error message. The agent therefore never proceeds to downstream stages until the current blocker is resolved.

**Agent debugging policy.** When a validator fails, the agent is instructed to (1) read the captured logs rather than guess the cause, (2) localize the failure to the exact stage, (3) apply the smallest possible code change, (4) re-run the failing stage's validator, and (5) also re-validate any downstream stages that depend on the modified artifact. This tight feedback loop prevents error propagation and keeps the integration diff minimal and reviewable.

**Path switching.** The harness supports switching between the custom (unregistered model) path and the registered (natively supported model) path immediately after `swift_support_check`. The `switch_path` function preserves the state of common stages (`hardware_probe`, `environment_setup`, `swift_support_check`) and rebuilds the remaining stage list, but it refuses to switch once any path-specific stage has already started or completed. This guards against inconsistent mixed-path executions.

## 3.4.3 Retry Mechanism

Retries are tracked per stage in `pipeline_state.json`. Each stage object carries a `retry_count` field that starts at `0` and is incremented every time `fail_stage` is invoked. The harness policy limits the agent to at most three retries for any single stage. If a stage still fails after three attempts, the pipeline stops and the agent reports the current state and the unresolved blocker to the user.

This design intentionally keeps retry logic lightweight: the state file records how many times a stage has been retried, and the agent uses that counter to decide whether to attempt another repair cycle or escalate. Re-execution itself is performed by re-invoking the same executor or validator after the agent has patched the adapter code. Because every retry updates `updated_at` and appends a new error record, the history of the debugging process remains fully auditable.

## 3.4.4 Persistent Pipeline State (`pipeline_state.json`)

The single source of truth for a run is `outputs/{run_id}/pipeline_state.json`. Its schema is formalized in `schemas/pipeline_state.schema.json` (JSON Schema draft-07) and is validated on every state update. The state file contains:

* **Run metadata:** `run_id`, `created_at`, `updated_at`, `overall_status`, and `error_summary`.
* **Input specification:** the contents of `input.json`, including `model_family`, `model_path`, `dataset_path`, `dataset_name`, `max_gpus`, and optional component paths.
* **Execution path:** `custom` or `registered`, determining which ordered stage list is active.
* **Environment summary:** Docker image, Dockerfile path, and import-test status.
* **Stages array:** one entry per pipeline stage. Each entry records `id`, `name`, `status` (`pending`, `in_progress`, `completed`, `failed`, or `skipped`), `retry_count`, `started_at`, `completed_at`, `output`, `validation`, `code_path`, `code_section`, `log_path`, and `error`.

The `validation` field embeds the JSON report produced by `run_validator.py`, so the state file is self-contained: a human or another process can open it and see not only that `dataset_register` passed, but also the dataset size, samples checked, output keys, and missing-audio count. Likewise, `code_path` links the completed stage to the exact artifact it produced (e.g., `custom/mimo_audio_swift_register.py`), enabling traceability from result to source.

An excerpt from a MiMo-Audio run illustrates the structure:

```json
{
  "run_id": "20260627-160445",
  "path": "custom",
  "current_stage": "full_training",
  "overall_status": "in_progress",
  "stages": [
    {
      "id": "hardware_probe",
      "status": "completed",
      "validation": {
        "passed": true,
        "gpu_count": 8,
        "usable_gpu_count": 7,
        "usable_gpus": [
          {"id": 0, "name": "NVIDIA A800-SXM4-80GB", "memory_total_gb": 80.0}
        ]
      }
    },
    {
      "id": "dataset_register",
      "status": "completed",
      "code_path": "custom/mimo_audio_swift_register.py",
      "validation": {
        "passed": true,
        "dataset_size": 134423,
        "samples_checked": 3,
        "missing_audio": 0
      }
    }
  ]
}
```

By persisting this state, the agent can resume after interruption, the user can monitor progress without parsing logs, and downstream tools can programmatically determine whether a run is ready for full training. The combination of per-stage validators, structured failure records, bounded retries, and a machine-readable state file is what makes the adapter agent robust enough to integrate complex speech and audio models without human micromanagement.
