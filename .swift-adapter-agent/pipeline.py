#!/usr/bin/env python3
"""Pipeline state manager for ms-swift adapter agent.

This script does NOT call LLMs. It is a tool for Kimi Code agent to:
1. Initialize a new pipeline run.
2. Check current status.
3. Mark stage start/complete/fail.
4. Switch between custom (unregistered) and registered model paths.
5. Update pipeline_state.json.

Usage:
    # Initialize (defaults to custom/unregistered model path)
    python pipeline.py --init --input input.json

    # Check status
    python pipeline.py --status --run-id 20260623-143052

    # Mark stage start
    python pipeline.py --start-stage dataset_register --run-id 20260623-143052

    # Mark stage complete with validation report
    python pipeline.py --complete-stage dataset_register \
        --run-id 20260623-143052 \
        --validation-report outputs/20260623-143052/validation_dataset.json

    # Mark stage failed
    python pipeline.py --fail-stage dataset_register \
        --run-id 20260623-143052 \
        --error-file outputs/20260623-143052/error_dataset.json

    # Switch to registered model path after swift_support_check
    python pipeline.py --switch-path registered --run-id 20260623-143052
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# Common stages for both paths.
COMMON_STAGE_ORDER = [
    "hardware_probe",
    "environment_setup",
    "swift_support_check",
]

# Stages for unregistered / custom models.
CUSTOM_STAGE_ORDER = [
    "model_analysis",
    "user_decision",
    "download_weights",
    "dataset_register",
    "model_register",
    "template_register",
    "integration_test",
    "smoke_test",
    "training_script",
    "full_training",
]

# Stages for ms-swift native / registered models.
REGISTERED_STAGE_ORDER = [
    "registered_model_info",
    "registered_checkpoint_assembly",
    "registered_dataset_register",
    "registered_integration_test",
    "registered_smoke_test",
    "registered_training_script",
    "registered_full_training",
]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_state_path(run_id: str) -> Path:
    return Path("outputs") / run_id / "pipeline_state.json"


def load_state(run_id: str) -> dict:
    path = get_state_path(run_id)
    if not path.exists():
        print(f"[ERROR] State not found: {path}")
        sys.exit(1)
    return load_json(str(path))


def save_state(state: dict):
    state["updated_at"] = now_iso()
    path = get_state_path(state["run_id"])
    save_json(str(path), state)


DEFAULT_MODEL_PATHS = {
    "kimi_audio": {
        "model_path": "/workspace/model/Qwen2.5-7B",
        "whisper_path": "/workspace/model/whisper-large-v3",
    },
    "mimo_audio": {
        "model_path": "/workspace/model/Qwen2.5-7B-Instruct",
        "whisper_path": "/workspace/model/whisper-large-v3",
    },
    "qwen2_audio": {
        "model_path": "/workspace/model/Qwen2-Audio-7B",
    },
    "qwen_omni": {
        "model_path": "/workspace/model/Qwen2.5-Omni-3B",
    },
    "qwen2_5_omni": {
        "model_path": "/workspace/model/Qwen2.5-Omni-3B",
    },
    "qwen3_omni": {
        "model_path": "/workspace/model/Qwen3-Omni-3B",
    },
    "step_audio": {
        "model_path": "/workspace/model/Step-Audio-Chat",
    },
    "step_audio2_mini": {
        "model_path": "/workspace/model/Step-Audio-2-mini",
    },
    "qwen2": {
        "model_path": "/workspace/model/Qwen2.5-7B",
    },
    "llama3": {
        "model_path": "/workspace/model/Meta-Llama-3-8B",
    },
}


def infer_defaults(inp: dict) -> dict:
    family = inp.get("model_family", "other")
    defaults = DEFAULT_MODEL_PATHS.get(family, {})
    for key, value in defaults.items():
        if not inp.get(key):
            inp[key] = value
    if "max_gpus" not in inp or not inp["max_gpus"]:
        inp["max_gpus"] = 7
    return inp


def build_stage_list(path: str) -> list:
    """Build the ordered stage list for a given path."""
    if path == "registered":
        return COMMON_STAGE_ORDER + REGISTERED_STAGE_ORDER
    return COMMON_STAGE_ORDER + CUSTOM_STAGE_ORDER


def init_state(args):
    inp = load_json(args.input)
    inp = infer_defaults(inp)
    run_id = inp.get("output_run_id") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = Path("outputs") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    path = getattr(args, "path", "custom")
    stage_order = build_stage_list(path)

    state = {
        "run_id": run_id,
        "input": inp,
        "path": path,
        "environment": {},
        "stages": [
            {
                "id": sid,
                "name": sid.replace("_", " ").title(),
                "status": "pending",
                "retry_count": 0,
                "output": {},
                "validation": None,
                "code_path": None,
                "code_section": None,
                "log_path": None,
                "error": None,
            }
            for sid in stage_order
        ],
        "current_stage": stage_order[0],
        "overall_status": "pending",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "error_summary": None,
    }
    save_state(state)
    print(f"[OK] Initialized pipeline: {run_id}")
    print(f"[INFO] Initial path: {path}")
    print(f"[INFO] State file: {get_state_path(run_id)}")


def get_stage(state: dict, stage_id: str):
    for s in state["stages"]:
        if s["id"] == stage_id:
            return s
    return None


def get_active_stage_order(state: dict) -> list:
    """Return the ordered list of stage IDs currently in the state."""
    return [s["id"] for s in state["stages"]]


def check_stage_order(state: dict, stage_id: str):
    stage_order = get_active_stage_order(state)
    idx = stage_order.index(stage_id)
    if idx > 0:
        prev_id = stage_order[idx - 1]
        prev = get_stage(state, prev_id)
        if prev["status"] != "completed":
            print(f"[ERROR] Cannot start {stage_id}: previous stage {prev_id} is {prev['status']}")
            sys.exit(1)


def start_stage(args):
    state = load_state(args.run_id)
    stage = get_stage(state, args.start_stage)
    if stage is None:
        print(f"[ERROR] Unknown stage: {args.start_stage}")
        sys.exit(1)

    check_stage_order(state, args.start_stage)

    stage["status"] = "in_progress"
    stage["started_at"] = now_iso()
    state["current_stage"] = args.start_stage
    state["overall_status"] = "in_progress"
    save_state(state)
    print(f"[OK] Stage started: {args.start_stage}")


def complete_stage(args):
    state = load_state(args.run_id)
    stage = get_stage(state, args.complete_stage)
    if stage is None:
        print(f"[ERROR] Unknown stage: {args.complete_stage}")
        sys.exit(1)

    stage["status"] = "completed"
    stage["completed_at"] = now_iso()

    if args.validation_report:
        report = load_json(args.validation_report)
        stage["validation"] = report

    if args.code_path:
        stage["code_path"] = args.code_path
    if args.code_section:
        stage["code_section"] = args.code_section
    if args.log_path:
        stage["log_path"] = args.log_path

    # Update current_stage to next stage
    stage_order = get_active_stage_order(state)
    idx = stage_order.index(args.complete_stage)
    if idx + 1 < len(stage_order):
        state["current_stage"] = stage_order[idx + 1]
    else:
        state["current_stage"] = "completed"
        state["overall_status"] = "completed"

    save_state(state)
    print(f"[OK] Stage completed: {args.complete_stage}")
    print(f"[INFO] Next stage: {state['current_stage']}")


def fail_stage(args):
    state = load_state(args.run_id)
    stage = get_stage(state, args.fail_stage)
    if stage is None:
        print(f"[ERROR] Unknown stage: {args.fail_stage}")
        sys.exit(1)

    stage["status"] = "failed"
    stage["completed_at"] = now_iso()
    stage["retry_count"] = stage.get("retry_count", 0) + 1

    if args.error_file:
        error = load_json(args.error_file)
        stage["error"] = error
        state["error_summary"] = error.get("message", "Unknown error")

    state["overall_status"] = "failed"
    save_state(state)
    print(f"[FAIL] Stage failed: {args.fail_stage}")


def switch_path(args):
    """Switch from custom path to registered path or vice versa.

    Only allowed after swift_support_check is pending/completed and before
    any path-specific stage has started.
    """
    state = load_state(args.run_id)
    current_path = state.get("path", "custom")
    new_path = args.switch_path

    if current_path == new_path:
        print(f"[INFO] Already on path: {new_path}")
        return

    swift_stage = get_stage(state, "swift_support_check")
    if swift_stage is None:
        print("[ERROR] Cannot switch path: swift_support_check stage not found")
        sys.exit(1)

    if swift_stage["status"] == "failed":
        print("[ERROR] Cannot switch path: swift_support_check failed")
        sys.exit(1)

    # Ensure no path-specific stage has started.
    path_specific_stages = set(CUSTOM_STAGE_ORDER + REGISTERED_STAGE_ORDER)
    for s in state["stages"]:
        if s["id"] in path_specific_stages and s["status"] in ("in_progress", "completed", "failed"):
            print(f"[ERROR] Cannot switch path: stage {s['id']} already {s['status']}")
            sys.exit(1)

    # Preserve state for common stages.
    common_state = {s["id"]: s for s in state["stages"] if s["id"] in COMMON_STAGE_ORDER}
    new_stage_order = build_stage_list(new_path)

    state["stages"] = []
    for sid in new_stage_order:
        if sid in common_state:
            state["stages"].append(common_state[sid])
        else:
            state["stages"].append({
                "id": sid,
                "name": sid.replace("_", " ").title(),
                "status": "pending",
                "retry_count": 0,
                "output": {},
                "validation": None,
                "code_path": None,
                "code_section": None,
                "log_path": None,
                "error": None,
            })

    state["path"] = new_path
    # current_stage should be the first pending stage after swift_support_check
    for s in state["stages"]:
        if s["status"] == "pending":
            state["current_stage"] = s["id"]
            break

    save_state(state)
    print(f"[OK] Switched path from {current_path} to {new_path}")
    print(f"[INFO] Current stage: {state['current_stage']}")


def show_status(args):
    state = load_state(args.run_id)
    print(f"Run ID: {state['run_id']}")
    print(f"Path: {state.get('path', 'custom')}")
    print(f"Overall Status: {state['overall_status']}")
    print(f"Current Stage: {state['current_stage']}")
    print(f"Error Summary: {state.get('error_summary') or 'None'}")
    print("")
    print(f"{'Stage':<30} {'Status':<12} {'Validated':<10} {'Code Path'}")
    print("-" * 90)
    for s in state["stages"]:
        validated = "-"
        if s.get("validation"):
            validated = "YES" if s["validation"].get("passed") else "NO"
        code = s.get("code_path") or "-"
        print(f"{s['id']:<30} {s['status']:<12} {validated:<10} {code}")


def main():
    parser = argparse.ArgumentParser(description="ms-swift adapter pipeline state manager")
    parser.add_argument("--run-id", help="Pipeline run id")
    parser.add_argument("--input", help="Input JSON file for init")
    parser.add_argument("--init", action="store_true", help="Initialize new pipeline")
    parser.add_argument("--path", default="custom", choices=["custom", "registered"],
                        help="Initial pipeline path (default: custom)")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--start-stage", help="Mark stage as in_progress")
    parser.add_argument("--complete-stage", help="Mark stage as completed")
    parser.add_argument("--fail-stage", help="Mark stage as failed")
    parser.add_argument("--validation-report", help="Path to validation report JSON")
    parser.add_argument("--error-file", help="Path to error JSON")
    parser.add_argument("--code-path", help="Path to generated code")
    parser.add_argument("--code-section", help="Section in generated code")
    parser.add_argument("--log-path", help="Path to stage log")
    parser.add_argument("--switch-path", choices=["custom", "registered"],
                        help="Switch pipeline path after swift_support_check")
    args = parser.parse_args()

    if args.init:
        init_state(args)
    elif args.status:
        if not args.run_id:
            print("[ERROR] --run-id required")
            sys.exit(1)
        show_status(args)
    elif args.switch_path:
        if not args.run_id:
            print("[ERROR] --run-id required")
            sys.exit(1)
        switch_path(args)
    elif args.start_stage:
        if not args.run_id:
            print("[ERROR] --run-id required")
            sys.exit(1)
        start_stage(args)
    elif args.complete_stage:
        if not args.run_id:
            print("[ERROR] --run-id required")
            sys.exit(1)
        complete_stage(args)
    elif args.fail_stage:
        if not args.run_id:
            print("[ERROR] --run-id required")
            sys.exit(1)
        fail_stage(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
