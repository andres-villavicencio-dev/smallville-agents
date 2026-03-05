# config_validator.py
"""Configuration validator and recovery helper for SmallvilleSimulation"""

import os
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import subprocess

CONFIG_DIR = Path(__file__).parent
CONFIG_FILE = CONFIG_DIR / "config.json"
ENV_VARS_FILE = CONFIG_DIR / ".env"
LOG_FILE = CONFIG_DIR.parent / "validation_errors.log"

VALID_REQUIRED_FILES = [
    "main.py",
    "agent.py",
    "memory.py",
    "skillbank.py",
    "environment.py",
    "conversation.py",
    "llm.py",
    "personas.py",
    "prompts.py",
    "db/memories.db"
]

REQUIRED_DEPS = ["pyyaml", "pytest", "rich", "asyncio"]

MODEL_PATHS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "gemma3:1b",
    "gemma3:4b",
    "smallville-social",
    "smallville-actor"
]

DEFAULT_CONFIG = {
    "TICK_DURATION_SECONDS": 180,
    "MEMORY_RETRIEVAL_WEIGHTS": {
        "recency": 0.4,
        "importance": 0.35,
        "relevance": 0.25
    },
    "IMPORTANCE_THRESHOLD": 150,
    "RECENCY_DECAY_FACTOR": 0.99,
    "CONVERSATION_PROBABILITY": 0.15,
    "MAX_REPLAN_PER_DAY": 3,
    "AGENT_BATCH_SIZE": 8,
    "USE_COMMITTEE": False,
    "COMMITTEE_MODEL_JUDGE": "qwen2.5:3b",
    "COMMITTEE_MODEL_SOCIAL": "smallville-social",
    "COMMITTEE_MODEL_SPATIAL": "qwen2.5:3b",
    "COMMITTEE_MODEL_TEMPORAL": "gemma3:1b",
    "COMMITTEE_MODEL_EMOTIONAL": "llama3.2:3b",
    "COMMITTEE_MODEL_MEMORY": "gemma3:4b",
    "COMMITTEE_MODEL_DIALOGUE": "smallville-actor",
    "GPU_QUEUE_ENABLED": True,
    "AUTO_SAVE_INTERVAL_TICKS": 100
}


class ConfigValidator:
    """Validates and helps recover SmallvilleSimulation configuration"""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixes_applied: List[str] = []

    def run_all_checks(self, quiet: bool = False) -> bool:
        """Run all validation checks"""
        if not quiet:
            self._print_header("Configuration Validator")

        self._check_dependencies()
        self._check_required_files()
        self._check_model_availability()
        self._check_config_structure()
        self._check_env_vars()

        is_valid = len(self.errors) == 0
        if not quiet:
            self._print_summary(is_valid)
            if self.fixes_applied:
                self._print_fixes()

        return is_valid

    def _check_dependencies(self, quiet: bool = False):
        """Check if all required Python dependencies are installed"""
        if not quiet:
            self._print_section("Checking Dependencies")

        for dep in REQUIRED_DEPS:
            try:
                import importlib
                importlib.import_module(dep)
                if not quiet:
                    print(f"  ✓ {dep:12} - available")
            except ImportError:
                self.errors.append(f"Missing dependency: {dep}")
                if not quiet:
                    print(f"  ✗ {dep:12} - missing")

            # Try to install missing packages
            if dep in REQUIRED_DEPS and not quiet:
                self._try_install_dep(dep)

    def _try_install_dep(self, dep_name: str, quiet: bool = False):
        """Try to install a missing dependency"""
        print(f"  Attempting to install {dep_name}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"],
                         capture_output=True, text=True)
            if "pyyaml" in dep_name:
                self.fixes_applied.append(f"Installed pyyaml")
                if not quiet:
                    print("  ✓ Installed")
        except Exception as e:
            if not quiet:
                print(f"  ! Could not install {dep_name}: {e}")

    def _check_required_files(self, quiet: bool = False):
        """Check if all required Python files exist"""
        if not quiet:
            self._print_section("Checking Required Files")

        for f in VALID_REQUIRED_FILES:
            file_path = CONFIG_DIR.parent / f
            if not file_path.exists():
                self.errors.append(f"Missing required file: {f}")
                if not quiet:
                    print(f"  ✗ {f} - missing")
            else:
                if not quiet:
                    print(f"  ✓ {f}")

    def _check_model_availability(self, quiet: bool = False):
        """Check if configured models are available in Ollama"""
        if not quiet:
            self._print_section("Checking Model Availability")

        try:
            result = subprocess.run(
                [sys.executable, "-c", "import requests; requests.get('http://localhost:11434/api/tags')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                models = json.loads(result.stdout)["models"]
                existing = [m["name"] for m in models]

                for model in MODEL_PATHS:
                    if model in existing:
                        if not quiet:
                            print(f"  ✓ {model:20} - available")
                    else:
                        # Check if it's a custom model
                        self.warnings.append(f"Model '{model}' not found in Ollama")
                        if not quiet:
                            print(f"  ! {model:20} - not found (may be custom model)")
            else:
                self.warnings.append("Ollama API not reachable")
                if not quiet:
                    print(f"  ! Ollama API not reachable")

        except Exception as e:
            self.warnings.append(f"Could not check models: {e}")
            if not quiet:
                print(f"  ! Error checking models: {e}")

    def _check_config_structure(self, quiet: bool = False):
        """Check if config file is valid JSON"""
        if not quiet:
            self._print_section("Checking Configuration File")

        if not CONFIG_FILE.exists():
            self.errors.append(f"Config file missing: {CONFIG_FILE}")
            if not quiet:
                print(f"  ✗ Config file missing: {CONFIG_FILE}")
            return

        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)

            # Check required config fields
            if "TICK_DURATION_SECONDS" not in config:
                self.errors.append("Config missing TICK_DURATION_SECONDS")
                if not quiet:
                    print("  ✗ Missing TICK_DURATION_SECONDS")

            if "MEMORY_RETRIEVAL_WEIGHTS" not in config:
                self.errors.append("Config missing MEMORY_RETRIEVAL_WEIGHTS")
                if not quiet:
                    print("  ✗ Missing MEMORY_RETRIEVAL_WEIGHTS")

            # Validate weights if present
            if "MEMORY_RETRIEVAL_WEIGHTS" in config:
                weights = config["MEMORY_RETRIEVAL_WEIGHTS"]
                for key, val in weights.items():
                    if not isinstance(val, (int, float)) or not (0 <= val <= 1):
                        self.errors.append(f"Invalid memory weight: {key} = {val}")
                        if not quiet:
                            print(f"  ✗ Invalid memory weight: {key}")

            # Check for common PyYAML issues
            if config.get("MEMORY_RETRIEVAL_WEIGHTS"):
                config_str = json.dumps(config["MEMORY_RETRIEVAL_WEIGHTS"])
                try:
                    yaml.safe_load(config_str)
                except:
                    self.warnings.append("MEMORY_RETRIEVAL_WEIGHTS may contain YAML syntax issues")
                    if not quiet:
                        print("  ! MEMORY_RETRIEVAL_WEIGHTS may have YAML issues")

        except json.JSONDecodeError as e:
            self.errors.append(f"Config file is not valid JSON: {e}")
            if not quiet:
                print(f"  ✗ Invalid JSON: {e}")

    def _check_env_vars(self, quiet: bool = False):
        """Check for common environment variable issues"""
        if not quiet:
            self._print_section("Checking Environment Variables")

        if ENV_VARS_FILE.exists():
            with open(ENV_VARS_FILE) as f:
                env_vars = yaml.safe_load(f)

            if "OLLAMA_HOST" not in env_vars:
                self.errors.append("OLLAMA_HOST not set")
                self.fixes_applied.append("Defaulted OLLAMA_HOST to localhost:11434")
                if not quiet:
                    print(f"  ! Missing OLLAMA_HOST, defaulted to localhost:11434")

            if "LLM_MODEL" not in env_vars:
                self.errors.append("LLM_MODEL not set")
                self.fixes_applied.append("Defaulted LLM_MODEL to qwen2.5:3b")
                if not quiet:
                    print(f"  ! Missing LLM_MODEL, defaulted to qwen2.5:3b")

            # Check for model in config but not in env
            config_models = [
                config.get("COMMITTEE_MODEL_JUDGE"),
                config.get("COMMITTEE_MODEL_SOCIAL"),
                config.get("COMMITTEE_MODEL_SPATIAL"),
                config.get("COMMITTEE_MODEL_TEMPORAL"),
                config.get("COMMITTEE_MODEL_EMOTIONAL"),
                config.get("COMMITTEE_MODEL_MEMORY"),
                config.get("COMMITTEE_MODEL_DIALOGUE")
            ]

            missing_models = []
            for model in config_models:
                if model and model not in env_vars:
                    missing_models.append(model)

            for model in missing_models:
                self.fixes_applied.append(f"Defaulted {model} from config")
                if not quiet:
                    print(f"  ! Missing {model}, defaulted from config")

    def _print_header(self, title: str):
        """Print section header"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    def _print_section(self, title: str, quiet: bool = False):
        """Print section header"""
        if not quiet:
            print(f"\n{title}")
            print("-" * 40)

    def _print_summary(self, is_valid: bool):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print(" VALIDATION SUMMARY")
        print("=" * 60)

        if is_valid:
            print("  ✓ All checks passed!")
        else:
            print(f"  ✗ Found {len(self.errors)} errors and {len(self.warnings)} warnings")

        if self.errors:
            print(f"\n  Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"    - {error}")

        if self.warnings:
            print(f"\n  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"    - {warning}")

    def _print_fixes(self):
        """Print applied fixes"""
        print(f"\n  Fixes Applied ({len(self.fixes_applied)}):")
        for fix in self.fixes_applied:
            print(f"    - {fix}")

    def save_log(self):
        """Save validation errors to log file"""
        if self.errors or self.warnings:
            with open(LOG_FILE, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Validation run at {datetime.now()}\n")
                f.write(f"Errors: {len(self.errors)}\n")
                f.write(f"Warnings: {len(self.warnings)}\n")
                f.write(f"Fixes: {len(self.fixes_applied)}\n\n")

                f.write(f"Errors:\n")
                for error in self.errors:
                    f.write(f"  - {error}\n")

                f.write(f"\nWarnings:\n")
                for warning in self.warnings:
                    f.write(f"  - {warning}\n")

                f.write(f"\nFixes:\n")
                for fix in self.fixes_applied:
                    f.write(f"  - {fix}\n")
