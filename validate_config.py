"""Command-line interface for config validation"""
import argparse
import sys
from pathlib import Path

from config_validator import ConfigValidator


def main():
    parser = argparse.ArgumentParser(description="Validate SmallvilleSimulation configuration")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    parser.add_argument("--check-only", action="store_true", help="Only check, don't auto-fix")
    parser.add_argument("--fix-all", action="store_true", help="Attempt to fix all issues")
    parser.add_argument("--output", "-o", help="Output format (text/json)")

    args = parser.parse_args()

    validator = ConfigValidator()
    is_valid = validator.run_all_checks(quiet=args.quiet)

    if args.output == "json":
        import json
        print(json.dumps({
            "valid": is_valid,
            "errors": validator.errors,
            "warnings": validator.warnings,
            "fixes": validator.fixes_applied
        }, indent=2))

    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()

