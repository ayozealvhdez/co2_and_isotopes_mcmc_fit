"""
Run the test suite without requiring pytest or unittest classes.

Each file named test_*.py is imported, and each function whose name starts with
test_ is executed. This keeps the tests close to normal scientific Python
scripts: simple functions, explicit arrays, and direct assertions.
"""

import importlib.util
import traceback
from pathlib import Path

import context  # noqa: F401


def load_module(module_path):
    """Import one test module from its file path."""
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_test_functions(module):
    """Yield test functions from one imported module."""
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue

        candidate = getattr(module, name)
        if callable(candidate):
            yield name, candidate


def main():
    test_dir = Path(__file__).resolve().parent
    test_files = sorted(test_dir.glob("test_*.py"))

    n_passed = 0
    n_failed = 0

    for module_path in test_files:
        module = load_module(module_path)

        for test_name, test_function in iter_test_functions(module):
            label = f"{module_path.name}::{test_name}"
            try:
                test_function()
            except Exception:
                n_failed += 1
                print(f"FAILED  {label}")
                traceback.print_exc()
            else:
                n_passed += 1
                print(f"PASSED  {label}")

    print("-------------------------------------------------------")
    print(f"Passed: {n_passed}")
    print(f"Failed: {n_failed}")

    if n_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
