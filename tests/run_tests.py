"""
Run the test suite.Each file named test_*.py is imported, and each function whose name starts with test_ is executed.
"""


# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import importlib.util
import sys
import traceback
from pathlib import Path


# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

current_file = Path(__file__).resolve()
test_dir = current_file.parent

PROJECT_ROOT = None

for parent in current_file.parents:
    if (parent / "functions").is_dir() and (parent / "scripts").is_dir():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise RuntimeError("Project root not found.")


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

test_files = sorted(test_dir.glob("test_*.py"))


# -------------------------------------------------------
# -------------------- MAIN WORKFLOW --------------------
# -------------------------------------------------------

print("Step 1: Find test files")
print(f"Test directory: {test_dir}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Number of test files: {len(test_files)}")
print("-------------------------------------------------------")


n_passed = 0
n_failed = 0

print("Step 2: Run tests")

for module_path in test_files:
    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    module.PROJECT_ROOT = PROJECT_ROOT
    spec.loader.exec_module(module)

    test_names = []

    for name in sorted(dir(module)):
        if name.startswith("test_") and callable(getattr(module, name)):
            test_names.append(name)

    for test_name in test_names:
        test_function = getattr(module, test_name)
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
print("Step 3: Summary")
print(f"Passed: {n_passed}")
print(f"Failed: {n_failed}")

if n_failed > 0:
    raise SystemExit(1)
