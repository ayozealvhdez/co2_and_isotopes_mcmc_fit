"""
Run the test suite.

Each file named test_*.py is imported, and each function whose name starts with test_ is executed.
"""


# -------------------------------------------------------
# --------------- PACKAGES AND FUNCTIONS ----------------
# -------------------------------------------------------

import importlib.util
import os
import sys
import traceback


# -------------------------------------------------------
# ---------------------- PATHS --------------------------
# -------------------------------------------------------

current_file = os.path.abspath(__file__)
test_dir = os.path.dirname(current_file)

PROJECT_ROOT = None

current_dir = test_dir
while True:
    if os.path.isdir(os.path.join(current_dir, "functions")) and os.path.isdir(os.path.join(current_dir, "scripts")):
        PROJECT_ROOT = current_dir
        break

    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        break

    current_dir = parent_dir

if PROJECT_ROOT is None:
    raise RuntimeError("Project root not found.")


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

test_files = []
for filename in os.listdir(test_dir):
    if filename.startswith("test_") and filename.endswith(".py"):
        test_files.append(os.path.join(test_dir, filename))
test_files.sort()


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
    module_name = os.path.splitext(os.path.basename(module_path))[0]
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
        label = f"{os.path.basename(module_path)}::{test_name}"

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
