import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# the package of this checkout, unless the tests are to run on an installed one (a wheel)
if not os.environ.get("HYDROBEM_TEST_INSTALLED"):
    sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
