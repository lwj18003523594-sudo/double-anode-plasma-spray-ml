"""A Mac with new app.py and old quick_analysis.py must see guidance, not traceback."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest
from src import quick_analysis as qa


def test_old_module_displays_repair_instructions():
    with patch.object(qa, "load_quick_workspace", None):
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
        at.run()
        assert not at.exception, [str(e.message) for e in at.exception]
        assert any("程序文件版本不一致" in str(e.value) for e in at.error)
        assert any("整套源码" in str(e.value) for e in at.error)


if __name__ == "__main__":
    test_old_module_displays_repair_instructions()
    print("PASS: mixed installation is blocked with clear upgrade guidance")
