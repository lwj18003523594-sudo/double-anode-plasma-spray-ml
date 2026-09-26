"""Regression: a mixed Mac installation can be repaired without losing local results."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_code_update import ROOT, apply_source_update


def test_code_update_preserves_local_work():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "DoubleAnode_ML_Foolproof"
        target.mkdir()
        (target / "src").mkdir()
        (target / "src" / "quick_analysis.py").write_text("# old helper\n", encoding="utf-8")
        (target / "app.py").write_text("# old app\n", encoding="utf-8")
        protected = [".venv/bin/python", "data/literature_test/workbook.xlsx",
                     "runs/quick_analysis/run_history.json", "models/latest_chain.joblib",
                     "outputs/reports/user_report.md", "config/objectives.yaml"]
        for name in protected:
            p = target / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"preserve my local research")

        backup, n = apply_source_update(ROOT, target)
        assert n > 10 and (target / "app.py").read_bytes() == (ROOT / "app.py").read_bytes()
        assert (target / "src/quick_analysis.py").read_bytes() == (ROOT / "src/quick_analysis.py").read_bytes()
        assert (backup / "src/quick_analysis.py").read_text() == "# old helper\n"
        assert (backup / "app.py").read_text() == "# old app\n"
        assert all((target / p).read_bytes() == b"preserve my local research" for p in protected)


def test_incomplete_download_is_rejected_before_copy():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "incomplete"
        target = Path(tmp) / "installed"
        source.mkdir()
        target.mkdir()
        (source / "app.py").write_text("# new\n", encoding="utf-8")
        (target / "app.py").write_text("# existing\n", encoding="utf-8")
        try:
            apply_source_update(source, target)
        except ValueError as error:
            assert "不完整" in str(error)
        else:
            raise AssertionError("An incomplete archive must not be installed")
        assert (target / "app.py").read_text() == "# existing\n"


if __name__ == "__main__":
    test_code_update_preserves_local_work()
    test_incomplete_download_is_rejected_before_copy()
    print("PASS: complete source sync backs up code and preserves research data")
