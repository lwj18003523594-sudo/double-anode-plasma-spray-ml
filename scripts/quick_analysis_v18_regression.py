# -*- coding: utf-8 -*-
"""Regression for APS literature import, process-only training and figure export.

Run: python scripts/quick_analysis_v18_regression.py
"""
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import quick_analysis as qa
from src.paper_labels import target_slug


def test_literature_process_only():
    n = 32
    idx = np.arange(n)
    frame = pd.DataFrame({
        "喷涂序号": idx + 1,
        "Ar压力(psi)": 55 + (idx % 5) * 10,
        "H2压力(psi)": 25 + (idx % 3) * 5,
        "电流(A)": 300 + (idx % 9) * 40,
        "喷距(mm)": 50 + (idx % 7) * 15,
        "粒子温度(°C)": 2700 + (idx % 9) * 12 - (idx % 7) * 20,
        "粒子速度(m/s)": 180 + (idx % 9) * 3 - (idx % 7) * 2,
    })
    selected, ambiguous = qa.infer_field_map(frame)
    assert not ambiguous, ambiguous
    assert len(selected["x"]) == 4 and len(selected["states"]) == 2
    assert "喷涂序号" not in selected["x"]
    assert "Ar压力(psi)" in selected["x"]  # pressure must retain its unit
    schema = qa.build_runtime_schema(frame, selected)
    assert qa.data_readiness_issues(frame, schema)[0] == []
    assert qa.data_readiness_issues(frame.assign(**{"粒子温度(°C)": "unknown",
                                                     "粒子速度(m/s)": "unknown"}), schema)[0]
    assert "/" not in target_slug("粒子速度(m/s)")

    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(qa, "QUICK_ROOT", Path(tmp)), \
         patch.object(qa, "HISTORY_FILE", Path(tmp) / "history.json"), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = qa.run_quick_analysis(frame, run_id="REGRESSION_LITERATURE", demo=False)
        assert not result["record"]["failed"], result["steps"]
        assert len(result["bundle"]["stage1_models"]) == 2
        assert result["bundle"]["stage3_models"] == {}
        assert result["package"]["figures"] >= 4
        assert any("缺少可训练的涂层性能目标" in reason for _, status, reason
                   in result["steps"] if status == "skip")
        assert Path(result["package"]["zip"]).exists()


if __name__ == "__main__":
    test_literature_process_only()
    print("PASS: APS 32-row process data trains, exports figures and skips unsupported Pareto")
