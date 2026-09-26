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
from streamlit.testing.v1 import AppTest


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
        workspace = qa.load_quick_workspace(result["run_id"])
        assert workspace["df"].equals(frame)
        assert workspace["schema"] == result["schema"]
        assert workspace["bundle"]["dataset_hash"] == result["bundle"]["dataset_hash"]
        assert workspace["objectives"] == {}

        # All Streamlit tabs render in one run. A Quick workspace must populate
        # their shared status without copying the model to formal models/.
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                               default_timeout=90)
        at.session_state["research_mode"] = "literature"
        at.session_state["active_quick_run_id"] = result["run_id"]
        at.session_state["smart_summary_run_id"] = result["run_id"]
        at.run()
        assert not at.exception, [str(x.message) for x in at.exception]
        assert len(at.tabs) == 8
        assert at.session_state["ctx"]["dataset_loaded"]
        assert at.session_state["ctx"]["model_trained"]
        assert at.session_state["ctx"]["active_run_id"] == result["run_id"]
        assert any("已加载快速运行" in str(x.value) for x in at.success)
        assert any("当前运行已生成过程模型" in str(x.value) for x in at.info)
        assert not any("请先训练模型" in str(x.value) for x in at.warning)
        assert any("本次智能分析结果" in str(x.value) for x in at.markdown)
        assert any("尚无可用的涂层性能模型" in str(x.value) for x in at.info)
        for title in ("查看完整指标、影响因素与数据覆盖", "查看结论依据与运行信息",
                      "研究路线（可选）", "平台状态与证据概览（可选）"):
            assert any(title in x.label for x in at.expander), title
        at.session_state["qa_file_cache"] = Path(__file__)
        at.session_state["qa_file_hash"] = "regression-file"
        at.session_state["qa_last_analyzed_hash"] = "regression-file"
        at.session_state["qa_last_analyzed_run_id"] = result["run_id"]
        at.run()
        assert at.button(key="smart_analyze_btn").label == "查看本次结果"
        history_count = len(qa.load_history())
        at.button(key="smart_analyze_btn").click().run()
        assert not at.exception, [str(x.message) for x in at.exception]
        assert len(qa.load_history()) == history_count  # Viewing must not retrain.
        at.sidebar.selectbox[0].select("正式数据与模型").run()
        assert not at.exception, [str(x.message) for x in at.exception]
        assert at.session_state["ctx"]["active_run_id"] is None

        qa.append_history({**result["record"], "run_id": "LEGACY_WITHOUT_SNAPSHOT"})
        try:
            qa.load_quick_workspace("LEGACY_WITHOUT_SNAPSHOT")
        except ValueError as error:
            assert "缺少原始数据快照" in str(error)
        else:
            raise AssertionError("A legacy run must request reanalysis instead of mixing data")


if __name__ == "__main__":
    test_literature_process_only()
    print("PASS: APS 32-row process data trains, exports figures and skips unsupported Pareto")
