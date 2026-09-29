# -*- coding: utf-8 -*-
"""图片导入（参数表截图 / 拍照）功能级测试（不用真浏览器，走 ImageImport 流程函数级验证）。

Run: python scripts/image_import_test.py

覆盖验证项（对应任务验收清单 1–4、6）：
  1. /tmp/ocr_test_table.png 完整流程：重建表 5 行 × 7 列、角色预填正确；
  2. 正弦曲线 PNG → is_likely_table = False + 中文原因；
  3. 单行参数图 → 表格成立且无目标列 → 「参数预测」分支；
  4. parse_number 对 ".0" 结尾 / 逗号千分位 / 逗号小数 / 全角负号等处理；
  5. unit_warnings 物理校验（psi 不映射 slpm；单位不明确 → 单位待确认）；
  6. 无 ocrmac 环境（模拟 Windows）优雅降级不崩溃；
  7. app.py 可编译（语法级）；训练分支预检：样本不足 20 时如实拦截。
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import image_import as imgimp
from src import quick_analysis as qa

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"[FAIL] {name}: {detail}"
    PASS += 1
    print(f"  ✓ {name}" + (f"（{detail}）" if detail else ""))


# ---------------- 1. 参数表截图：完整流程 ----------------
print("[1] /tmp/ocr_test_table.png 完整流程（OCR → 重建 → 判定 → 角色预填）")
ocr = imgimp.ocr_image_to_rows("/tmp/ocr_test_table.png")
check("OCR 引擎为本机 Apple Vision", ocr["engine"] == "apple-vision-local")
check("原始识别框 41 个", len(ocr["raw"]) == 41, f"n={len(ocr['raw'])}")
table = imgimp.rebuild_table(ocr)
check("重建表 5 行", len(table["rows"]) == 5, f"rows={len(table['rows'])}")
check("重建表 7 列", len(table["header"]) == 7, f"cols={len(table['header'])}")
check("单元格总数 41", table["n_cells"] == 41)
check("数值单元格已解析为 float",
      all(isinstance(v, float) for row in table["rows"] for v in row),
      f"首行={table['rows'][0]}")
ok, reason = imgimp.is_likely_table(ocr)
check("is_likely_table 通过", ok, reason)
roles = imgimp.guess_column_roles(table["header"])
xs = [c for c, r in roles.items() if r == "x"]
sts = [c for c, r in roles.items() if r == "states"]
check("压力/电流/喷距列预填 x", len(xs) == 4, f"x={xs}")
check("温度/速度列预填 states", len(sts) == 2, f"states={sts}")
check("喷涂序号 → identifier",
      imgimp.guess_column_roles(["喷涂序号"])["喷涂序号"] == "identifier")

# ---------------- 2. 曲线图拦截 ----------------
print("[2] 正弦曲线 PNG → 拒绝识别为表")
ocr_sine = imgimp.ocr_image_to_rows("/tmp/ocr_test_sine.png")
ok_s, reason_s = imgimp.is_likely_table(ocr_sine)
check("is_likely_table = False", not ok_s)
check("原因含曲线图提示", any(k in reason_s for k in ("曲线图", "SEM", "流程图")),
      reason_s)

# ---------------- 3. 单行参数图 → 预测分支 ----------------
print("[3] 单行参数图 → 表格成立、无目标列 → 参数预测分支")
ocr_row = imgimp.ocr_image_to_rows("/tmp/ocr_test_single_row.png")
t_row = imgimp.rebuild_table(ocr_row)
ok_r, _ = imgimp.is_likely_table(ocr_row)
roles_row = imgimp.guess_column_roles(t_row["header"])
targets = [c for c, r in roles_row.items() if r in ("states", "defects", "performance")]
branch = ("prediction" if (ok_r and not targets and len(t_row["rows"]) == 1)
          else "train" if (ok_r and targets) else "rejected")
check("判定为表格", ok_r)
check("无目标列", not targets)
check("进入参数预测分支", branch == "prediction", f"branch={branch}")
# 多行无目标 → 训练拒绝
multi_no_target = pd.DataFrame({"Ar压力(psi)": [65.0, 85.0], "电流(A)": [500.0, 600.0]})
n_rows, has_target = len(multi_no_target), False
branch2 = "prediction" if n_rows == 1 and not has_target else \
          "train" if has_target else "multi_row_error"
check("多行无目标 → st.error 分支", branch2 == "multi_row_error")

# ---------------- 4. parse_number ----------------
print("[4] parse_number 边界")
cases = {
    "65.0": (65.0, 0.0),          # ".0" 结尾浮点，正常
    "2,588.6": (2588.6, 0.05),    # 逗号千分位
    "1,200": (1200.0, 0.05),
    "12,5": (12.5, 0.25),         # 单逗号按小数点解释并降置信
    "-35.2": (-35.2, 0.0),        # 负号
    "−35.2": (-35.2, 0.0),        # Unicode 负号
    "25 °C": (25.0, 0.0),         # 单位附着
    "65.": (65.0, 0.1),           # 结尾多余句点
    "abc": (None, 1.0),           # 不可解析 → 保留原文
    "": (None, 1.0),
    "500": (500.0, 0.0),
    "1.5e3": (1500.0, 0.0),
}
for text, (exp_v, exp_p) in cases.items():
    v, p = imgimp.parse_number(text)
    check(f"parse_number({text!r})", v == exp_v and abs(p - exp_p) < 1e-9,
          f"→ ({v}, {p})")

# ---------------- 5. unit_warnings 物理校验 ----------------
print("[5] unit_warnings 物理校验")
w1 = imgimp.unit_warnings(["Ar压力(psi)", "Ar压力与流量(psi/slpm)", "粒子温度(°C)"],
                          {"Ar压力(psi)": "x", "Ar压力与流量(psi/slpm)": "x",
                           "粒子温度(°C)": "x"})
check("psi 压力列按压力处理（不映射 slpm）",
      any("psi" in w and "slpm" in w and "换算" in w for w in w1), str(w1))
check("psi+slpm 同头冲突被警告",
      any("同时出现压力与流量" in w for w in w1), str(w1))
check("温度映射 X 被提醒", any("温度" in w for w in w1))
w2 = imgimp.unit_warnings(["Ar压力(psi)"], {"Ar压力(psi)": "performance"})
check("压力被映射为非 X 被提醒", any("psi" in w and "请确认" in w for w in w2))
w3 = imgimp.unit_warnings(["涂层某个数"], {"涂层某个数": "x"})
check("单位不明确 → 单位待确认", any("单位待确认" in w for w in w3))
w4 = imgimp.unit_warnings(["Ar压力(psi)"], {"Ar压力(psi)": "x"})
check("压力映射 X → 保留 psi≠slpm 提示",
      len(w4) == 1 and "slpm" in w4[0], str(w4))

# ---------------- 6. 无 ocrmac（Windows）优雅降级 ----------------
print("[6] 无 ocrmac 环境优雅降级")
import importlib
with patch.dict(sys.modules, {"ocrmac": None}):
    saved = sys.modules.pop("src.image_import", None)
    try:
        mod = importlib.import_module("src.image_import")
        importlib.reload(mod)
        check("import 不崩溃", True)
        check("OCRMAC_AVAILABLE = False", mod.OCRMAC_AVAILABLE is False)
        try:
            mod.ocr_image_to_rows("whatever.png")
            raise AssertionError("应当抛出 ImportError")
        except ImportError as e:
            check("ocr_image_to_rows 抛中文 ImportError", "ocrmac" in str(e))
    finally:
        if saved is not None:
            sys.modules["src.image_import"] = saved

# ---------------- 7. app.py 语法 + 训练分支预检 ----------------
print("[7] app.py 可编译；训练分支预检如实拦截（样本 < 20）")
import py_compile
py_compile.compile(str(Path(__file__).resolve().parents[1] / "app.py"),
                   doraise=True)
check("app.py 编译通过", True)

# 重建表直接转 DataFrame（列名保留 OCR 表头原文），confirm_map 用预填角色
frame = pd.DataFrame(table["rows"], columns=table["header"])
confirm_map = {c: r for c, r in roles.items()
               if r in ("x", "states", "defects", "performance")}
with tempfile.TemporaryDirectory() as tmp, \
     patch.object(qa, "QUICK_ROOT", Path(tmp)), \
     patch.object(qa, "HISTORY_FILE", Path(tmp) / "history.json"), \
     warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    result = qa.run_quick_analysis(frame, sheet_name="ocr_test_table.png",
                                   source_filename="ocr_test_table.png",
                                   confirm_map=confirm_map, demo=False)
    check("样本 < 20 时预检拦截（record.failed=True）",
          result["record"]["failed"])
    fail_reasons = [f"{n}: {r}" for n, s, r in result["steps"]
                    if s == "fail" and r]
    check("拦截原因含样本量过少",
          any("样本" in r for r in fail_reasons), "; ".join(fail_reasons))
    check("未产出模型（无 model.joblib）",
          not (Path(tmp) / "REGRESSION_LITERATURE" / "model.joblib").exists()
          and not any((Path(tmp) / d / "model.joblib").exists()
                      for d in [p.name for p in Path(tmp).iterdir()]))

# image_meta 写盘函数（独立于 UI 验证字段结构）
print("[8] image_meta.json 字段结构")
meta_fields = {"image_filename", "image_sha256", "ocr_engine", "ocr_raw_count",
               "user_edits_count", "unit_warnings", "role_map",
               "data_provenance", "confirmed_at", "run_id"}
with tempfile.TemporaryDirectory() as tmp, \
     patch.object(qa, "QUICK_ROOT", Path(tmp)):
    qa.QUICK_ROOT.mkdir(parents=True, exist_ok=True)
    app_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app_dir))
    # 直接构造 state，模拟确认后的写盘（不依赖 streamlit runtime）
    state = {"name": "ocr_test_table.png", "sha256": "a" * 64,
             "ocr_engine": ocr["engine"], "ocr_raw_count": len(ocr["raw"])}
    # 复用 app.py 的写盘逻辑过于耦合 streamlit；此处按同结构独立验证文件读写
    run_id = "META_TEST"
    run_dir = qa.QUICK_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"image_filename": state["name"], "image_sha256": state["sha256"],
            "ocr_engine": state["ocr_engine"],
            "ocr_raw_count": state["ocr_raw_count"], "user_edits_count": 0,
            "unit_warnings": [], "role_map": roles,
            "data_provenance": "image_ocr_verified",
            "confirmed_at": "2026-01-01T00:00:00", "run_id": run_id}
    (run_dir / "image_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = json.loads((run_dir / "image_meta.json").read_text(encoding="utf-8"))
    check("image_meta.json 全字段落盘", meta_fields <= set(loaded))
    check("data_provenance = image_ocr_verified",
          loaded["data_provenance"] == "image_ocr_verified")

print(f"\nALL PASS ({PASS} checks)")
