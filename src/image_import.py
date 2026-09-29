# -*- coding: utf-8 -*-
"""图片导入（参数表截图 / 手机拍照）→ 本地 OCR → 表格重建 → 角色预填。

技术路线（本机已验证，勿改其他路线）：
  ocrmac 1.0.1（Apple Vision 本地 OCR）：
      ocrmac.OCR(path, language_preference=['zh-Hans', 'en-US'],
                 recognition_level='accurate').recognize()
  返回 [(text, confidence, (x, y, w, h)), ...]，坐标归一化（y 自图片底部起）。
  表格重建：按 y 坐标聚类（容差 ~0.03）成行，行内按 x 排序成列。

隐私与安全约束：
  - 全部识别在本机（Apple Vision）完成，任何图片都不会上传到在线服务。
  - 不造数据点：OCR 解析失败的一律保留原文并标注低置信，绝不填充、插值或猜测。
  - 曲线图 / SEM 照片 / 流程图一律在 is_likely_table 拦截，绝不强行识别为表。
  - 压力单位（psi 等）绝不换算为气体流量（slpm）；unit_warnings 只提示，不做换算。

Windows / Linux 降级：ocrmac 仅 macOS 可用；import 失败时本模块不崩溃，
OCRMAC_AVAILABLE = False，ocr_image_to_rows 抛出带中文说明的 ImportError，
调用方（app.py）据此隐藏图片导入入口。
"""
import re
from pathlib import Path

# ocrmac 为 macOS 专属依赖；缺失时优雅降级（不崩溃，由调用方隐藏入口）
try:
    from ocrmac import ocrmac  # noqa: N813
    OCRMAC_AVAILABLE = True
except Exception:  # ImportError 或底层框架缺失（非 macOS）
    ocrmac = None
    OCRMAC_AVAILABLE = False

ENGINE_NAME = "apple-vision-local"
LOW_CONF_THRESHOLD = 0.6      # 低于该置信度的单元格标记为「需人工核对」
ROW_Y_TOLERANCE = 0.03        # y 坐标聚类容差（归一化坐标）
MIN_TABLE_CELLS = 6           # 少于 6 个文本块 → 不视为表格
MIN_TABLE_ROWS = 2            # 少于 2 行 → 不视为表格
MIN_NUMERIC_RATIO = 0.3       # 数据单元格数值占比下限

OCR_LANGUAGE = ["zh-Hans", "en-US"]

# ---------------- OCR 调用 ----------------

def ocr_image_to_rows(image_path):
    """对本地图片执行 Apple Vision 本地 OCR，并按 y 聚类成行。

    返回 dict：
      cells  [[{"text","conf","x","y","h"}...]]  聚类后的行（自上而下），行内按 x 升序
      raw    [(text, conf, [x, y, w, h])...]     ocrmac 原始识别结果
      engine "apple-vision-local"
      image_path 识别所用图片路径
    """
    if not OCRMAC_AVAILABLE:
        raise ImportError(
            "图片识别依赖 ocrmac（Apple Vision 本地 OCR），仅 macOS 可用；"
            "当前环境不可用，请改用 Excel / CSV 导入，或在 macOS 上使用本入口。")
    path = str(image_path)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    results = ocrmac.OCR(path, language_preference=OCR_LANGUAGE,
                         recognition_level="accurate").recognize()
    raw = []
    items = []
    for text, conf, box in results:
        t = str(text).strip()
        if not t:
            continue
        x, y, w, h = [float(v) for v in box]
        c = float(conf) if conf is not None else 0.0
        raw.append((t, c, [x, y, w, h]))
        items.append({"text": t, "conf": c, "x": x, "y": y, "h": h})
    rows = _cluster_rows(items)
    return {"cells": rows, "raw": raw, "engine": ENGINE_NAME, "image_path": path}


def _cluster_rows(items, y_tol=ROW_Y_TOLERANCE):
    """按 y 中心聚类成行（y 自底部起，值大者在上）；行内按 x 升序。"""
    if not items:
        return []
    ordered = sorted(items, key=lambda c: (-(c["y"] + c.get("h", 0.0) / 2.0), c["x"]))
    rows, cur, cur_cy = [], [], None
    for cell in ordered:
        cy = cell["y"] + cell.get("h", 0.0) / 2.0
        if cur_cy is None or abs(cy - cur_cy) <= y_tol:
            # 行中心取滑动平均，避免整行整体倾斜时被拆成多行
            cur_cy = cy if cur_cy is None else (cur_cy * len(cur) + cy) / (len(cur) + 1)
            cur.append(cell)
        else:
            cur.sort(key=lambda d: d["x"])
            rows.append(cur)
            cur, cur_cy = [cell], cy
    if cur:
        cur.sort(key=lambda d: d["x"])
        rows.append(cur)
    return rows


# ---------------- 数值解析 ----------------

# 结尾附着的常见物理单位（长 token 优先匹配；"°C/psi 等单位附着"场景）
_TRAILING_UNIT_RE = re.compile(
    r"\s*(?:°\s*[cfk]|℃|℉|degc|psf|psi|kpa|mpa|gpa|pa|bar|mbar|slpm|sccm|l/min|lpm|"
    r"mm|cm|µm|μm|um|dm|km|m/s|km/h|kv|mv|kw|mw|khz|mhz|hz|rpm|g/min|mg/s|kg/h|"
    r"hv|hk|hb|°|a|v|w|s|min|h)$", re.IGNORECASE)

_THOUSANDS_RE = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_COMMA_DECIMAL_RE = re.compile(r"^[-+]?\d+,\d+$")
_PLAIN_NUMBER_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")


def parse_number(text):
    """解析数值单元格；返回 (value, confidence_penalty)。

    - 解析失败 → (None, 1.0)，调用方必须保留原文，绝不造数。
    - 逗号作千分位（如 2,588.6）→ 可解析，小罚 0.05。
    - 单个逗号更可能是小数点（如 12,5）→ 按小数点解释，罚 0.25。
    - ".0" 结尾浮点（如 65.0）属正常浮点，不罚。
    - 结尾多余句点（65.）→ 去掉后解析，罚 0.1。
    - 负号的全角/Unicode 变体（−、–、—）自动归一为 -。
    """
    if text is None:
        return (None, 1.0)
    s = str(text).strip()
    if not s:
        return (None, 1.0)
    # 全角符号归一
    for a, b in (("（", "("), ("）", ")"), ("，", ","), ("．", "."),
                 ("。", "."), ("：", ":"), ("−", "-"), ("–", "-"), ("—", "-"),
                 ("＋", "+"), ("　", " ")):
        s = s.replace(a, b)
    s = s.strip()
    # 去掉附着单位（最多循环两次，覆盖 "25 mm2" 类复合情况）
    had_unit = False
    for _ in range(2):
        stripped = _TRAILING_UNIT_RE.sub("", s).strip()
        if stripped == s:
            break
        s, had_unit = stripped, True
    if not s:
        return (None, 1.0)
    penalty = 0.0
    if s.endswith(".") and s[:-1].lstrip("-").isdigit():
        s, penalty = s[:-1], 0.1
    if _THOUSANDS_RE.match(s):
        return (float(s.replace(",", "")), max(penalty, 0.05))
    if _COMMA_DECIMAL_RE.match(s):
        return (float(s.replace(",", ".")), max(penalty, 0.25))
    if _PLAIN_NUMBER_RE.match(s):
        return (float(s), penalty)
    return (None, 1.0)


# ---------------- 表格重建 ----------------

def rebuild_table(ocr_result):
    """把 OCR 结果重建为表格。

    返回 dict：
      header  [str, ...]                首行文本作表头（不足列数时补「列N」）
      rows    [[value, ...], ...]       数据行；可解析为数值的存数值，否则存原文
      low_confidence  [(row, col, text, conf), ...]
                                        需人工核对的单元格（row 0 = 表头行；
                                        置信度 conf<0.6，或解析含糊按
                                        effective = conf × (1 − penalty) 判定）
      n_cells int                       识别到的单元格总数
    """
    cells = ocr_result.get("cells") or []
    # 兼容扁平输入（单层 list of cell dict）：先聚类
    if cells and isinstance(cells[0], dict):
        cells = _cluster_rows(list(cells))
    if not cells:
        return {"header": [], "rows": [], "low_confidence": [], "n_cells": 0}

    n_cols = max(len(r) for r in cells)
    header_cells = cells[0]
    header = [str(c["text"]).strip() for c in header_cells]
    if len(header) < n_cols:
        pad = ["列%d" % (i + 1) for i in range(n_cols - len(header))]
        # 首个数据列明显位于表头最左单元格左侧 → 缺失表头在行首（如索引列）
        first_data_x = cells[1][0]["x"] if len(cells) > 1 and cells[1] else None
        first_header_x = header_cells[0]["x"] if header_cells else None
        if (first_data_x is not None and first_header_x is not None
                and first_data_x < first_header_x - 0.015):
            header = pad + header
        else:
            header = header + pad

    low_confidence = []
    for col, c in enumerate(header_cells):
        conf = float(c.get("conf", 0.0))
        if conf < LOW_CONF_THRESHOLD:
            low_confidence.append((0, col, c["text"], round(conf, 2)))

    rows = []
    for r, row in enumerate(cells[1:], start=1):
        values = []
        for col in range(n_cols):
            if col >= len(row):
                values.append("")
                continue
            cell = row[col]
            text = str(cell["text"]).strip()
            conf = float(cell.get("conf", 0.0))
            value, penalty = parse_number(text)
            if value is None:
                values.append(text)          # 解析失败保留原文，绝不造数
            else:
                values.append(value)
                if penalty > 0:
                    # 解析含糊（千分位/逗号/句点等）→ 降低有效置信度，交用户核对
                    effective = conf * (1.0 - penalty)
                    if effective < LOW_CONF_THRESHOLD:
                        low_confidence.append((r, col, text, round(effective, 2)))
            if conf < LOW_CONF_THRESHOLD and (r, col, text, round(conf, 2)) \
                    not in low_confidence:
                low_confidence.append((r, col, text, round(conf, 2)))
        rows.append(values)

    return {"header": header, "rows": rows, "low_confidence": low_confidence,
            "n_cells": sum(len(r) for r in cells)}


def _flatten_cells(ocr_result):
    """取单元格扁平列表（兼容嵌套行 / 扁平两种输入）。"""
    cells = ocr_result.get("cells") or []
    if cells and isinstance(cells[0], dict):
        return list(cells)
    return [c for row in cells for c in row]


# ---------------- 表格判定 ----------------

def is_likely_table(ocr_result):
    """判断 OCR 结果是否像参数表；返回 (bool, 中文原因)。

    以下任一成立即判非表格（曲线图 / SEM 照片 / 流程图等）：
      - 文本块 < 6 个；
      - 聚类行数 < 2；
      - 含 ≥3 个单元格的「实质行」不足 2 行（曲线图坐标刻度常为单元素行）；
      - 数据单元格数值占比 < 0.3（文字型截图，如文献段落 / 流程图标注）。
    """
    cells = ocr_result.get("cells") or []
    if cells and isinstance(cells[0], dict):
        cells = _cluster_rows(list(cells))
    flat = [c for row in cells for c in row]
    n_cells = len(flat)
    n_rows = len(cells)
    if n_cells < MIN_TABLE_CELLS:
        return (False, f"仅识别到 {n_cells} 个文本块（少于 {MIN_TABLE_CELLS} 个），"
                       "不像参数表——可能是曲线图、SEM 照片或流程图截图。")
    if n_rows < MIN_TABLE_ROWS:
        return (False, f"识别到的文本聚成 {n_rows} 行（不足两行），无法构成表格——"
                       "请确认截图是行列清晰的参数表，而非曲线图、SEM 照片或流程图。")
    substantial = [r for r in cells if len(r) >= 3]
    if len(substantial) < 2:
        return (False, "文本块过于分散（每行不足 3 个文本块的行数过多），"
                       "更像曲线图、SEM 照片或流程图，而不是规整的参数表。")
    data_flat = [c for row in cells[1:] for c in row]
    if data_flat:
        numeric = sum(1 for c in data_flat if parse_number(c["text"])[0] is not None)
        ratio = numeric / len(data_flat)
        if ratio < MIN_NUMERIC_RATIO:
            return (False, f"数据单元格中仅 {ratio:.0%} 可解析为数值"
                           f"（低于 {MIN_NUMERIC_RATIO:.0%}）——更像文字型截图"
                           "（曲线图坐标、SEM 标注、流程图说明或文献段落），"
                           "而不是参数表。")
    return (True, f"识别到 {n_rows - 1} 行 × {n_cols_of(cells)} 列，"
                  f"共 {n_cells} 个文本块，数值占比符合参数表特征。")


def n_cols_of(cells):
    """单元格行集合的列数（最大行宽）。"""
    return max((len(r) for r in cells), default=0)


# ---------------- 表头角色预填 ----------------

def _normalize_header(s):
    """表头归一：小写、全角括号/摄氏度符号归一、去空白。"""
    s = str(s).strip().lower()
    s = (s.replace("（", "(").replace("）", ")").replace("℃", "°c")
          .replace("°c", "°c").replace("μ", "u"))
    s = re.sub(r"\s+", "", s)
    return s


_ROLE_BY_SECTION = {
    "structure_inputs": "x", "process_inputs": "x",
    "process_states": "states", "defect_network": "defects",
    "performance_outputs": "performance", "material_inputs": "material",
}


def guess_column_roles(header):
    """按表头预填各列角色；未命中返回 None（交用户人工确认）。

    复用 quick_analysis 的别名体系：
      1. 精确匹配 IDENTIFIER_COLUMNS / LITERATURE_ROLE_ALIASES / 规范字段 / GENERIC_ALIAS；
      2. 兜底：单位骨架匹配——OCR 常把中文表头识别错（如「压力」→「国国」），
         但括号内的单位通常可识别，故按「(单位)」骨架匹配别名表
         （"ar国国(psi)" ↔ "ar压力(psi)" → x）。
    返回 {原表头: "x"/"states"/"defects"/"performance"/"material"/"identifier"/None}。
    """
    from .quick_analysis import (GENERIC_ALIAS, IDENTIFIER_COLUMNS,
                                 LITERATURE_ROLE_ALIASES, _all_canonical_fields)
    canon = {k.lower(): v for k, v in _all_canonical_fields().items()}
    # 单位骨架 → 角色（仅取带括号单位的别名）
    skeleton_roles = {}
    for alias_key, role in LITERATURE_ROLE_ALIASES.items():
        m = re.search(r"\(([^()]*)\)$", _normalize_header(alias_key))
        if m and m.group(1) and m.group(1) not in skeleton_roles:
            skeleton_roles[m.group(1)] = role
    for idc in IDENTIFIER_COLUMNS:
        m = re.search(r"\(([^()]*)\)$", _normalize_header(idc))
        if m and m.group(1) and m.group(1) not in skeleton_roles:
            skeleton_roles[m.group(1)] = "identifier"

    roles = {}
    for col in header:
        key = _normalize_header(col)
        role = None
        if key in IDENTIFIER_COLUMNS:
            role = "identifier"
        elif key in LITERATURE_ROLE_ALIASES:
            role = LITERATURE_ROLE_ALIASES[key]
        elif key in canon:
            role = _ROLE_BY_SECTION.get(canon[key]["role"])
        else:
            alias = GENERIC_ALIAS.get(key)
            if alias:
                info = canon.get(str(alias).lower())
                role = _ROLE_BY_SECTION.get(info["role"]) if info else None
            if role is None:
                m = re.search(r"\(([^()]*)\)$", key)
                if m:
                    role = skeleton_roles.get(m.group(1))
        roles[col] = role
    return roles


# ---------------- 单位物理校验 ----------------

_PRESSURE_TOKENS = ("psi", "kpa", "mpa", "gpa", "mbar", "bar", "psf")
_FLOW_TOKENS = ("slpm", "sccm", "l/min", "lpm", "l/min")
_TEMPERATURE_TOKENS = ("°c", "℃", "°f")
_KNOWN_UNIT_TOKENS = sorted(
    ["°c", "℃", "°f", "psi", "kpa", "mpa", "gpa", "pa", "bar", "mbar",
     "slpm", "sccm", "l/min", "lpm", "m/s", "mm", "cm", "um", "μm",
     "kv", "mv", "kw", "mw", "khz", "mhz", "hz", "rpm", "g/min", "mg/s",
     "kg/h", "hv", "hk", "hb", "%", "a", "v", "w", "s", "min", "h"],
    key=len, reverse=True)


def _detect_unit(key):
    """从归一化表头里探测物理单位 token；探测不到返回 None。"""
    for u in _KNOWN_UNIT_TOKENS:
        i = key.find(u)
        if i < 0:
            continue
        if len(u) == 1:
            # 单字母单位必须出现在括号内（"(a)"）或紧跟数字/空格（"650A"），
            # 避免把 "bar"/"hard" 等单词中的字母误判为单位
            prev = key[i - 1] if i > 0 else ""
            nxt = key[i + len(u)] if i + len(u) < len(key) else ""
            if not (prev in "(-0123456789" or (prev in " (" and nxt in ")/")):
                continue
        return u
    return None


def unit_warnings(header, roles):
    """对参与建模的列做单位物理校验；返回中文警告列表（可为空）。

    规则：
      - 压力单位列绝不被当作气体流量：表头同时含压力与流量线索 → 明确警告；
      - 压力单位列被映射为非 X 角色 → 提醒确认（压力通常是工艺输入 X）；
      - 温度单位列被映射为输入 X → 提醒（粒子温度等通常是过程状态 states）；
      - 单位不明确 → 「单位待确认」。
    """
    warnings = []
    for col in header:
        role = roles.get(col)
        if role in (None, "identifier", "ignore"):
            continue
        key = _normalize_header(col)
        has_pressure = any(t in key for t in _PRESSURE_TOKENS)
        has_flow = (any(t in key for t in _FLOW_TOKENS) or "流量" in key)
        has_temperature = any(t in key for t in _TEMPERATURE_TOKENS)
        if has_pressure and has_flow:
            warnings.append(
                f"「{col}」的表头同时出现压力与流量线索：压力（psi 等）将按压力输入处理，"
                "绝不会按气体流量（slpm）换算或解读，请人工确认该列的物理含义。")
        elif has_pressure and role != "x":
            warnings.append(
                f"「{col}」为压力单位（psi 等），通常应为工艺输入 X；当前被标记为"
                f"「{role}」，请确认。")
        elif has_pressure:
            warnings.append(
                f"「{col}」为压力输入（psi 等）：按压力处理，绝不换算为气体流量"
                "（slpm），也不会并入任何流量类字段。")
        elif has_temperature and role == "x":
            warnings.append(
                f"「{col}」为温度单位，粒子温度类温度通常是过程状态（states）而非输入 X；"
                "当前被标记为输入 X，请确认。")
        elif _detect_unit(key) is None:
            warnings.append(
                f"「{col}」单位待确认：未能从表头识别出明确物理单位，"
                "请在进入分析前人工核对该列的单位与量纲。")
    return warnings
