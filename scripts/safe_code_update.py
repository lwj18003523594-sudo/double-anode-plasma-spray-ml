"""Update program files from a complete checkout without touching research data.

Run from the newly downloaded repository, not the potentially mixed installation:
    python3 scripts/safe_code_update.py --target ~/Downloads/DoubleAnode_ML_Foolproof
"""
import argparse
import ast
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUICK_API = {"latest_runs", "quick_analyze_file", "load_smart_summary",
             "save_quick_workspace", "load_quick_workspace"}
ROOT_FILES = ("app.py", "VERSION", "requirements.txt", "README_先看我.md",
              "SYNC_GUIDE.md", "01_首次安装.command", "02_启动平台.command",
              "03_数据模板位置.command", "04_同步源码.command",
              "01_首次安装_Windows.bat", "02_启动平台_Windows.bat")


def source_files(source):
    files = [source / name for name in ROOT_FILES if (source / name).is_file()]
    for part in ("src", "scripts"):
        files.extend(p for p in (source / part).rglob("*.py") if p.is_file())
    files.extend(p for p in (source / "scripts").glob("*.sh") if p.is_file())
    return sorted(files)


def validate_source(source):
    for name in ("app.py", "VERSION", "src/quick_analysis.py",
                 "scripts/platform_check.py", "scripts/safe_code_update.py"):
        if not (source / name).is_file():
            raise ValueError(f"下载的代码包不完整：缺少 {name}")
    tree = ast.parse((source / "src/quick_analysis.py").read_text(encoding="utf-8"))
    actual = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = QUICK_API - actual
    if missing:
        raise ValueError("下载的代码包与新版首页不匹配：缺少 " + "、".join(sorted(missing)))
    for p in source_files(source):
        if p.suffix == ".py":
            compile(p.read_bytes(), str(p), "exec")


def apply_source_update(source, target):
    source, target = Path(source).expanduser().resolve(), Path(target).expanduser().resolve()
    if source == target:
        raise ValueError("源目录与运行目录相同；请先下载完整代码 ZIP 并解压，再从新目录运行同步脚本")
    if not (target / "app.py").is_file():
        raise ValueError(f"找不到现有平台目录：{target}（缺少 app.py）")
    validate_source(source)  # Validate everything before modifying the installation.
    files = source_files(source)
    backup = target / "backups" / ("code_before_update_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    staging = Path(tempfile.mkdtemp(prefix=".code-update-", dir=target))
    replaced = []
    try:
        for p in files:
            relative = p.relative_to(source)
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, staged)
        for p in files:
            relative = p.relative_to(source)
            old = target / relative
            if old.exists():
                dest = backup / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, dest)
        for p in files:
            relative = p.relative_to(source)
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / relative, dest)
            replaced.append(relative)
        (backup / "update_manifest.json").write_text(json.dumps({
            "source": str(source), "target": str(target),
            "updated": [str(p) for p in replaced],
            "preserved": [".venv", "data", "models", "runs", "outputs", "config"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        for relative in reversed(replaced):
            old_backup = backup / relative
            if old_backup.is_file():
                shutil.copy2(old_backup, target / relative)
            else:
                (target / relative).unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return backup, len(replaced)


def main():
    parser = argparse.ArgumentParser(description="仅更新平台源码，保留本机实验数据和运行结果")
    parser.add_argument("--target", required=True, help="现有平台文件夹")
    args = parser.parse_args()
    backup, count = apply_source_update(ROOT, args.target)
    print(f"同步完成：{count} 个程序文件。旧程序备份：{backup}")
    print("数据、模型、分析结果、配置及 Python 虚拟环境未覆盖。")
    print("请退出旧版 Streamlit 进程后，重新双击平台启动按钮。")


if __name__ == "__main__":
    main()
