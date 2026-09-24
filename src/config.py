from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_schema():
    with open(ROOT / "config" / "data_schema.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_objectives():
    with open(ROOT / "config" / "objectives.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def groups(schema):
    return {k:list(v.keys()) for k,v in schema.items()}
