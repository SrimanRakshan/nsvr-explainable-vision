from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import os, sys, json
from tqdm import tqdm

SRC_DIR   = Path(__file__).resolve().parents[1]
PROJ_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from reasoning.reason_clevr_soft import SoftExecutor

CLEVR_ROOT = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0")
GRAPH_DIR  = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed")
OUT_DIR    = Path(os.environ.get("OUT_DIR",    PROJ_ROOT / "data" / "results"))
SPLIT      = "val"

def load_graphs(path: Path) -> Dict[int, Dict[str,Any]]:
    gid = {}
    with open(path, "r") as f:
        for line in f:
            g = json.loads(line)
            gid[g["image_id"]] = g
    return gid

def load_questions(clevr_root: Path, split: str) -> List[Dict[str, Any]]:
    with open(clevr_root / "questions" / f"CLEVR_{split}_questions.json", "r") as f:
        return json.load(f)["questions"]

def qtype_from_program(prog: List[Dict[str,Any]]) -> str:
    ops = [s.get("type") or s.get("function") for s in prog]
    if "count" in ops: return "count"
    if "exist" in ops: return "exist"
    if any(o in ops for o in ("greater_than","less_than","equal_integer")): return "compare_integer"
    if any((o or "").startswith("query_") for o in ops): return "query_attr"
    return "other"

def norm_bool(x: float) -> str:
    return "yes" if x >= 0.5 else "no"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = GRAPH_DIR / f"pred_scene_graphs_{SPLIT}.jsonl"
    graphs = load_graphs(pred_path)
    qs = load_questions(CLEVR_ROOT, SPLIT)

    tot = cor = 0
    confs = []
    for q in tqdm(qs, desc="Soft VQA"):
        g = graphs.get(q["image_index"])
        if not g: 
            continue
        ex = SoftExecutor(g)
        out, conf, trace = ex.execute(q["program"])
        qtyp = qtype_from_program(q["program"])
        gt = str(q.get("answer"))

        # interpret prediction by type
        if qtyp == "query_attr":
            pred = out[0] if isinstance(out, tuple) else str(out)
        elif qtyp == "exist":
            if isinstance(out, (int, float)):
                pred = norm_bool(float(out))
            else:
                pred = str(out)
        elif qtyp == "compare_integer":
            # executor returns truth degree 0.0/1.0
            if isinstance(out, (int, float)):
                pred = norm_bool(float(out))
            else:
                pred = str(out)
        elif qtyp == "count":
            # expected count -> nearest int
            if isinstance(out, (int, float)):
                pred = str(int(round(float(out))))
            else:
                pred = str(out)
        else:
            # fallback
            pred = out[0] if isinstance(out, tuple) else (norm_bool(out) if isinstance(out, (int,float)) else str(out))

        ok = int(str(pred) == gt)
        tot += 1; cor += ok; confs.append(float(conf))

    acc = cor / max(1, tot)
    avg_conf = sum(confs)/max(1,len(confs))
    res = {"total": tot, "correct": cor, "acc": acc, "avg_conf": avg_conf}
    with open(OUT_DIR / "vqa_soft_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved:", OUT_DIR / "vqa_soft_summary.json")
    print(f"Acc: {acc:.4f} | Avg confidence: {avg_conf:.3f}")

if __name__ == "__main__":
    main()