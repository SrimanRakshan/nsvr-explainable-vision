from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import os, sys, json, csv
from tqdm import tqdm

# ----- repo paths -----
SRC_DIR   = Path(__file__).resolve().parents[1]
PROJ_ROOT = SRC_DIR.parent
sys.path.append(str(SRC_DIR))

from reasoning.reason_clevr import CLEVRExecutor  # uses your current executor

def P(x): 
    return x if isinstance(x, Path) else Path(x)

# ----- config -----
CLEVR_ROOT = P(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0")
GRAPH_DIR  = P(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed")
OUT_DIR    = P(os.environ.get("OUT_DIR",    PROJ_ROOT / "data" / "results"))
SPLIT      = "val"

def load_graphs(path: str | Path) -> Dict[int, Dict[str,Any]]:
    path = P(path)
    gid = {}
    if not path.exists():
        return gid
    with open(path, "r") as f:
        for line in f:
            g = json.loads(line)
            gid[g["image_id"]] = g
    return gid

def load_questions(clevr_root: str | Path, split: str) -> List[Dict[str, Any]]:
    clevr_root = P(clevr_root)
    with open(clevr_root / "questions" / f"CLEVR_{split}_questions.json", "r") as f:
        return json.load(f)["questions"]

def qtype_from_program(prog: List[Dict[str,Any]]) -> str:
    ops = [s.get("type") or s.get("function") for s in prog]
    if "count" in ops: return "count"
    if "exist" in ops: return "exist"
    if any(o in ops for o in ("greater_than","less_than","equal_integer")): return "compare_integer"
    if any(o and o.startswith("query_") for o in ops): return "query_attr"
    return "other"

def evaluate(graphs_by_id: Dict[int, Dict[str,Any]], questions: List[Dict[str,Any]], tag: str) -> Dict[str,Any]:
    total = correct = 0
    by_type = {}
    rows = []

    for q in tqdm(questions, desc=f"VQA [{tag}]"):
        img_id = q["image_index"]
        if img_id not in graphs_by_id:
            continue
        typ = qtype_from_program(q["program"])
        trace = []
        ex = CLEVRExecutor(graphs_by_id[img_id])
        try:
            pred = ex.execute(q["program"], trace=trace)
            pred_norm = "yes" if isinstance(pred, bool) and pred else ("no" if isinstance(pred, bool) else str(pred))
        except Exception as e:
            pred_norm = f"<ERROR:{type(e).__name__}>"
        gt_norm = str(q.get("answer"))
        ok = int(pred_norm == gt_norm)

        total += 1
        correct += ok
        bt = by_type.setdefault(typ, {"total":0,"correct":0})
        bt["total"] += 1; bt["correct"] += ok

        if len(rows) < 200:  # cap CSV size
            rows.append({
                "question_index": q["question_index"],
                "image_index": img_id,
                "type": typ,
                "question": q["question"],
                "answer_gt": gt_norm,
                "answer_pred": pred_norm,
                "correct": ok,
                "tag": tag,
            })

    acc = correct / max(total,1)
    for k,v in by_type.items():
        v["acc"] = v["correct"] / max(v["total"],1)
    return {"tag": tag, "total": total, "correct": correct, "acc": acc, "by_type": by_type, "rows": rows}

def main():
    print("=== Evaluate VQA (Symbolic) ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qs = load_questions(CLEVR_ROOT, SPLIT)

    gt_path   = GRAPH_DIR / f"scene_graphs_{SPLIT}.jsonl"
    pred_path = GRAPH_DIR / f"pred_scene_graphs_{SPLIT}.jsonl"

    gt_graphs   = load_graphs(gt_path)
    pred_graphs = load_graphs(pred_path)

    results = []
    if gt_graphs:
        res_gt = evaluate(gt_graphs, qs, tag="GT")
        results.append(res_gt)
    else:
        print(f"Warning: missing {gt_path}")

    if pred_graphs:
        res_pred = evaluate(pred_graphs, qs, tag="PRED")
        results.append(res_pred)
    else:
        print(f"Warning: missing {pred_path}")

    # save summary
    summary = {r["tag"]: {k:v for k,v in r.items() if k not in ("rows",)} for r in results}
    with open(OUT_DIR / "vqa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", OUT_DIR / "vqa_summary.json")

    # save sample rows CSV
    csv_path = OUT_DIR / "vqa_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tag","question_index","image_index","type","question","answer_gt","answer_pred","correct"])
        w.writeheader()
        for r in results:
            for row in r["rows"]:
                w.writerow(row)
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()
