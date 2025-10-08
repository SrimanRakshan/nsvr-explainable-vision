# src/reasoning/preview_explanations.py
from __future__ import annotations
from pathlib import Path
import os, sys, json

# --- make src importable ---
SRC_DIR   = Path(__file__).resolve().parents[1]   # .../src
PROJ_ROOT = SRC_DIR.parent                        # repo root
sys.path.append(str(SRC_DIR))

from reasoning.reason_clevr_dag import CLEVRExecutorDAG
from reasoning.explainer import trace_to_explanation

def P(x): return x if isinstance(x, Path) else Path(x)

# --- resolve paths (env overrides allowed) ---
CLEVR_ROOT = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0")
GRAPH_DIR  = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed")
GRAPH_PATH = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed\scene_graphs_val.jsonl")
if not GRAPH_PATH.exists():
    alt = GRAPH_DIR / "pred_scene_graphs_val.jsonl"
    if alt.exists():
        GRAPH_PATH = alt
    else:
        raise FileNotFoundError(
            f"Could not find graphs at:\n  {GRAPH_PATH}\n  {alt}\n"
            f"Tip: set GRAPH_PATH env var to a specific file."
        )
QUESTIONS = CLEVR_ROOT / "questions" / "CLEVR_val_questions.json"

def load_graphs(p: Path):
    gid = {}
    with open(p, "r") as f:
        for line in f:
            g = json.loads(line)
            gid[g["image_id"]] = g
    return gid

def main(n=5):
    print("Using:")
    print("  PROJ_ROOT :", PROJ_ROOT)
    print("  GRAPH_PATH:", GRAPH_PATH)
    print("  QUESTIONS :", QUESTIONS)

    graphs = load_graphs(GRAPH_PATH)
    qs = json.load(open(QUESTIONS, "r"))["questions"]

    shown = 0
    for q in qs:
        img = q["image_index"]
        if img not in graphs:
            continue
        ex = CLEVRExecutorDAG(graphs[img])
        ans, trace = ex.execute(q["program"])
        ans_str = "yes" if isinstance(ans, bool) and ans else ("no" if isinstance(ans, bool) else str(ans))
        exp = trace_to_explanation(trace, graphs[img], ans_str, q["question"])
        print("\n---")
        print(exp)
        shown += 1
        if shown >= n:
            break

if __name__ == "__main__":
    main(5)
