"""
Step 5: Symbolic Reasoning over CLEVR functional programs (GROUND-TRUTH graphs)

Inputs:
  - data/processed/scene_graphs_{split}.jsonl     (from Step 4)
  - CLEVR_v1.0/questions/CLEVR_{split}_questions.json

Outputs:
  - prints overall accuracy
  - optional per-sample trace when --preview N > 0
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, List, Set

# --------- Adjust defaults to your paths ----------
DEFAULT_CLEVR_ROOT = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0"
DEFAULT_GRAPH_DIR  = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed"
DEFAULT_SPLIT      = "val"   # run on val for evaluation

# ------------ Graph I/O ------------
def load_graphs(graph_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load scene_graphs_{split}.jsonl into dict image_id -> graph."""
    gid2graph = {}
    with open(graph_path, "r") as f:
        for line in f:
            g = json.loads(line)
            gid2graph[g["image_id"]] = g
    return gid2graph

def load_questions(clevr_root: Path, split: str) -> List[Dict[str, Any]]:
    qpath = clevr_root / "questions" / f"CLEVR_{split}_questions.json"
    with open(qpath, "r") as f:
        data = json.load(f)
    return data["questions"]

# ------------ Executor ------------
class CLEVRExecutor:
    """
    Minimal executor for CLEVR-style functional programs on our scene graphs.
    Operates on sets of node indices and simple scalar answers.
    """
    def __init__(self, graph: Dict[str, Any]):
        self.g = graph
        self.nodes = graph["nodes"]
        # Build adjacency by relation: rel_name -> src -> set(dst)
        self.rels: Dict[str, Dict[int, Set[int]]] = {}
        for e in graph["edges"]:
            rel = e["rel"]
            self.rels.setdefault(rel, {}).setdefault(e["src"], set()).add(e["dst"])

    # --- helpers over node sets ---
    def _all(self) -> Set[int]:
        return set(range(len(self.nodes)))

    def _filter_attr(self, ids: Set[int], key: str, value: str) -> Set[int]:
        return {i for i in ids if self.nodes[i][key] == value}

    def _relate(self, ids: Set[int], rel: str) -> Set[int]:
        out: Set[int] = set()
        R = self.rels.get(rel, {})
        for i in ids:
            out |= R.get(i, set())
        return out

    def _unique(self, ids: Set[int]) -> int:
        if len(ids) != 1:
            raise ValueError(f"unique() expects 1, got {len(ids)}")
        return next(iter(ids))

    # --- primitive ops returning sets/ints/strings/bools ---
    def execute(self, program: List[Dict[str, Any]], trace: List[str] | None = None) -> Any:
        """
        program: list of {"type": op, "value_inputs": [...], "inputs": [...]}
        Returns final answer (str/int/bool) and optionally appends to trace.
        """
        # CLEVR programs are typically sequential; maintain a stack of intermediate results.
        stack: List[Any] = []

        def t(msg: str):
            if trace is not None: trace.append(msg)

        for idx, step in enumerate(program):
            op = step.get("type") or step.get("function")  # some variants use 'function'
            # normalize op names we handle
            if op in ("scene",):
                S = self._all()
                stack.append(S)
                t(f"{idx}: scene → |S|={len(S)}")

            elif op.startswith("filter_"):  # filter_color/shape/material/size
                # expected: last stack item is a set
                S = stack.pop()
                assert isinstance(S, set)
                # value is in 'value_inputs' or 'side_inputs' depending on variant
                val = _extract_value(step)
                key = op.replace("filter_", "")
                S2 = self._filter_attr(S, key, val)
                stack.append(S2)
                t(f"{idx}: {op}={val} → |S|={len(S2)}")

            elif op == "relate":
                # value is relation: left, right, front, behind → convert to *_of
                S = stack.pop()
                assert isinstance(S, set)
                rel_word = _extract_value(step)  # e.g., "left"
                rel_map = {"left":"left_of","right":"right_of","front":"front_of","behind":"behind"}
                rel = rel_map[rel_word]
                S2 = self._relate(S, rel)
                stack.append(S2)
                t(f"{idx}: relate({rel}) → |S|={len(S2)}")

            elif op == "intersect":
                B = stack.pop(); A = stack.pop()
                stack.append(A & B); t(f"{idx}: intersect → |S|={len(A & B)}")

            elif op == "union":
                B = stack.pop(); A = stack.pop()
                stack.append(A | B); t(f"{idx}: union → |S|={len(A | B)}")

            elif op == "unique":
                S = stack.pop(); nid = self._unique(S)
                stack.append(nid); t(f"{idx}: unique → node {nid}")

            elif op == "count":
                S = stack.pop(); c = len(S)
                stack.append(c); t(f"{idx}: count → {c}")

            elif op == "exist":
                S = stack.pop(); b = (len(S) > 0)
                stack.append(b); t(f"{idx}: exist → {b}")

            elif op.startswith("query_"):  # query_color/shape/size/material
                nid = stack.pop()
                assert isinstance(nid, int)
                key = op.replace("query_", "")
                val = self.nodes[nid][key]
                stack.append(val); t(f"{idx}: {op} → {val}")

            # comparisons (common in CLEVR)
            elif op == "equal_integer":
                b = stack.pop(); a = stack.pop()
                stack.append(int(a == b)); t(f"{idx}: equal_integer({a},{b}) → {int(a==b)}")

            elif op == "greater_than":
                b = stack.pop(); a = stack.pop()
                stack.append(int(a > b)); t(f"{idx}: greater_than({a},{b}) → {int(a>b)}")

            elif op == "less_than":
                b = stack.pop(); a = stack.pop()
                stack.append(int(a < b)); t(f"{idx}: less_than({a},{b}) → {int(a<b)}")

            # same_attribute operations (returns set of nodes sharing attr with a unique node)
            elif op.startswith("same_"):   # same_color/shape/size/material
                nid = stack.pop(); assert isinstance(nid, int)
                key = op.replace("same_", "")
                val = self.nodes[nid][key]
                S = self._all()
                # exclude self unless CLEVR expects inclusion (typically includes same object too)
                S2 = {i for i in S if self.nodes[i][key] == val}
                stack.append(S2); t(f"{idx}: {op}({val}) → |S|={len(S2)}")

            else:
                # unsupported op; raise to notice if encountered
                raise NotImplementedError(f"Unsupported op: {op}")

        assert len(stack) == 1, f"Program ended with stack size {len(stack)}"
        return stack[0]

def _extract_value(step: Dict[str, Any]) -> str:
    """
    Pulls the literal value from a step; CLEVR variants use 'value_inputs', 'side_inputs', or 'inputs' with literals.
    """
    for key in ("value_inputs","side_inputs"):
        v = step.get(key)
        if v: return v[0]
    # sometimes encoded as 'inputs' when literal, ensure string
    inp = step.get("inputs", [])
    if inp and isinstance(inp[0], str):
        return inp[0]
    raise ValueError(f"Cannot extract literal from step: {step}")

# ------------ Evaluation ------------
def evaluate(split: str = DEFAULT_SPLIT,
             clevr_root: str | Path = DEFAULT_CLEVR_ROOT,
             graph_dir: str | Path = DEFAULT_GRAPH_DIR,
             preview: int = 3):
    clevr_root = Path(clevr_root)
    graph_dir  = Path(graph_dir)

    graphs = load_graphs(graph_dir / f"scene_graphs_{split}.jsonl")
    questions = load_questions(clevr_root, split)

    total, correct = 0, 0
    previews_left = preview

    for q in questions:
        img_id = q["image_index"]
        prog   = q["program"]
        gt     = q.get("answer")
        if img_id not in graphs:
            continue
        trace: List[str] = []
        ex = CLEVRExecutor(graphs[img_id])
        try:
            pred = ex.execute(prog, trace=trace)
            # normalize answer forms: CLEVR answers can be str/int/bool; cast ints 0/1 to str "yes/no" only when appropriate
            if isinstance(pred, bool):
                pred_norm = "yes" if pred else "no"
            else:
                pred_norm = str(pred)
            gt_norm = str(gt)
            ok = (pred_norm == gt_norm)
        except Exception as e:
            pred_norm = f"<ERROR:{e.__class__.__name__}>"
            ok = False

        total += 1
        correct += int(ok)

        # Small preview
        if previews_left > 0:
            print(f"\nQ[{q['question_index']}] img={img_id}  ok={ok}")
            print("Q:", q["question"])
            print("GT:", gt, "| PRED:", pred_norm)
            for line in trace:
                print("  ", line)
            previews_left -= 1

    acc = correct / max(total, 1)
    print(f"\nTotal: {total} | Correct: {correct} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Defaults so you can just run `python reason_clevr.py`
    split = DEFAULT_SPLIT
    evaluate(split=split,
             clevr_root=DEFAULT_CLEVR_ROOT,
             graph_dir=DEFAULT_GRAPH_DIR,
             preview=3)