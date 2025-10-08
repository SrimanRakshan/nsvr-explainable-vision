from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import json

REL_MAP = {"left": "left_of", "right": "right_of", "front": "front_of", "behind": "behind"}

class CLEVRExecutorDAG:
    """
    DAG executor for CLEVR programs.
    Produces a structured trace (per-step outputs) for explainability.
    """
    def __init__(self, graph: Dict[str, Any]):
        self.g = graph
        self.nodes = graph["nodes"]
        # rel adjacency
        self.rels: Dict[str, Dict[int, Set[int]]] = {}
        for e in graph["edges"]:
            self.rels.setdefault(e["rel"], {}).setdefault(e["src"], set()).add(e["dst"])

    # ---------- helpers ----------
    def _all(self) -> Set[int]:
        return set(range(len(self.nodes)))

    def _filter_attr(self, s: Set[int], key: str, val: str) -> Set[int]:
        return {i for i in s if self.nodes[i][key] == val}

    def _relate(self, s: Set[int], rel_word: str) -> Set[int]:
        rel = REL_MAP[rel_word]
        out: Set[int] = set()
        adj = self.rels.get(rel, {})
        for i in s:
            out |= adj.get(i, set())
        return out
    
    def _as_set(self, x):
        if isinstance(x, set):
            return x
        if isinstance(x, (list, tuple)):
            return set(x)
        if isinstance(x, int):
            return {x}
        raise TypeError(f"Cannot treat {type(x)} as a set")

    def _unique(self, s: Set[int]) -> int:
        if len(s) != 1:
            raise ValueError(f"unique() expects 1, got {len(s)}")
        return next(iter(s))

    # ---------- exec ----------
    def execute(self, program: List[Dict[str, Any]]):
        results: List[Any] = []
        trace: List[Dict[str, Any]] = []

        def get_val(step: Dict[str, Any]) -> str | None:
            for k in ("value_inputs", "side_inputs"):
                v = step.get(k)
                if v: return v[0]
            return None

        for si, step in enumerate(program):
            op = step.get("type") or step.get("function")
            inp_idx: List[int] = step.get("inputs", [])
            val = get_val(step)
            ins = [results[i] for i in inp_idx] if inp_idx else []

            if op == "scene":
                out = self._all()
                trace.append({"op": op, "inputs": [], "value": None, "out_kind": "set", "out_set": sorted(out), "out_val": None})
                results.append(out); continue

            if op.startswith("filter_"):
                key = op.replace("filter_", "")
                s = self._as_set(ins[0]) if ins else self._all()      # <-- changed
                out = self._filter_attr(set(s), key, val)
                trace.append({"op": op, "inputs": inp_idx, "value": val, "out_kind": "set", "out_set": sorted(out), "out_val": None})
                results.append(out); continue

            if op == "relate":
                s = self._as_set(ins[0])                               # <-- changed
                out = self._relate(set(s), val)
                trace.append({"op": op, "inputs": inp_idx, "value": val, "out_kind": "set", "out_set": sorted(out), "out_val": None})
                results.append(out); continue

            if op == "intersect":
                A, B = self._as_set(ins[0]), self._as_set(ins[1])      # <-- changed
                out = set(A) & set(B)
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "set", "out_set": sorted(out), "out_val": None})
                results.append(out); continue

            if op == "union":
                A, B = self._as_set(ins[0]), self._as_set(ins[1])      # <-- changed
                out = set(A) | set(B)
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "set", "out_set": sorted(out), "out_val": None})
                results.append(out); continue

            if op == "unique":
                s = self._as_set(ins[0])                               # <-- changed
                nid = self._unique(set(s))
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "node", "out_set": None, "out_val": nid})
                results.append(nid); continue

            if op == "count":
                s = self._as_set(ins[0])                               # <-- changed
                c = int(len(s))
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "int", "out_set": None, "out_val": c})
                results.append(c); continue

            if op == "exist":
                s = self._as_set(ins[0])                               # <-- changed
                b = bool(len(s) > 0)
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "bool", "out_set": None, "out_val": b})
                results.append(b); continue

            if op.startswith("query_"):
                key = op.replace("query_", "")
                nid_in = ins[0]
                nid = nid_in if isinstance(nid_in, int) else self._unique(self._as_set(nid_in))  # <-- changed
                sval = self.nodes[nid][key]
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "str", "out_set": None, "out_val": sval})
                results.append(sval); continue

            if op in ("equal_integer", "greater_than", "less_than"):
                a, b = ins
                if op == "equal_integer": v = int(a == b)
                elif op == "greater_than": v = int(a > b)
                else: v = int(a < b)
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "int", "out_set": None, "out_val": v})
                results.append(v); continue

            if op.startswith("same_"):
                key = op.replace("same_", "")
                nid_in = ins[0]
                nid = nid_in if isinstance(nid_in, int) else self._unique(self._as_set(nid_in))  # <-- changed
                val_str = self.nodes[nid][key]
                out = {i for i in self._all() if self.nodes[i][key] == val_str}
                trace.append({"op": op, "inputs": inp_idx, "value": val_str, "out_kind": "set", "out_set": sorted(out), "out_val": None})
                results.append(out); continue

            if op.startswith("equal_"):
                key = op.replace("equal_", "")
                a, b = ins
                v = int(str(a) == str(b))
                trace.append({"op": op, "inputs": inp_idx, "value": None, "out_kind": "int", "out_set": None, "out_val": v})
                results.append(v); continue

            raise NotImplementedError(f"Unsupported op: {op}")

        assert len(results) == len(program)
        return results[-1], trace
