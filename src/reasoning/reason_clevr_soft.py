#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, Any, List, Set, Tuple

# T-norms / T-conorms
def t_norm(a, b):    # Gödel (min)
    return min(a, b)
def s_norm(a, b):    # Gödel dual (max)
    return max(a, b)

REL_MAP = {"left":"left_of","right":"right_of","front":"front_of","behind":"behind"}

class SoftExecutor:
    """
    Fuzzy (soft) CLEVR executor.
    Represent sets as {node_id: membership in [0,1]}.
    Uses node attribute probabilities if available; falls back to crisp labels otherwise.
    Returns (answer, confidence, trace)
    """
    def __init__(self, graph: Dict[str, Any]):
        self.g = graph
        self.nodes = graph["nodes"]
        # adjacency per relation
        self.rels: Dict[str, Dict[int, Set[int]]] = {}
        for e in graph["edges"]:
            self.rels.setdefault(e["rel"], {}).setdefault(e["src"], set()).add(e["dst"])

    # ----- fuzzy set helpers -----
    def _all(self) -> Dict[int,float]:
        return {i: 1.0 for i in range(len(self.nodes))}

    def _to_fset(self, x) -> Dict[int,float]:
        # NEW: accept (nid, conf) tuples from 'unique'
        if isinstance(x, dict):
            return x
        if isinstance(x, set):
            return {i: 1.0 for i in x}
        if isinstance(x, list):
            return {i: 1.0 for i in x}
        if isinstance(x, int):
            return {x: 1.0}
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], int):
            return {x[0]: float(x[1])}
        raise TypeError(f"Cannot coerce {type(x)} to fuzzy set")

    def _filter_attr(self, fset: Dict[int,float], key: str, val: str) -> Dict[int,float]:
        out: Dict[int,float] = {}
        for i, mu in fset.items():
            node = self.nodes[i]
            if node.get("probs") and key in node["probs"]:
                p = float(node["probs"][key].get(val, 0.0))
                mu_new = t_norm(mu, p)
            else:
                mu_new = t_norm(mu, 1.0 if node[key] == val else 0.0)
            if mu_new > 0.0:
                out[i] = mu_new
        return out

    def _relate(self, fset: Dict[int,float], rel_word: str) -> Dict[int,float]:
        rel = REL_MAP[rel_word]
        adj = self.rels.get(rel, {})
        out: Dict[int,float] = {}
        # μ_out(j) = max_i μ_in(i) * R(i,j); here R is 1/0
        for i, mu_i in fset.items():
            for j in adj.get(i, set()):
                out[j] = max(out.get(j, 0.0), mu_i)  # product with 1
        return out

    def _intersect(self, A: Dict[int,float], B: Dict[int,float]) -> Dict[int,float]:
        keys = set(A) | set(B)
        return {k: t_norm(A.get(k,0.0), B.get(k,0.0)) for k in keys if t_norm(A.get(k,0.0), B.get(k,0.0)) > 0.0}

    def _union(self, A: Dict[int,float], B: Dict[int,float]) -> Dict[int,float]:
        keys = set(A) | set(B)
        return {k: s_norm(A.get(k,0.0), B.get(k,0.0)) for k in keys if s_norm(A.get(k,0.0), B.get(k,0.0)) > 0.0}

    def _argmax(self, fset):
        if not fset:
            return -1, 0.0   # sentinel: no object, zero confidence
        nid = max(fset, key=lambda k: fset[k])
        return nid, float(fset[nid])


    # ----- execution -----
    def execute(self, program: List[Dict[str,Any]]) -> Tuple[Any, float, List[Dict[str,Any]]]:
        res: List[Any] = []
        trace: List[Dict[str,Any]] = []

        def lit(step):
            for k in ("value_inputs","side_inputs"):
                v = step.get(k)
                if v: return v[0]
            return None

        for step in program:
            op = step.get("type") or step.get("function")
            idx = step.get("inputs", [])
            X = [res[i] for i in idx] if idx else []
            val = lit(step)

            if op == "scene":
                out = self._all()
                res.append(out); trace.append({"op":op,"kind":"set","size":len(out)}); continue

            if op.startswith("filter_"):
                key = op.replace("filter_","")
                S = self._to_fset(X[0]) if X else self._all()
                out = self._filter_attr(S, key, val)
                res.append(out); trace.append({"op":op,"value":val,"kind":"set","size":len(out)}); continue

            if op == "relate":
                S = self._to_fset(X[0])
                out = self._relate(S, val)
                res.append(out); trace.append({"op":op,"value":val,"kind":"set","size":len(out)}); continue

            if op == "intersect":
                out = self._intersect(self._to_fset(X[0]), self._to_fset(X[1]))
                res.append(out); trace.append({"op":op,"kind":"set","size":len(out)}); continue

            if op == "union":
                out = self._union(self._to_fset(X[0]), self._to_fset(X[1]))
                res.append(out); trace.append({"op":op,"kind":"set","size":len(out)}); continue

            if op == "unique":
                S = self._to_fset(X[0])
                nid, conf = self._argmax(S)                 # now safe on empty
                res.append((nid, conf))
                trace.append({"op": op, "kind": "node", "nid": nid, "conf": conf, "empty": (nid == -1)})
                continue

            if op == "count":
                S = self._to_fset(X[0])
                exp_count = sum(S.values())
                res.append(exp_count); trace.append({"op":op,"kind":"float","val":exp_count}); continue

            if op == "exist":
                S = self._to_fset(X[0])
                truth = max(S.values()) if S else 0.0
                res.append(truth); trace.append({"op":op,"kind":"truth","val":truth}); continue

            if op.startswith("query_"):
                key = op.replace("query_", "")
                arg = X[0]

                if isinstance(arg, tuple):                   # (nid, conf) from unique
                    nid, conf = arg
                    if nid == -1:                            # came from empty set
                        ans, ans_conf = "<none>", 0.0
                    else:
                        node = self.nodes[nid]
                        if node.get("probs") and key in node["probs"]:
                            dist = node["probs"][key]
                        else:
                            dist = {node[key]: 1.0}
                        ans = max(dist, key=lambda k: dist[k])
                        ans_conf = float(dist[ans]) * float(conf)
                else:
                    S = self._to_fset(arg)
                    scores = {}
                    for i, mu in S.items():
                        node = self.nodes[i]
                        if node.get("probs") and key in node["probs"]:
                            for lab, p in node["probs"][key].items():
                                scores[lab] = scores.get(lab, 0.0) + mu * float(p)
                        else:
                            lab = node[key]
                            scores[lab] = scores.get(lab, 0.0) + mu
                    if not scores:
                        ans, ans_conf = "<none>", 0.0
                    else:
                        ans = max(scores, key=lambda k: scores[k])
                        tot = sum(scores.values()) or 1.0
                        ans_conf = float(scores.get(ans, 0.0)) / float(tot)

                res.append((ans, ans_conf))
                trace.append({"op": op, "kind": "attr", "key": key, "ans": ans, "conf": ans_conf})
                continue


            if op in ("equal_integer","greater_than","less_than"):
                a, b = X
                if op == "equal_integer": v = 1.0 if int(a)==int(b) else 0.0
                elif op == "greater_than": v = 1.0 if float(a) > float(b) else 0.0
                else: v = 1.0 if float(a) < float(b) else 0.0
                res.append(v); trace.append({"op":op,"kind":"truth","val":v}); continue

            if op.startswith("same_"):
                key = op.replace("same_","")
                ref = X[0]
                nid = ref[0] if isinstance(ref, tuple) else int(ref)
                val_ref = self.nodes[nid][key]
                S = {i:1.0 for i in range(len(self.nodes)) if self.nodes[i][key]==val_ref}
                res.append(S); trace.append({"op":op,"kind":"set","size":len(S),"val":val_ref}); continue

            if op.startswith("equal_"):  # equal_color/shape/size/material
                a, b = X
                v = 1.0 if str(a)==str(b) else 0.0
                res.append(v); trace.append({"op":op,"kind":"truth","val":v}); continue

            raise NotImplementedError(f"Unsupported op: {op}")

        # confidence for final answer
        if isinstance(res[-1], tuple):     # (label, conf)
            return res[-1], res[-1][1], trace
        if isinstance(res[-1], float):     # truth in [0,1] or expected count
            return res[-1], res[-1], trace
        return res[-1], 1.0, trace
