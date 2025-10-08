from __future__ import annotations
from typing import Dict, Any, List

def summarize_object(n: Dict[str, Any]) -> str:
    # e.g., "large red metal cube"
    return f"{n['size']} {n['color']} {n['material']} {n['shape']}"

def trace_to_explanation(trace: List[Dict[str, Any]], graph: Dict[str, Any], answer: Any, question: str) -> str:
    """
    Build a concise NL explanation from the structured trace.
    Heuristics:
      - if final op is query_*, refer to the unique object found earlier
      - if final is count/exist, mention the set cardinality
      - include 1–2 key filters and any relation
    """
    ops = [t["op"] for t in trace]
    nodes = graph["nodes"]

    # find last set before a unique/query step for context
    ctx = None
    for t in reversed(trace):
        if t["out_kind"] == "set":
            ctx = t
            break

    parts = [f"Q: {question}"]
    if ops and ops[-1].startswith("query_"):
        key = ops[-1].replace("query_", "")
        # find the node index produced just before query
        nid = None
        # walk back to find a 'node' output
        for t in reversed(trace):
            if t["out_kind"] == "node":
                nid = int(t["out_val"]); break
        obj_phrase = summarize_object(nodes[nid]) if nid is not None else "the referred object"
        parts.append(f"A: {answer}.")
        parts.append(f"Because the question refers to {obj_phrase}, whose {key} is '{answer}'.")
        if ctx and ctx.get("value"):
            parts.append(f"Key filters included {ctx['op'].replace('filter_', '')} = {ctx['value']}.")
    elif ops and ops[-1] == "count":
        k = len(ctx["out_set"]) if ctx else answer
        parts.append(f"A: {answer}.")
        parts.append(f"Because there are {k} objects that satisfy the filters in the scene.")
    elif ops and ops[-1] == "exist":
        parts.append(f"A: {'yes' if answer else 'no'}.")
        parts.append("Because the filtered set was " + ("non-empty." if answer else "empty."))
    else:
        parts.append(f"A: {answer}.")
        parts.append("Derived by executing the CLEVR program over the scene graph step by step.")

    return " ".join(parts)
