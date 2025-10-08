from __future__ import annotations
import os, sys, json
from pathlib import Path

# ---------- project paths ----------
APP_DIR   = Path(__file__).resolve().parent
PROJ_ROOT = APP_DIR.parent
SRC_DIR   = PROJ_ROOT / "src"
sys.path.append(str(SRC_DIR))

import streamlit as st

# ---------- your data paths (edit if needed) ----------
CLEVR_ROOT = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0")
GRAPH_DIR  = Path(r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed")
GT_GRAPHS  = GRAPH_DIR / "scene_graphs_val.jsonl"
PRED_GRAPHS= GRAPH_DIR / "pred_scene_graphs_val.jsonl"
QUEST_PATH = CLEVR_ROOT / "questions" / "CLEVR_val_questions.json"
SPLIT      = "val"

# ---------- imports from src ----------
from reasoning.reason_clevr_dag import CLEVRExecutorDAG
from reasoning.reason_clevr_soft import SoftExecutor
from reasoning.explainer import trace_to_explanation

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def load_graphs(path: Path):
    d = {}
    if not path.exists():
        return d
    with open(path, "r") as f:
        for line in f:
            g = json.loads(line)
            d[g["image_id"]] = g
    return d

@st.cache_data(show_spinner=False)
def load_questions(path: Path):
    with open(path, "r") as f:
        return json.load(f)["questions"]

def norm_ans(x):
    if isinstance(x, bool):
        return "yes" if x else "no"
    return str(x)

def run_hard(graph, prog):
    ex = CLEVRExecutorDAG(graph)
    ans, trace = ex.execute(prog)
    return norm_ans(ans), trace, 1.0

def run_soft(graph, prog):
    ex = SoftExecutor(graph)
    out, conf, trace = ex.execute(prog)
    if isinstance(out, tuple):   # (label, conf)
        ans = out[0]
    elif isinstance(out, (int, float)):
        ans = out
    else:
        ans = out
    return norm_ans(ans), trace, float(conf)

def img_path_from_graph(graph):
    img_file = graph.get("image_filename")
    return CLEVR_ROOT / "images" / SPLIT / img_file

def app():
    st.set_page_config(page_title="NSVR — CLEVR", layout="wide")
    st.title("Neuro-Symbolic Visual Reasoning (CLEVR) — Demo")

    with st.sidebar:
        st.subheader("Paths")
        st.code(f"CLEVR_ROOT = {CLEVR_ROOT}")
        st.code(f"GRAPH_DIR  = {GRAPH_DIR}")
        src_choice = st.radio("Graph source", ["Predicted (from CNN)", "Ground-truth (GT)"], index=0)
        graphs_path = PRED_GRAPHS if src_choice.startswith("Predicted") else GT_GRAPHS
        st.write("Using graphs:", graphs_path.name)

    graphs = load_graphs(graphs_path) or {}
    qs = load_questions(QUEST_PATH)

    if not graphs:
        st.error(f"No graphs found at {graphs_path}. Run your pipeline to generate them.")
        return

    # group questions by image id
    from collections import defaultdict
    by_img = defaultdict(list)
    for q in qs:
        by_img[q["image_index"]].append(q)

    all_img_ids = sorted(list(graphs.keys()))
    col1, col2 = st.columns([1,2], vertical_alignment="center")
    with col1:
        img_id = st.selectbox("Image ID", all_img_ids, index=0)
    with col2:
        exec_mode = st.radio("Executor", ["Hard (DAG)", "Soft (Fuzzy)"], horizontal=True)

    g = graphs[img_id]
    img_file = img_path_from_graph(g)
    if img_file.exists():
        st.image(str(img_file), caption=g.get("image_filename",""), use_container_width=True)  # fixed deprecated arg
    else:
        st.warning(f"Image file not found: {img_file}")

    q_list = by_img.get(img_id, [])
    if not q_list:
        st.info("No questions found for this image ID in the selected split.")
        return

    q_idx = st.selectbox(
        "Question",
        roptions=list(range(len(q_list))),
        format_func=lambda i: q_list[i]["question"],
        index=0
    )
    q = q_list[q_idx]

    run = st.button("Run reasoning")
    if run:
        if exec_mode.startswith("Hard"):
            pred, trace, conf = run_hard(g, q["program"])
        else:
            pred, trace, conf = run_soft(g, q["program"])

        exp = trace_to_explanation(trace, g, pred, q["question"])

        a1, a2 = st.columns([2,1])
        with a1:
            st.markdown(f"### Answer: `{pred}`")
            if exec_mode.startswith("Soft"):
                st.markdown(f"**Confidence:** {conf:.3f}")
            st.markdown("### Explanation")
            st.write(exp)

        with a2:
            st.markdown("### Meta")
            st.write(f"**Image ID:** {img_id}")
            st.write(f"**Graph source:** {'pred' if src_choice.startswith('Predicted') else 'gt'}")
            st.write(f"**Program length:** {len(q['program'])}")

        with st.expander("Show program (JSON)"):
            st.json(q["program"])

        with st.expander("Trace (per step)"):
            st.json(trace)

        with st.expander("Scene nodes (predicted attributes)"):
            import pandas as pd
            rows = []
            for n in g["nodes"]:
                row = {
                    "id": n["id"],
                    "size": n.get("size"),
                    "color": n.get("color"),
                    "material": n.get("material"),
                    "shape": n.get("shape"),
                }
                probs = n.get("probs")
                if probs and isinstance(probs, dict):
                    for head in ("size","color","material","shape"):
                        if head in probs and isinstance(probs[head], dict) and len(probs[head]) > 0:
                            lab = max(probs[head], key=lambda k: probs[head][k])
                            row[f"{head}_p@1"] = f"{lab} ({probs[head][lab]:.2f})"
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

    st.caption("Tip: switch Predicted ↔ GT graphs, and Hard ↔ Soft executors to see differences.")

def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

# If Streamlit launched this script, run the app (no CLI calls here).
if _running_in_streamlit():
    app()

# If executed directly, show instructions and exit cleanly (prevents double-runtime).
if __name__ == "__main__":
    print(
        "\nThis UI must be launched by Streamlit.\n"
        "Run one of these:\n"
        "  streamlit run app/main_app.py\n"
        "  python run_app.py\n"
    )
    raise SystemExit(0)
