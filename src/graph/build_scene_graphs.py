"""
Step 4: Build symbolic scene graphs (GROUND-TRUTH) from CLEVR scenes.
Outputs JSONL: one graph per image
  {
    "image_id": int,
    "image_filename": str,
    "nodes": [
      {"id": int, "color": str, "shape": str, "size": str, "material": str, "position": [x,y,z]}
    ],
    "edges": [
      {"src": int, "rel": "left_of|right_of|front_of|behind", "dst": int}
    ]
  }
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, List
from tqdm import tqdm

# ---- Adjust defaults to your setup ----
DEFAULT_DATA_ROOT = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\raw\clevr\CLEVR_v1.0"
DEFAULT_OUT_DIR   = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\data\processed"

def load_scenes(clevr_root: Path, split: str) -> Dict[str, Any]:
    scenes_path = clevr_root / "scenes" / f"CLEVR_{split}_scenes.json"
    with open(scenes_path, "r") as f:
        data = json.load(f)
    return data

def make_graph_for_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single CLEVR scene dict to a graph dict.
    Uses provided relationships field for left/right/front/behind.
    """
    image_id = scene["image_index"]
    image_filename = scene["image_filename"]

    # nodes
    nodes = []
    for i, obj in enumerate(scene["objects"]):
        nodes.append({
            "id": i,
            "color": obj["color"],
            "shape": obj["shape"],
            "size": obj["size"],
            "material": obj["material"],
            "position": obj["3d_coords"],  # keep 3D coords for later heuristics
        })

    # edges from relationships (indices refer to nodes)
    edges: List[Dict[str, Any]] = []
    rels = scene.get("relationships", {})
    # relationships typically contain lists of lists: for each object index, list of object indices in that relation
    # Example: rels["left"] = [[1,2], [], [0], ...]
    mapping = {
        "left":  "left_of",
        "right": "right_of",
        "front": "front_of",
        "behind":"behind",
    }
    for rel_key, rel_name in mapping.items():
        lists = rels.get(rel_key, [])
        for src_idx, dst_list in enumerate(lists):
            for dst_idx in dst_list:
                edges.append({"src": src_idx, "rel": rel_name, "dst": dst_idx})

    return {
        "image_id": image_id,
        "image_filename": image_filename,
        "nodes": nodes,
        "edges": edges,
    }

def save_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

def build_and_save(clevr_root: Path, out_dir: Path, split: str, preview: int = 3) -> Path:
    data = load_scenes(clevr_root, split)
    scenes = data["scenes"]

    graphs = []
    for scene in tqdm(scenes, desc=f"Building graphs [{split}]"):
        graphs.append(make_graph_for_scene(scene))

    out_path = out_dir / f"scene_graphs_{split}.jsonl"
    save_jsonl(graphs, out_path)

    # quick sanity
    print(f"\nSaved {len(graphs)} graphs → {out_path}")
    print("Preview:")
    for g in graphs[:preview]:
        print(f"- image_id={g['image_id']} | nodes={len(g['nodes'])} | edges={len(g['edges'])} | file={g['image_filename']}")
    return out_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Path to CLEVR_v1.0")
    parser.add_argument("--out-dir",   type=str, default=DEFAULT_OUT_DIR, help="Folder to write processed graphs")
    parser.add_argument("--splits",    type=str, nargs="+", default=["train","val"], help="Which splits to process")
    args = parser.parse_args([])  # use defaults automatically

    clevr_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    for split in args.splits:
        build_and_save(clevr_root, out_dir, split)
