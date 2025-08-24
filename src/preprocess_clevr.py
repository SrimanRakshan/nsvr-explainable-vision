import json
import os
from pathlib import Path

DATA_DIR = Path("/home/Zoro/Documents/data/raw/clevr/CLEVR_v1.0/")
OUTPUT_DATA_DIR = Path("/home/Zoro/Documents/data/processed")

def load_scenes(split="train"):
    path = DATA_DIR / "scenes" / f"CLEVR_{split}_scenes.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["scenes"]

def load_questions(split="train"):
    path = DATA_DIR / "questions" / f"CLEVR_{split}_questions.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["questions"]

def preprocess_scenes(split="train"):
    scenes = load_scenes(split)
    processed = []
    for scene in scenes:
        image_id = scene["image_index"]
        objects = []
        for obj in scene["objects"]:
            objects.append({
                "color": obj["color"],
                "shape": obj["shape"],
                "size": obj["size"],
                "material": obj["material"],
                "position": obj["3d_coords"]
            })
        processed.append({
            "image_id": image_id,
            "objects": objects
        })
    return processed

def preprocess_questions(split="train"):
    questions = load_questions(split)
    processed = []
    for q in questions:
        processed.append({
            "image_id": q["image_index"],
            "question": q["question"],
            "program": q["program"],   # functional program
            "answer": q["answer"]
        })
    return processed

def save_jsonl(data, out_path):
    with open(out_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    for split in ["train", "val"]:
        scenes = preprocess_scenes(split)
        questions = preprocess_questions(split)

        save_jsonl(scenes, OUTPUT_DATA_DIR / f"processed_{split}_scenes.jsonl")
        save_jsonl(questions, OUTPUT_DATA_DIR / f"processed_{split}_questions.jsonl")

    print("✅ Preprocessing done! Check processed JSONL files.")
