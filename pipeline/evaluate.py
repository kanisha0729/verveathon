"""Evaluates the system against the labeled contrastive dataset (data/labels.csv).

1. Mean overall score per severity level (should fall as severity rises).
2. Temporal grounding: for each labeled flaw, the IoU with the best-matching detected region of the
   same type; recall / precision at IoU >= 0.3 and mean IoU per flaw type.

Run:  python pipeline/evaluate.py      -> prints a table and writes data/eval.json
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.analyze import analyze  # noqa: E402

IOU_HIT = 0.3


def iou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main():
    with open(ROOT / "data/labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    scores_by_sev = defaultdict(list)
    per_type = defaultdict(lambda: {"labels": 0, "hits": 0, "ious": [], "detections": 0, "true_detections": 0})
    files = []
    for fname, labels in by_file.items():
        path = ROOT / fname
        if not path.exists():
            continue
        print("Analysing", fname)
        res = analyze(path)
        sev = int(labels[0]["severity"])
        scores_by_sev[sev].append(res["overall"])
        real = [l for l in labels if l["flaw_type"] != "none"]
        types_here = {l["flaw_type"] for l in real}
        file_ious = []
        for l in real:
            s, e = float(l["start"]), float(l["end"])
            cands = [r for r in res["regions"] if r["flaw_type"] == l["flaw_type"]]
            best = max([iou(s, e, r["start"], r["end"]) for r in cands], default=0.0)
            st = per_type[l["flaw_type"]]
            st["labels"] += 1
            st["ious"].append(best)
            st["hits"] += best >= IOU_HIT
            file_ious.append(round(best, 2))
        for r in res["regions"]:
            if r["flaw_type"] in types_here:
                st = per_type[r["flaw_type"]]
                st["detections"] += 1
                st["true_detections"] += any(
                    l["flaw_type"] == r["flaw_type"] and iou(float(l["start"]), float(l["end"]), r["start"], r["end"]) >= IOU_HIT
                    for l in real)
        files.append({"file": fname, "severity": sev, "overall": res["overall"],
                      "regions": len(res["regions"]), "label_ious": file_ious})

    score_table = {str(s): round(sum(v) / len(v), 1) for s, v in sorted(scores_by_sev.items())}
    type_table = {}
    for t, st in sorted(per_type.items()):
        type_table[t] = {
            "labels": st["labels"],
            "recall": round(st["hits"] / st["labels"], 2) if st["labels"] else None,
            "precision": round(st["true_detections"] / st["detections"], 2) if st["detections"] else None,
            "mean_iou": round(sum(st["ious"]) / len(st["ious"]), 2) if st["ious"] else None,
        }
    out = {"score_by_severity": score_table, "per_flaw_type": type_table, "files": files, "iou_threshold": IOU_HIT}
    (ROOT / "data/eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\nMean overall score by severity:")
    for s, v in score_table.items():
        print(f"  severity {s}: {v}")
    print(f"\nPer flaw type (hit = IoU >= {IOU_HIT}):")
    print(f"  {'type':<12}{'labels':>7}{'recall':>8}{'precision':>11}{'mean IoU':>10}")
    for t, v in type_table.items():
        print(f"  {t:<12}{v['labels']:>7}{str(v['recall']):>8}{str(v['precision']):>11}{str(v['mean_iou']):>10}")
    print("\nSaved data/eval.json")


if __name__ == "__main__":
    main()
