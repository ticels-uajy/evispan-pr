"""Export EviSpan-PR model and test predictions for the Streamlit application.

Run this inside the notebook kernel after the single-split EviSpan-PR model has
been trained and evaluated:

    exec(open("export_streamlit_artifacts.py", encoding="utf-8").read(), globals())

Expected notebook variables/functions:
    CFG, model, tokenizer, best_thresholds, test_features, test_loader,
    predict_model, decode_biluo_to_spans, LABELS, NER_TAGS, MODEL_VARIANTS, device
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_REQUIRED_GLOBALS = [
    "CFG",
    "model",
    "tokenizer",
    "best_thresholds",
    "test_features",
    "test_loader",
    "predict_model",
    "decode_biluo_to_spans",
    "LABELS",
    "NER_TAGS",
    "MODEL_VARIANTS",
    "device",
]
missing = [name for name in _REQUIRED_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "Missing notebook objects: "
        + ", ".join(missing)
        + ". Set CFG.run_single_split=True and run the training/evaluation cells first."
    )

artifact_dir = Path(CFG.output_dir) / "streamlit_artifacts" / "evispan_pr"
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
(artifact_dir / "encoder_config").mkdir(parents=True, exist_ok=True)

# Save tokenizer and encoder configuration locally. The Streamlit loader can
# reconstruct the encoder architecture without downloading the base model.
tokenizer.save_pretrained(artifact_dir / "tokenizer")
model.encoder.config.save_pretrained(artifact_dir / "encoder_config")

# Move tensors in the exported state_dict to CPU for portable loading.
state_dict_cpu = {
    key: value.detach().cpu() if torch.is_tensor(value) else value
    for key, value in model.state_dict().items()
}
torch.save(
    {
        "model_state_dict": state_dict_cpu,
        "model_class": "EviSpanPR",
    },
    artifact_dir / "model_state.pt",
)

threshold_map = {
    label: float(best_thresholds[i]) for i, label in enumerate(LABELS)
}
with (artifact_dir / "thresholds.json").open("w", encoding="utf-8") as f:
    json.dump(threshold_map, f, indent=2, ensure_ascii=False)

artifact_config = {
    "model_name": CFG.model_name,
    "labels": list(LABELS),
    "ner_tags": list(NER_TAGS),
    "max_length": int(CFG.max_length),
    "dropout_prob": 0.10,
    "use_crf": bool(CFG.use_crf),
    "crf_enforce_biluo_constraints": bool(CFG.crf_enforce_biluo_constraints),
    "auxiliary_token_ce_loss_weight": float(CFG.auxiliary_token_ce_loss_weight),
    "evidence_pooling": str(CFG.evidence_pooling),
    "logsumexp_temperature": float(CFG.logsumexp_temperature),
    "consistency_type": str(CFG.consistency_type),
    "consistency_confidence_threshold": float(
        CFG.consistency_confidence_threshold
    ),
    "use_biluo_constraint": bool(CFG.use_biluo_constraint),
    "tokenizer_dir": "tokenizer",
    "encoder_config_dir": "encoder_config",
    "source_notebook_config": asdict(CFG),
}
with (artifact_dir / "artifact_config.json").open("w", encoding="utf-8") as f:
    json.dump(artifact_config, f, indent=2, ensure_ascii=False)

main_variant = MODEL_VARIANTS["EviSpan-PR"]
if "test_pred" not in globals():
    test_pred = predict_model(
        model,
        test_loader,
        test_features,
        device,
        cls_loss_weight=main_variant["cls_loss_weight"],
        ner_loss_weight=main_variant["ner_loss_weight"],
        consistency_loss_weight=main_variant["consistency_loss_weight"],
    )

cls_probs = np.asarray(test_pred["cls_probs"])
cls_pred = (cls_probs >= np.asarray(best_thresholds).reshape(1, -1)).astype(int)
example_indices = test_pred.get("example_indices", list(range(len(test_features))))

# Decode predicted spans and map every prediction back to its feature index.
prediction_by_feature_index = {}
for output_pos, (pred_ids, true_ids, feature_idx) in enumerate(
    zip(
        test_pred["ner_pred_ids"],
        test_pred["ner_true_ids"],
        example_indices,
    )
):
    feature = test_features[int(feature_idx)]
    predicted_spans = decode_biluo_to_spans(
        text=feature["text"],
        offsets=feature["offset_mapping"],
        tag_ids=pred_ids,
        true_tag_ids_for_mask=true_ids,
    )
    prediction_by_feature_index[int(feature_idx)] = {
        "output_pos": output_pos,
        "predicted_spans": predicted_spans,
    }

records = []
for feature_idx, feature in enumerate(test_features):
    mapped = prediction_by_feature_index[feature_idx]
    output_pos = int(mapped["output_pos"])
    true_labels = list(feature.get("doc_labels", []))
    predicted_labels = [
        label for j, label in enumerate(LABELS) if int(cls_pred[output_pos, j]) == 1
    ]
    true_spans = [
        {
            "start": int(span["start"]),
            "end": int(span["end"]),
            "label": str(span["label"]),
            "text": str(
                span.get(
                    "text",
                    feature["text"][int(span["start"]) : int(span["end"])],
                )
            ),
        }
        for span in feature.get("spans", [])
    ]
    predicted_spans = [
        {
            "start": int(span["start"]),
            "end": int(span["end"]),
            "label": str(span["label"]),
            "text": str(span.get("text", "")),
        }
        for span in mapped["predicted_spans"]
    ]

    records.append(
        {
            "id": feature.get("id", feature_idx),
            "text": feature["text"],
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
            "label_probabilities": {
                label: float(cls_probs[output_pos, j])
                for j, label in enumerate(LABELS)
            },
            "true_spans": true_spans,
            "predicted_spans": predicted_spans,
            "document_exact_match": set(true_labels) == set(predicted_labels),
        }
    )

with (artifact_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with (artifact_dir / "test_data.jsonl").open("w", encoding="utf-8") as f:
    for feature in test_features:
        row = {
            "id": feature.get("id"),
            "text": feature.get("text", ""),
            "doc_labels": feature.get("doc_labels", []),
            "spans": feature.get("spans", []),
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Streamlit artifacts exported to:", artifact_dir)
for file_path in sorted(artifact_dir.rglob("*")):
    if file_path.is_file():
        print(" -", file_path.relative_to(artifact_dir), file_path.stat().st_size, "bytes")
