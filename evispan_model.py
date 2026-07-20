"""Inference utilities for the tuned EviSpan-PR Transformer + CRF model.

The architecture mirrors the model defined in EviSpan_PR_crf_tuned_final.ipynb.
It is intentionally inference-focused: training losses are not required by the
Streamlit application, but all parameter and buffer names remain compatible
with the exported state_dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoConfig, AutoModel, AutoTokenizer

LABELS = ["Problem", "Suggestion", "Neutral", "Appreciation"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

NER_TAGS = ["O"]
for _label in LABELS:
    NER_TAGS.extend(
        [f"B-{_label}", f"I-{_label}", f"L-{_label}", f"U-{_label}"]
    )
TAG2ID = {tag: i for i, tag in enumerate(NER_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}


def split_tag(tag: str) -> Tuple[str, Optional[str]]:
    if tag == "O" or "-" not in tag:
        return tag, None
    return tuple(tag.split("-", 1))  # type: ignore[return-value]


def build_biluo_transition_masks() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return legal BILUO start, end, and transition masks."""
    num_tags = len(TAG2ID)
    allowed_start = torch.zeros(num_tags, dtype=torch.bool)
    allowed_end = torch.zeros(num_tags, dtype=torch.bool)
    allowed_transitions = torch.zeros(num_tags, num_tags, dtype=torch.bool)

    for i in range(num_tags):
        prefix_i, _ = split_tag(ID2TAG[i])
        if prefix_i in {"O", "B", "U"}:
            allowed_start[i] = True
        if prefix_i in {"O", "L", "U"}:
            allowed_end[i] = True

    for i in range(num_tags):
        prefix_i, label_i = split_tag(ID2TAG[i])
        for j in range(num_tags):
            prefix_j, label_j = split_tag(ID2TAG[j])
            allowed = False
            if prefix_i in {"O", "L", "U"}:
                allowed = prefix_j in {"O", "B", "U"}
            elif prefix_i in {"B", "I"}:
                allowed = label_i == label_j and prefix_j in {"I", "L"}
            allowed_transitions[i, j] = allowed

    return allowed_start, allowed_end, allowed_transitions


class EviSpanPR(nn.Module):
    """Deployment-compatible EviSpan-PR architecture."""

    def __init__(
        self,
        encoder_config_dir: str | Path,
        num_labels_cls: int = len(LABELS),
        num_ner_tags: int = len(NER_TAGS),
        dropout_prob: float = 0.10,
        use_crf: bool = True,
        crf_enforce_biluo_constraints: bool = True,
        auxiliary_token_ce_loss_weight: float = 0.20,
        evidence_pooling: str = "noisy_or",
        logsumexp_temperature: float = 0.10,
        consistency_type: str = "positive_only_detach_evidence",
        consistency_confidence_threshold: float = 0.60,
    ) -> None:
        super().__init__()

        encoder_config = AutoConfig.from_pretrained(
            str(encoder_config_dir), local_files_only=True
        )
        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = int(self.encoder.config.hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.cls_head = nn.Linear(hidden_size, num_labels_cls)
        self.ner_head = nn.Linear(hidden_size, num_ner_tags)

        self.use_crf = bool(use_crf)
        self.crf_enforce_biluo_constraints = bool(crf_enforce_biluo_constraints)
        self.auxiliary_token_ce_loss_weight = float(auxiliary_token_ce_loss_weight)
        self.evidence_pooling = str(evidence_pooling)
        self.logsumexp_temperature = float(logsumexp_temperature)
        self.consistency_type = str(consistency_type)
        self.consistency_confidence_threshold = float(
            consistency_confidence_threshold
        )

        if self.use_crf:
            self.crf = CRF(num_tags=num_ner_tags, batch_first=True)
            allowed_start, allowed_end, allowed_transitions = (
                build_biluo_transition_masks()
            )
            self.register_buffer("crf_allowed_start", allowed_start)
            self.register_buffer("crf_allowed_end", allowed_end)
            self.register_buffer("crf_allowed_transitions", allowed_transitions)
            if self.crf_enforce_biluo_constraints:
                self.apply_crf_constraints()

        # These buffers are overwritten by the exported state_dict. They must
        # exist because the training notebook registers them in the model.
        self.register_buffer(
            "cls_pos_weight", torch.ones(num_labels_cls, dtype=torch.float)
        )
        self.register_buffer(
            "ner_class_weights", torch.ones(num_ner_tags, dtype=torch.float)
        )

    @torch.no_grad()
    def apply_crf_constraints(self) -> None:
        if not self.use_crf:
            return
        neg = -10000.0
        self.crf.start_transitions.data.masked_fill_(
            ~self.crf_allowed_start.to(self.crf.start_transitions.device), neg
        )
        self.crf.end_transitions.data.masked_fill_(
            ~self.crf_allowed_end.to(self.crf.end_transitions.device), neg
        )
        self.crf.transitions.data.masked_fill_(
            ~self.crf_allowed_transitions.to(self.crf.transitions.device), neg
        )

    def decode_ner(
        self, ner_logits: torch.Tensor, token_mask: torch.Tensor
    ) -> List[List[int]]:
        if self.use_crf:
            return self.crf.decode(ner_logits, mask=token_mask.bool())
        pred_ids = torch.argmax(ner_logits, dim=-1)
        return [
            row_pred[row_mask.bool()].detach().cpu().tolist()
            for row_pred, row_mask in zip(pred_ids, token_mask)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        cls_logits = self.cls_head(self.dropout(pooled_output))
        ner_logits = self.ner_head(self.dropout(sequence_output))
        return {"cls_logits": cls_logits, "ner_logits": ner_logits}


def repair_biluo_sequence(tags: List[str]) -> List[str]:
    repaired: List[str] = []
    n = len(tags)
    i = 0
    while i < n:
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            repaired.append("O")
            i += 1
            continue

        prefix, label = tag.split("-", 1)
        if label not in LABEL2ID:
            repaired.append("O")
            i += 1
            continue
        if prefix == "U":
            repaired.append(f"U-{label}")
            i += 1
            continue
        if prefix == "L":
            repaired.append(f"U-{label}")
            i += 1
            continue

        if prefix in {"B", "I"}:
            start = i
            j = i + 1
            while j < n:
                next_tag = tags[j]
                if "-" not in next_tag:
                    break
                next_prefix, next_label = next_tag.split("-", 1)
                if next_label != label:
                    break
                if next_prefix == "I":
                    j += 1
                    continue
                if next_prefix == "L":
                    length = j - start + 1
                    if length == 1:
                        repaired.append(f"U-{label}")
                    else:
                        repaired.append(f"B-{label}")
                        repaired.extend([f"I-{label}"] * (length - 2))
                        repaired.append(f"L-{label}")
                    i = j + 1
                    break
                break
            if i > start:
                continue
            length = j - start
            if length <= 1:
                repaired.append(f"U-{label}")
            else:
                repaired.append(f"B-{label}")
                repaired.extend([f"I-{label}"] * (length - 2))
                repaired.append(f"L-{label}")
            i = j
            continue

        repaired.append("O")
        i += 1
    return repaired


def repair_biluo_ids(
    tag_ids: List[int], valid_token_mask: List[bool]
) -> List[int]:
    repaired_ids = list(tag_ids)
    valid_positions = [i for i, valid in enumerate(valid_token_mask) if valid]
    valid_tags = [ID2TAG[int(tag_ids[i])] for i in valid_positions]
    repaired_tags = repair_biluo_sequence(valid_tags)
    for pos, tag in zip(valid_positions, repaired_tags):
        repaired_ids[pos] = TAG2ID.get(tag, TAG2ID["O"])
    return repaired_ids


def decoded_sequence_to_full_ids(
    decoded: List[int],
    attention_mask: List[int],
    valid_text_mask: List[bool],
) -> List[int]:
    """Restore CRF-decoded IDs to tokenizer sequence length."""
    full_ids: List[int] = []
    k = 0
    for attended, valid_text in zip(attention_mask, valid_text_mask):
        if attended:
            pred_id = int(decoded[k]) if k < len(decoded) else TAG2ID["O"]
            k += 1
        else:
            pred_id = TAG2ID["O"]
        full_ids.append(pred_id if valid_text else -100)
    return full_ids


def decode_biluo_to_spans(
    text: str,
    offsets: List[Tuple[int, int]],
    tag_ids: List[int],
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    def close_span() -> None:
        nonlocal current_label, current_start, current_end
        if (
            current_label is not None
            and current_start is not None
            and current_end is not None
            and current_start < current_end
        ):
            spans.append(
                {
                    "start": int(current_start),
                    "end": int(current_end),
                    "label": current_label,
                    "text": text[int(current_start) : int(current_end)],
                }
            )
        current_label = None
        current_start = None
        current_end = None

    for i, tag_id in enumerate(tag_ids):
        if int(tag_id) == -100:
            continue
        tok_start, tok_end = map(int, offsets[i])
        if tok_start == tok_end:
            continue
        tag = ID2TAG[int(tag_id)]
        if tag == "O" or "-" not in tag:
            close_span()
            continue

        prefix, label = tag.split("-", 1)
        if label not in LABEL2ID:
            close_span()
            continue
        if prefix == "U":
            close_span()
            spans.append(
                {
                    "start": tok_start,
                    "end": tok_end,
                    "label": label,
                    "text": text[tok_start:tok_end],
                }
            )
        elif prefix == "B":
            close_span()
            current_label = label
            current_start = tok_start
            current_end = tok_end
        elif prefix == "I":
            if current_label == label:
                current_end = tok_end
            else:
                close_span()
                current_label = label
                current_start = tok_start
                current_end = tok_end
        elif prefix == "L":
            if current_label == label:
                current_end = tok_end
                close_span()
            else:
                close_span()
                spans.append(
                    {
                        "start": tok_start,
                        "end": tok_end,
                        "label": label,
                        "text": text[tok_start:tok_end],
                    }
                )
        else:
            close_span()
    close_span()
    return spans


def _safe_torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Older PyTorch
        return torch.load(path, map_location=device)


def load_artifacts(
    artifact_dir: str | Path, device: Optional[torch.device] = None
) -> Tuple[EviSpanPR, Any, np.ndarray, Dict[str, Any], torch.device]:
    """Load model, tokenizer, thresholds, and export configuration."""
    artifact_path = Path(artifact_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with (artifact_path / "artifact_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (artifact_path / "thresholds.json").open("r", encoding="utf-8") as f:
        threshold_map = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        str(artifact_path / config.get("tokenizer_dir", "tokenizer")),
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("EviSpan-PR requires a fast tokenizer for offset mapping.")

    model = EviSpanPR(
        encoder_config_dir=artifact_path
        / config.get("encoder_config_dir", "encoder_config"),
        num_labels_cls=len(config.get("labels", LABELS)),
        num_ner_tags=len(config.get("ner_tags", NER_TAGS)),
        dropout_prob=float(config.get("dropout_prob", 0.10)),
        use_crf=bool(config.get("use_crf", True)),
        crf_enforce_biluo_constraints=bool(
            config.get("crf_enforce_biluo_constraints", True)
        ),
        auxiliary_token_ce_loss_weight=float(
            config.get("auxiliary_token_ce_loss_weight", 0.20)
        ),
        evidence_pooling=str(config.get("evidence_pooling", "noisy_or")),
        logsumexp_temperature=float(config.get("logsumexp_temperature", 0.10)),
        consistency_type=str(
            config.get("consistency_type", "positive_only_detach_evidence")
        ),
        consistency_confidence_threshold=float(
            config.get("consistency_confidence_threshold", 0.60)
        ),
    )

    checkpoint = _safe_torch_load(artifact_path / "model_state.pt", device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    if model.use_crf and model.crf_enforce_biluo_constraints:
        model.apply_crf_constraints()

    thresholds = np.asarray(
        [float(threshold_map.get(label, 0.5)) for label in LABELS],
        dtype=np.float32,
    )
    return model, tokenizer, thresholds, config, device


@torch.inference_mode()
def predict_text(
    text: str,
    model: EviSpanPR,
    tokenizer: Any,
    thresholds: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Run document multi-label and evidence-span inference on one comment."""
    text = str(text).strip()
    if not text:
        raise ValueError("Comment text must not be empty.")

    max_length = int(config.get("max_length", 256))
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = [tuple(map(int, x)) for x in encoding["offset_mapping"]]
    sequence_ids = (
        encoding.sequence_ids()
        if hasattr(encoding, "sequence_ids")
        else [None] * len(offsets)
    )
    valid_text_mask = [
        seq_id is not None and offset != (0, 0)
        for offset, seq_id in zip(offsets, sequence_ids)
    ]

    batch: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long).to(
            device
        ),
        "attention_mask": torch.tensor(
            [encoding["attention_mask"]], dtype=torch.long
        ).to(device),
    }
    if "token_type_ids" in encoding:
        batch["token_type_ids"] = torch.tensor(
            [encoding["token_type_ids"]], dtype=torch.long
        ).to(device)

    outputs = model(**batch)
    cls_probs = torch.sigmoid(outputs["cls_logits"])[0].detach().cpu().numpy()
    decoded = model.decode_ner(
        outputs["ner_logits"], token_mask=batch["attention_mask"].bool()
    )[0]
    full_ids = decoded_sequence_to_full_ids(
        decoded=decoded,
        attention_mask=list(map(int, encoding["attention_mask"])),
        valid_text_mask=valid_text_mask,
    )
    if bool(config.get("use_biluo_constraint", True)):
        full_ids = repair_biluo_ids(full_ids, valid_text_mask)

    predicted_labels = [
        label for i, label in enumerate(LABELS) if cls_probs[i] >= thresholds[i]
    ]
    predicted_spans = decode_biluo_to_spans(text, offsets, full_ids)

    return {
        "text": text,
        "predicted_labels": predicted_labels,
        "label_probabilities": {
            label: float(cls_probs[i]) for i, label in enumerate(LABELS)
        },
        "predicted_spans": predicted_spans,
        "truncated": bool(
            len(tokenizer(text, add_special_tokens=True)["input_ids"]) > max_length
        ),
    }
