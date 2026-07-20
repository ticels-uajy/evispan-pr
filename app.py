from __future__ import annotations

import html
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from evispan_model import LABELS, load_artifacts, predict_text

st.set_page_config(
    page_title="EviSpan-PR Test Explorer",
    page_icon="🧾",
    layout="wide",
)

LABEL_COLORS = {
    "Problem": "#ffd6d6",
    "Suggestion": "#d9ecff",
    "Neutral": "#eeeeee",
    "Appreciation": "#d9f7df",
}
LABEL_BORDERS = {
    "Problem": "#b42318",
    "Suggestion": "#175cd3",
    "Neutral": "#475467",
    "Appreciation": "#067647",
}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = Path(
    os.getenv("EVISPAN_ARTIFACT_DIR", str(PROJECT_DIR / "artifacts" / "evispan_pr"))
)


@st.cache_data(show_spinner=False)
def read_prediction_bytes(data: bytes, filename: str) -> List[Dict[str, Any]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(data))
        records = df.to_dict(orient="records")
    elif suffix == ".json":
        parsed = json.loads(data.decode("utf-8"))
        records = parsed if isinstance(parsed, list) else parsed.get("records", [])
    else:
        records = [
            json.loads(line)
            for line in data.decode("utf-8").splitlines()
            if line.strip()
        ]
    return [normalise_record(record, i) for i, record in enumerate(records)]


@st.cache_data(show_spinner=False)
def read_prediction_path(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    return read_prediction_bytes(file_path.read_bytes(), file_path.name)


def parse_json_like(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [x.strip() for x in value.split(",") if x.strip()]
    return default


def first_present(record: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalise_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    # Ground-truth keys supported by both the raw dataset and the exported file.
    true_labels = parse_json_like(
        first_present(
            record,
            [
                "true_labels",
                "doc_labels",
                "labels",
                "accept",
                "classification_labels",
                "label",
            ],
            [],
        ),
        [],
    )

    # Prediction keys must come from test_predictions.jsonl. They are not
    # available in the original peer-review dataset.
    prediction_keys = {
        "predicted_labels",
        "pred_labels",
        "label_probabilities",
        "pred_probs",
        "predicted_spans",
        "pred_spans",
    }
    prediction_fields_present = bool(prediction_keys.intersection(record.keys()))

    predicted_labels = parse_json_like(
        first_present(record, ["predicted_labels", "pred_labels"], []), []
    )
    probabilities = parse_json_like(
        first_present(record, ["label_probabilities", "pred_probs"], {}), {}
    )
    true_spans = parse_json_like(
        first_present(record, ["true_spans", "spans", "entities"], []), []
    )
    predicted_spans = parse_json_like(
        first_present(record, ["predicted_spans", "pred_spans"], []), []
    )

    # Support flattened probability columns from CSV exports.
    if not probabilities:
        probabilities = {
            label: float(record.get(f"prob_{label.lower()}", np.nan))
            for label in LABELS
            if pd.notna(record.get(f"prob_{label.lower()}", np.nan))
        }

    return {
        **record,
        "row_index": index,
        "id": record.get("id", record.get("sample_id", index)),
        "text": str(record.get("text", record.get("comment", ""))),
        "true_labels": [x for x in true_labels if x in LABELS],
        "predicted_labels": [x for x in predicted_labels if x in LABELS],
        "label_probabilities": {
            label: float(probabilities.get(label, np.nan)) for label in LABELS
        },
        "true_spans": normalise_spans(true_spans),
        "predicted_spans": normalise_spans(predicted_spans),
        "_prediction_fields_present": prediction_fields_present,
    }


def normalise_spans(spans: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if not isinstance(spans, list):
        return output
    for span in spans:
        if not isinstance(span, dict):
            continue
        label = span.get("label")
        if label not in LABELS:
            continue
        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except (TypeError, ValueError):
            continue
        if start >= end:
            continue
        output.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": str(span.get("text", "")),
            }
        )
    return sorted(output, key=lambda x: (x["start"], x["end"], x["label"]))


def set_equal(a: Iterable[str], b: Iterable[str]) -> bool:
    return set(a) == set(b)


def label_badges(labels: Iterable[str]) -> str:
    labels = list(labels)
    if not labels:
        return '<span class="empty-badge">None</span>'
    parts = []
    for label in labels:
        parts.append(
            f'<span class="label-badge" style="background:{LABEL_COLORS[label]};'
            f'border-color:{LABEL_BORDERS[label]};color:{LABEL_BORDERS[label]}">'
            f'{html.escape(label)}</span>'
        )
    return " ".join(parts)


def render_highlighted_text(text: str, spans: List[Dict[str, Any]]) -> str:
    """Highlight non-overlapping predicted or gold spans by character offsets."""
    safe_spans: List[Dict[str, Any]] = []
    cursor = 0
    for span in sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        start = max(0, min(int(span["start"]), len(text)))
        end = max(0, min(int(span["end"]), len(text)))
        if start < cursor or start >= end:
            continue
        safe_spans.append({**span, "start": start, "end": end})
        cursor = end

    parts: List[str] = []
    cursor = 0
    for span in safe_spans:
        start, end, label = span["start"], span["end"], span["label"]
        parts.append(html.escape(text[cursor:start]))
        parts.append(
            f'<span class="evidence" style="background:{LABEL_COLORS[label]};'
            f'border-bottom:3px solid {LABEL_BORDERS[label]};" '
            f'title="{html.escape(label)} [{start}:{end}]">'
            f'{html.escape(text[start:end])}'
            f'<small>{html.escape(label)}</small></span>'
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return '<div class="comment-box">' + "".join(parts).replace("\n", "<br>") + "</div>"


def spans_dataframe(spans: List[Dict[str, Any]]) -> pd.DataFrame:
    if not spans:
        return pd.DataFrame(columns=["Label", "Start", "End", "Evidence"])
    return pd.DataFrame(
        [
            {
                "Label": s["label"],
                "Start": s["start"],
                "End": s["end"],
                "Evidence": s.get("text", ""),
            }
            for s in spans
        ]
    )


def probability_dataframe(record: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Label": label,
                "Probability": record["label_probabilities"].get(label, np.nan),
                "Predicted": label in record["predicted_labels"],
                "Ground truth": label in record["true_labels"],
            }
            for label in LABELS
        ]
    )


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "row_index": r["row_index"],
                "id": r["id"],
                "text": r["text"],
                "true_labels": ", ".join(r["true_labels"]),
                "predicted_labels": ", ".join(r["predicted_labels"]),
                "exact_match": set_equal(r["true_labels"], r["predicted_labels"]),
                "n_predicted_spans": len(r["predicted_spans"]),
            }
        )
    return pd.DataFrame(rows)


def artifact_is_complete(path: Path) -> bool:
    required = [
        path / "artifact_config.json",
        path / "thresholds.json",
        path / "model_state.pt",
        path / "encoder_config" / "config.json",
        path / "tokenizer",
    ]
    return all(item.exists() for item in required)


def prediction_payload_is_valid(records: List[Dict[str, Any]]) -> bool:
    """Return True only for exported records that contain model outputs."""
    return bool(records) and all(
        bool(record.get("_prediction_fields_present")) for record in records
    )


@st.cache_resource(show_spinner="Loading EviSpan-PR model...")
def get_live_model(artifact_dir: str):
    return load_artifacts(artifact_dir)


st.markdown(
    """
<style>
.label-badge, .empty-badge {
    display:inline-block; padding:0.25rem 0.55rem; margin:0.08rem;
    border:1px solid #98a2b3; border-radius:999px; font-weight:650;
}
.empty-badge {background:#f2f4f7; color:#475467;}
.comment-box {
    border:1px solid #d0d5dd; border-radius:10px; padding:1rem 1.1rem;
    line-height:2.05; font-size:1.02rem; background:#ffffff;
}
.evidence {padding:0.10rem 0.18rem; border-radius:4px; position:relative;}
.evidence small {
    font-size:0.62rem; margin-left:0.25rem; padding:0.08rem 0.20rem;
    border-radius:3px; background:rgba(255,255,255,0.76); font-weight:700;
}
.section-label {font-size:0.83rem; color:#667085; font-weight:700; text-transform:uppercase;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("EviSpan-PR Test Data and Prediction Explorer")
st.caption(
    "Explore test comments, multi-label predictions, class probabilities, and CRF-based evidence spans."
)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader(
        "Upload exported test predictions",
        type=["jsonl", "json", "csv"],
        help="Expected file: test_predictions.jsonl generated by the export cell.",
    )
    default_prediction_path = DEFAULT_ARTIFACT_DIR / "test_predictions.jsonl"
    st.caption(f"Default artifact directory: `{DEFAULT_ARTIFACT_DIR}`")
    st.caption(
        "Upload the exported `test_predictions.jsonl`. The original "
        "`peer-review-masdig-final.jsonl` contains annotations only and has no model predictions."
    )

try:
    if uploaded is not None:
        records = read_prediction_bytes(uploaded.getvalue(), uploaded.name)
        source_label = uploaded.name
    elif default_prediction_path.exists():
        records = read_prediction_path(str(default_prediction_path))
        source_label = str(default_prediction_path)
    else:
        records = []
        source_label = "Not loaded"
except Exception as exc:
    st.error(f"Could not read prediction data: {exc}")
    records = []
    source_label = "Read error"

prediction_payload_ok = prediction_payload_is_valid(records)

if not records:
    st.info(
        "No test prediction file has been loaded. Run the export cell added to the notebook, "
        "then place `test_predictions.jsonl` in `artifacts/evispan_pr/`, or upload it in the sidebar."
    )
elif not prediction_payload_ok:
    st.error(
        "The uploaded file contains raw peer-review records, but no EviSpan-PR prediction fields. "
        "Upload `test_predictions.jsonl` generated by `export_streamlit_artifacts.py`, not the original dataset."
    )
    st.code(
        "Required prediction fields: predicted_labels, label_probabilities, predicted_spans",
        language="text",
    )

summary_tab, explorer_tab, inference_tab = st.tabs(
    ["Batch summary", "Test data explorer", "New comment inference"]
)

with summary_tab:
    if not records:
        st.warning("Load test predictions to view the summary.")
    elif not prediction_payload_ok:
        st.warning(
            "This is a raw dataset, so performance metrics cannot be calculated. "
            "The table below is only a ground-truth preview."
        )
        preview = pd.DataFrame(
            [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "true_labels": ", ".join(r["true_labels"]),
                    "n_true_spans": len(r["true_spans"]),
                }
                for r in records[:100]
            ]
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)
    else:
        df_all = records_to_dataframe(records)
        exact_rate = float(df_all["exact_match"].mean()) if len(df_all) else 0.0
        total_spans = int(df_all["n_predicted_spans"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test comments", f"{len(records):,}")
        c2.metric("Exact label match", f"{exact_rate:.1%}")
        c3.metric("Predicted evidence spans", f"{total_spans:,}")
        c4.metric("Source", Path(source_label).name)

        rows = []
        for label in LABELS:
            true = np.array([label in r["true_labels"] for r in records], dtype=int)
            pred = np.array(
                [label in r["predicted_labels"] for r in records], dtype=int
            )
            tp = int(((true == 1) & (pred == 1)).sum())
            fp = int(((true == 0) & (pred == 1)).sum())
            fn = int(((true == 1) & (pred == 0)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append(
                {
                    "Label": label,
                    "Gold support": int(true.sum()),
                    "Predicted": int(pred.sum()),
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )
        st.subheader("Document-level performance from exported predictions")
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precision": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1": st.column_config.NumberColumn(format="%.3f"),
            },
        )

        st.subheader("Test records")
        st.dataframe(
            df_all,
            use_container_width=True,
            hide_index=True,
            column_config={
                "text": st.column_config.TextColumn(width="large"),
                "exact_match": st.column_config.CheckboxColumn(),
            },
        )
        st.download_button(
            "Download summary CSV",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name="evispan_test_summary.csv",
            mime="text/csv",
        )

with explorer_tab:
    if not records:
        st.warning("Load test predictions to explore individual records.")
    elif not prediction_payload_ok:
        st.warning(
            "Predicted labels, probabilities, and evidence spans are unavailable because the uploaded "
            "file is the original dataset rather than an exported prediction file."
        )
    else:
        st.subheader("Filter test comments")
        f1, f2, f3 = st.columns([1, 1, 2])
        match_filter = f1.selectbox(
            "Label result", ["All", "Exact match", "Mismatch"]
        )
        label_filter = f2.selectbox("Contains label", ["All"] + LABELS)
        text_query = f3.text_input("Search comment text")

        filtered = []
        for record in records:
            is_exact = set_equal(record["true_labels"], record["predicted_labels"])
            if match_filter == "Exact match" and not is_exact:
                continue
            if match_filter == "Mismatch" and is_exact:
                continue
            if label_filter != "All" and label_filter not in set(
                record["true_labels"] + record["predicted_labels"]
            ):
                continue
            if text_query and text_query.lower() not in record["text"].lower():
                continue
            filtered.append(record)

        st.caption(f"Showing {len(filtered):,} of {len(records):,} comments")
        if not filtered:
            st.warning("No record matches the current filters.")
        else:
            options = {
                f"{i + 1}. ID {record['id']} — {record['text'][:80]}": record
                for i, record in enumerate(filtered)
            }
            selected_key = st.selectbox("Select a test comment", list(options.keys()))
            record = options[selected_key]

            exact = set_equal(record["true_labels"], record["predicted_labels"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Record ID", str(record["id"]))
            m2.metric("Predicted spans", len(record["predicted_spans"]))
            m3.metric("Document labels", "Exact" if exact else "Mismatch")

            c_gold, c_pred = st.columns(2)
            with c_gold:
                st.markdown('<div class="section-label">Ground-truth labels</div>', unsafe_allow_html=True)
                st.markdown(label_badges(record["true_labels"]), unsafe_allow_html=True)
            with c_pred:
                st.markdown('<div class="section-label">Predicted labels</div>', unsafe_allow_html=True)
                st.markdown(
                    label_badges(record["predicted_labels"]), unsafe_allow_html=True
                )

            st.subheader("Predicted evidence spans")
            st.markdown(
                render_highlighted_text(record["text"], record["predicted_spans"]),
                unsafe_allow_html=True,
            )

            with st.expander("Compare with ground-truth spans", expanded=False):
                st.markdown(
                    render_highlighted_text(record["text"], record["true_spans"]),
                    unsafe_allow_html=True,
                )

            left, right = st.columns([1, 1])
            with left:
                st.subheader("Label probabilities")
                st.dataframe(
                    probability_dataframe(record),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.3f"
                        )
                    },
                )
            with right:
                st.subheader("Evidence span details")
                st.dataframe(
                    spans_dataframe(record["predicted_spans"]),
                    hide_index=True,
                    use_container_width=True,
                )

            with st.expander("Raw exported record"):
                st.json(record)

with inference_tab:
    if not artifact_is_complete(DEFAULT_ARTIFACT_DIR):
        st.warning(
            "Live inference is unavailable because the full model artifacts have not been found. "
            "The test-data viewer still works with the precomputed prediction file."
        )
        st.code(
            "Required: artifact_config.json, thresholds.json, model_state.pt, "
            "encoder_config/config.json, and tokenizer/",
            language="text",
        )
    else:
        st.subheader("Run EviSpan-PR on a new peer-review comment")
        text = st.text_area(
            "Peer-review comment",
            value=(
                "Materi sudah disampaikan dengan jelas dan menarik. Namun, bagian evaluasi "
                "masih kurang mendalam. Sebaiknya tambahkan contoh data dan penjelasan metode "
                "yang lebih rinci."
            ),
            height=150,
        )
        if st.button("Predict", type="primary"):
            try:
                model, tokenizer, thresholds, config, device = get_live_model(
                    str(DEFAULT_ARTIFACT_DIR)
                )
                with st.spinner("Running document and span prediction..."):
                    result = predict_text(
                        text, model, tokenizer, thresholds, config, device
                    )
                st.markdown(
                    '<div class="section-label">Predicted labels</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    label_badges(result["predicted_labels"]), unsafe_allow_html=True
                )
                st.markdown(
                    render_highlighted_text(text, result["predicted_spans"]),
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Label": label,
                                "Probability": result["label_probabilities"][label],
                                "Threshold": float(thresholds[i]),
                                "Predicted": label in result["predicted_labels"],
                            }
                            for i, label in enumerate(LABELS)
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.3f"
                        ),
                        "Threshold": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                if result.get("truncated"):
                    st.info(
                        "The input exceeded the configured maximum token length and was truncated."
                    )
            except Exception as exc:
                st.exception(exc)
