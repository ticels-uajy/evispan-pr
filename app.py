from __future__ import annotations

import hashlib
import html
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st

from evispan_model import LABELS, load_artifacts, predict_text


st.set_page_config(
    page_title="EviSpan-PR: Evidence-Grounded Multi-Label Peer Review Feedback Analysis",
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
TEXT_COLUMN_CANDIDATES = [
    "text",
    "comment",
    "feedback",
    "review",
    "sentence",
    "content",
    "peer_feedback",
]
ID_COLUMN_CANDIDATES = ["id", "sample_id", "row_id", "comment_id", "index"]


@st.cache_data(show_spinner=False)
def read_test_dataset(data: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV, JSON, or JSONL test dataset."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(data))
    elif suffix == ".json":
        parsed = json.loads(data.decode("utf-8-sig"))
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = next(
                (
                    parsed[key]
                    for key in ("records", "data", "items")
                    if isinstance(parsed.get(key), list)
                ),
                [parsed],
            )
        else:
            raise ValueError("JSON must contain an object or a list of objects.")
        frame = pd.json_normalize(records)
    elif suffix in {".jsonl", ".ndjson"}:
        records = [
            json.loads(line)
            for line in data.decode("utf-8-sig").splitlines()
            if line.strip()
        ]
        frame = pd.json_normalize(records)
    else:
        raise ValueError("Unsupported file type. Use CSV, JSON, or JSONL.")

    if frame.empty:
        raise ValueError("The uploaded test dataset contains no rows.")

    # Avoid ambiguous duplicate column names in Streamlit widgets.
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns are not supported: {duplicated}")

    return frame.reset_index(drop=True)


def preferred_column(
    columns: Iterable[str],
    candidates: List[str],
    fallback_to_first: bool = True,
) -> Optional[str]:
    columns = list(columns)
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return str(columns[0]) if columns and fallback_to_first else None


def artifact_is_complete(path: Path) -> bool:
    required = [
        path / "artifact_config.json",
        path / "thresholds.json",
        path / "model_state.pt",
        path / "encoder_config" / "config.json",
        path / "tokenizer",
    ]
    return all(item.exists() for item in required)


@st.cache_resource(show_spinner="Memuat model EviSpan-PR...")
def get_live_model(artifact_dir: str):
    return load_artifacts(artifact_dir)


def label_badges(labels: Iterable[str]) -> str:
    labels = list(labels)
    if not labels:
        return '<span class="empty-badge">Tidak ada label</span>'

    badges = []
    for label in labels:
        color = LABEL_COLORS.get(label, "#f2f4f7")
        border = LABEL_BORDERS.get(label, "#475467")
        badges.append(
            f'<span class="label-badge" style="background:{color};'
            f'border-color:{border};color:{border}">{html.escape(label)}</span>'
        )
    return " ".join(badges)


def render_plain_text(text: str) -> str:
    escaped = html.escape(text).replace("\n", "<br>")
    return f'<div class="comment-box plain-text">{escaped}</div>'


def render_highlighted_text(text: str, spans: List[Dict[str, Any]]) -> str:
    """Highlight valid, non-overlapping evidence spans using character offsets."""
    safe_spans: List[Dict[str, Any]] = []
    cursor = 0

    for span in sorted(
        spans,
        key=lambda item: (
            int(item.get("start", 0)),
            -(int(item.get("end", 0)) - int(item.get("start", 0))),
        ),
    ):
        try:
            start = max(0, min(int(span["start"]), len(text)))
            end = max(0, min(int(span["end"]), len(text)))
        except (KeyError, TypeError, ValueError):
            continue

        label = str(span.get("label", ""))
        if label not in LABELS or start < cursor or start >= end:
            continue

        safe_spans.append({**span, "start": start, "end": end, "label": label})
        cursor = end

    if not safe_spans:
        return render_plain_text(text)

    parts: List[str] = []
    cursor = 0
    for span in safe_spans:
        start = int(span["start"])
        end = int(span["end"])
        label = str(span["label"])
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
    return (
        '<div class="comment-box evidence-text">'
        + "".join(parts).replace("\n", "<br>")
        + "</div>"
    )


def spans_dataframe(spans: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = ["Label", "Evidence", "Start", "End"]
    if not spans:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        [
            {
                "Label": span.get("label", ""),
                "Evidence": span.get("text", ""),
                "Start": span.get("start", ""),
                "End": span.get("end", ""),
            }
            for span in spans
        ],
        columns=columns,
    )


def probabilities_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    probabilities = result.get("label_probabilities", {})
    predicted = set(result.get("predicted_labels", []))
    return pd.DataFrame(
        [
            {
                "Label": label,
                "Probability": float(probabilities.get(label, 0.0)),
                "Predicted": label in predicted,
            }
            for label in LABELS
        ]
    )


def _set_page(index: int, total: int) -> None:
    bounded = max(0, min(index, total - 1))
    st.session_state.current_row = bounded
    st.session_state.page_jump = bounded + 1


def _previous_page(total: int) -> None:
    _set_page(int(st.session_state.current_row) - 1, total)


def _next_page(total: int) -> None:
    _set_page(int(st.session_state.current_row) + 1, total)


def _jump_to_page(total: int) -> None:
    requested = int(st.session_state.page_jump) - 1
    st.session_state.current_row = max(0, min(requested, total - 1))


def render_navigation(total: int, key_prefix: str) -> None:
    current = int(st.session_state.current_row)
    previous_col, page_col, next_col = st.columns([1, 2, 1])

    previous_col.button(
        "← Previous",
        key=f"{key_prefix}_previous",
        use_container_width=True,
        disabled=current <= 0,
        on_click=_previous_page,
        args=(total,),
    )
    page_col.number_input(
        "Baris",
        min_value=1,
        max_value=total,
        step=1,
        key="page_jump",
        on_change=_jump_to_page,
        args=(total,),
        label_visibility="collapsed",
    )
    next_col.button(
        "Next →",
        key=f"{key_prefix}_next",
        use_container_width=True,
        disabled=current >= total - 1,
        on_click=_next_page,
        args=(total,),
    )
    st.caption(f"Baris {current + 1:,} dari {total:,}")


st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
.label-badge, .empty-badge {
    display:inline-block; padding:0.30rem 0.68rem; margin:0.10rem 0.14rem 0.10rem 0;
    border:1px solid #98a2b3; border-radius:999px; font-weight:700;
}
.empty-badge {background:#f2f4f7; color:#475467;}
.comment-box {
    border:1px solid #d0d5dd; border-radius:12px; padding:1.15rem 1.25rem;
    line-height:2.05; font-size:1.05rem; background:#ffffff; color:#101828;
}
.plain-text {line-height:1.75;}
.evidence {padding:0.12rem 0.20rem; border-radius:5px; position:relative;}
.evidence small {
    font-size:0.62rem; margin-left:0.28rem; padding:0.08rem 0.22rem;
    border-radius:3px; background:rgba(255,255,255,0.80); font-weight:750;
}
.section-label {
    font-size:0.82rem; color:#667085; font-weight:750; text-transform:uppercase;
    letter-spacing:0.035em; margin-bottom:0.35rem;
}
.row-meta {
    border:1px solid #eaecf0; border-radius:10px; padding:0.70rem 0.85rem;
    background:#f9fafb; color:#344054;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title(
    "EviSpan-PR: Evidence-Grounded Multi-Label Peer Review Feedback Analysis"
)

st.caption(
    "Unggah test dataset untuk menelaah prediksi kategori feedback pada tingkat "
    "komentar—Problem, Suggestion, Neutral, dan Appreciation—serta evidence span "
    "yang mendukung setiap label, sehingga dosen dapat memverifikasi hasil analisis "
    "model secara transparan."
)

with st.sidebar:
    st.header("Test dataset")
    uploaded = st.file_uploader(
        "Upload test dataset",
        type=["csv", "json", "jsonl", "ndjson"],
        help="Dataset harus memiliki minimal satu kolom teks.",
    )
    artifact_dir_text = st.text_input(
        "Model artifact directory",
        value=str(DEFAULT_ARTIFACT_DIR),
        help="Folder yang berisi model_state.pt, thresholds.json, tokenizer, dan encoder_config.",
    )

if uploaded is None:
    st.info("Upload test dataset melalui panel sebelah kiri untuk mulai melakukan prediksi.")
    st.stop()

try:
    uploaded_bytes = uploaded.getvalue()
    test_df = read_test_dataset(uploaded_bytes, uploaded.name)
except Exception as exc:
    st.error(f"Test dataset tidak dapat dibaca: {exc}")
    st.stop()

available_columns = [str(column) for column in test_df.columns]
default_text_column = preferred_column(available_columns, TEXT_COLUMN_CANDIDATES)
default_text_index = (
    available_columns.index(default_text_column)
    if default_text_column in available_columns
    else 0
)

with st.sidebar:
    text_column = st.selectbox(
        "Kolom teks",
        options=available_columns,
        index=default_text_index,
    )

    id_options = ["(gunakan nomor baris)"] + available_columns
    preferred_id = preferred_column(
        available_columns,
        ID_COLUMN_CANDIDATES,
        fallback_to_first=False,
    )
    default_id_index = (
        id_options.index(preferred_id)
        if preferred_id in id_options
        else 0
    )
    id_column_selection = st.selectbox(
        "Kolom ID",
        options=id_options,
        index=default_id_index,
    )

# Keep only usable test rows. The displayed numbering still follows this cleaned view.
working_df = test_df.copy()
working_df[text_column] = working_df[text_column].fillna("").astype(str)
working_df = working_df[working_df[text_column].str.strip().ne("")].reset_index(drop=True)

if working_df.empty:
    st.error(f"Kolom `{text_column}` tidak memiliki teks yang dapat diprediksi.")
    st.stop()

artifact_dir = Path(artifact_dir_text).expanduser().resolve()
if not artifact_is_complete(artifact_dir):
    st.error(
        "Model artifact belum lengkap. Pastikan folder berisi `artifact_config.json`, "
        "`thresholds.json`, `model_state.pt`, `encoder_config/config.json`, dan folder `tokenizer`."
    )
    st.code(str(artifact_dir), language="text")
    st.stop()

file_signature = hashlib.sha256(uploaded_bytes).hexdigest()
dataset_signature = f"{file_signature}:{text_column}:{len(working_df)}"
model_state_path = artifact_dir / "model_state.pt"
model_signature = f"{artifact_dir}:{model_state_path.stat().st_mtime_ns}"

if st.session_state.get("active_dataset_signature") != dataset_signature:
    st.session_state.active_dataset_signature = dataset_signature
    st.session_state.current_row = 0
    st.session_state.page_jump = 1
    st.session_state.prediction_cache = {}

if "current_row" not in st.session_state:
    st.session_state.current_row = 0
if "page_jump" not in st.session_state:
    st.session_state.page_jump = 1
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}

_set_page(int(st.session_state.current_row), len(working_df))
render_navigation(len(working_df), "top")
st.divider()

row_position = int(st.session_state.current_row)
row = working_df.iloc[row_position]
text = str(row[text_column]).strip()
row_id = (
    row_position + 1
    if id_column_selection == "(gunakan nomor baris)"
    else row[id_column_selection]
)

cache_key = hashlib.sha256(
    f"{dataset_signature}:{model_signature}:{row_position}:{text}".encode("utf-8")
).hexdigest()

if cache_key not in st.session_state.prediction_cache:
    try:
        model, tokenizer, thresholds, config, device = get_live_model(str(artifact_dir))
        with st.spinner(f"Memprediksi baris {row_position + 1:,}..."):
            st.session_state.prediction_cache[cache_key] = predict_text(
                text=text,
                model=model,
                tokenizer=tokenizer,
                thresholds=thresholds,
                config=config,
                device=device,
            )
    except Exception as exc:
        st.error(f"Prediksi gagal dijalankan pada baris ini: {exc}")
        st.stop()

result = st.session_state.prediction_cache[cache_key]
predicted_labels = result.get("predicted_labels", [])
predicted_spans = result.get("predicted_spans", [])

meta_1, meta_2, meta_3 = st.columns(3)
meta_1.metric("ID data", str(row_id))
meta_2.metric("Jumlah label", len(predicted_labels))
meta_3.metric("Jumlah evidence span", len(predicted_spans))

st.markdown('<div class="section-label">Teks pada test dataset</div>', unsafe_allow_html=True)
st.markdown(render_plain_text(text), unsafe_allow_html=True)

st.markdown('<div class="section-label" style="margin-top:1.35rem">Prediksi label</div>', unsafe_allow_html=True)
st.markdown(label_badges(predicted_labels), unsafe_allow_html=True)

st.markdown('<div class="section-label" style="margin-top:1.35rem">Evidence span hasil prediksi</div>', unsafe_allow_html=True)
st.markdown(render_highlighted_text(text, predicted_spans), unsafe_allow_html=True)

if predicted_spans:
    st.dataframe(
        spans_dataframe(predicted_spans),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Evidence": st.column_config.TextColumn(width="large"),
            "Start": st.column_config.NumberColumn(format="%d"),
            "End": st.column_config.NumberColumn(format="%d"),
        },
    )
else:
    st.warning("Model tidak menghasilkan evidence span untuk baris ini.")

with st.expander("Lihat probabilitas setiap label", expanded=False):
    st.dataframe(
        probabilities_dataframe(result),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
            "Predicted": st.column_config.CheckboxColumn(),
        },
    )

if result.get("truncated"):
    st.info("Teks melebihi panjang maksimum model dan dipotong saat inferensi.")

st.caption("Gunakan tombol Previous/Next atau nomor baris di bagian atas untuk berpindah data.")
