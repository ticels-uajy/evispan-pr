from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import streamlit as st



LABELS = ["Problem", "Suggestion", "Neutral", "Appreciation"]

st.set_page_config(
    page_title="EviSpan-PR: Evidence-Grounded Peer Review Feedback Analysis",
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
LABEL_DESCRIPTIONS = {
    "Appreciation": (
        "Penilaian evaluatif positif yang secara eksplisit diberikan oleh rekan sejawat "
        "terhadap karya yang dikumpulkan, tanpa mengusulkan perubahan. Tujuannya adalah "
        "memberikan pengakuan atau pujian, misalnya apresiasi, ungkapan kepuasan, atau persetujuan."
    ),
    "Problem": (
        "Identifikasi terhadap kekurangan, kesalahan, ketidakkonsistenan, ambiguitas, "
        "atau kelemahan lain dalam karya yang memerlukan perbaikan atau klarifikasi."
    ),
    "Suggestion": (
        "Rekomendasi yang mengusulkan tindakan konkret dan dapat ditindaklanjuti untuk "
        "memperbaiki karya. Saran disampaikan sebagai nasihat atau usulan, misalnya "
        "‘pertimbangkan…’, ‘Anda dapat…’, atau ‘akan lebih jelas jika…’, bukan sekadar "
        "melaporkan adanya kekurangan."
    ),
    "Neutral": (
        "Teks peer review yang bersifat informatif, tidak menyatakan penilaian evaluatif—"
        "baik positif maupun negatif—dan tidak menyarankan perubahan."
    ),
}

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = Path(
    os.getenv("EVISPAN_ARTIFACT_DIR", str(PROJECT_DIR / "artifacts" / "evispan_pr"))
)
DEFAULT_PREDICTION_PATH = Path(
    os.getenv(
        "EVISPAN_TEST_PREDICTIONS",
        str(DEFAULT_ARTIFACT_DIR / "test_predictions.jsonl"),
    )
)
RESPONSE_DB_PATH = Path(
    os.getenv(
        "EVISPAN_RESPONSE_DB",
        str(PROJECT_DIR / "data" / "evispan_lecturer_responses.sqlite3"),
    )
)
TTF_QUESTIONNAIRE_URL = os.getenv(
    "EVISPAN_TTF_URL",
    "https://forms.gle/REPLACE_WITH_YOUR_TTF_FORM_ID",
).strip()
DEFAULT_REVIEW_TARGET = max(1, int(os.getenv("EVISPAN_REVIEW_TARGET", "10")))
RESPONDENT_CODE_PREFIX = (os.getenv("EVISPAN_RESPONDENT_PREFIX", "D").strip().upper() or "D")
RESPONDENT_CODE_WIDTH = max(3, int(os.getenv("EVISPAN_RESPONDENT_CODE_WIDTH", "4")))
CONSENT_VERSION = "2026-07-v1"
CONSENT_TEXT = (
    "Saya bersedia terlibat sebagai responden dalam penelitian EviSpan-PR. Saya memahami "
    "bahwa partisipasi ini bersifat sukarela dan respons yang saya berikan akan digunakan "
    "untuk kepentingan penelitian dan evaluasi sistem. Data pribadi tidak akan disalahgunakan. "
    "Identitas dan data pribadi akan disamarkan serta tidak ditampilkan dalam proses analisis, "
    "pelaporan, maupun publikasi hasil penelitian."
)

STUDY_STATE_KEYS = [
    "study_session_id",
    "study_phase",
    "review_indices",
    "review_position",
    "top_review_page_jump",
    "bottom_review_page_jump",
    "study_responses",
    "respondent_code",
    "respondent_name",
    "respondent_unit",
    "review_target",
    "sampling_method",
    "final_response",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@st.cache_data(show_spinner=False)
def read_prediction_file(path_text: str, modified_time_ns: int) -> pd.DataFrame:
    """Read and normalize precomputed predictions from JSON or JSONL."""
    del modified_time_ns  # Included only to invalidate Streamlit cache when the file changes.
    path = Path(path_text)
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
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
            raise ValueError("File prediksi JSON harus berisi objek atau daftar objek.")
    else:
        raise ValueError("File prediksi harus menggunakan format JSONL, NDJSON, atau JSON.")

    if not records:
        raise ValueError("File test_predictions tidak memiliki data.")

    normalized = [normalize_prediction_record(record, index) for index, record in enumerate(records)]
    frame = pd.DataFrame(normalized)
    frame = frame[frame["text"].str.strip().ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Tidak ada teks komentar yang dapat ditampilkan.")
    if not frame["prediction_fields_present"].all():
        raise ValueError(
            "Sebagian baris tidak memiliki field hasil prediksi. Pastikan file memuat "
            "predicted_labels, label_probabilities, dan predicted_spans."
        )
    return frame


def parse_json_like(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            if isinstance(default, list):
                return [part.strip() for part in stripped.split(",") if part.strip()]
    return default


def first_present(record: Dict[str, Any], keys: List[str], default: Any) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def normalize_spans(spans: Any, text: str) -> List[Dict[str, Any]]:
    parsed = parse_json_like(spans, [])
    if not isinstance(parsed, list):
        return []

    output: List[Dict[str, Any]] = []
    for span in parsed:
        if not isinstance(span, dict):
            continue
        label = str(span.get("label", ""))
        if label not in LABELS:
            continue
        try:
            start = int(span.get("start"))
            end = int(span.get("end"))
        except (TypeError, ValueError):
            continue
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        if start >= end:
            continue
        output.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": str(span.get("text") or text[start:end]),
            }
        )
    return sorted(output, key=lambda item: (item["start"], item["end"], item["label"]))


def normalize_prediction_record(record: Any, index: int) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Baris ke-{index + 1} bukan objek JSON yang valid.")

    text = str(
        first_present(
            record,
            ["text", "comment", "feedback", "review", "sentence", "content", "peer_feedback"],
            "",
        )
    ).strip()
    row_id = first_present(record, ["id", "sample_id", "row_id", "comment_id"], index + 1)

    predicted_labels_raw = parse_json_like(
        first_present(record, ["predicted_labels", "pred_labels"], []),
        [],
    )
    predicted_labels = [
        str(label) for label in predicted_labels_raw if str(label) in LABELS
    ] if isinstance(predicted_labels_raw, list) else []

    probabilities_raw = parse_json_like(
        first_present(record, ["label_probabilities", "pred_probs", "probabilities"], {}),
        {},
    )
    if not isinstance(probabilities_raw, dict):
        probabilities_raw = {}

    probabilities: Dict[str, float] = {}
    for label in LABELS:
        candidates = [
            probabilities_raw.get(label),
            probabilities_raw.get(label.lower()),
            record.get(f"prob_{label.lower()}"),
        ]
        value = next((candidate for candidate in candidates if candidate is not None), 0.0)
        try:
            probabilities[label] = float(value)
        except (TypeError, ValueError):
            probabilities[label] = 0.0

    spans_raw = first_present(record, ["predicted_spans", "pred_spans"], [])
    label_fields = {"predicted_labels", "pred_labels"}
    probability_fields = {"label_probabilities", "pred_probs", "probabilities"}
    span_fields = {"predicted_spans", "pred_spans"}
    prediction_fields_present = (
        bool(label_fields.intersection(record.keys()))
        and bool(probability_fields.intersection(record.keys()))
        and bool(span_fields.intersection(record.keys()))
    )

    return {
        "source_row_index": index,
        "row_id": str(row_id),
        "text": text,
        "predicted_labels": predicted_labels,
        "label_probabilities": probabilities,
        "predicted_spans": normalize_spans(spans_raw, text),
        "truncated": bool(record.get("truncated", False)),
        "prediction_fields_present": prediction_fields_present,
    }


def render_label_guide() -> str:
    cards = []
    for label in ["Appreciation", "Problem", "Suggestion", "Neutral"]:
        color = LABEL_COLORS[label]
        border = LABEL_BORDERS[label]
        description = LABEL_DESCRIPTIONS[label]
        cards.append(
            '<div class="label-guide-card" '
            f'style="border-left:5px solid {border};background:{color};">'
            f'<div class="label-guide-title" style="color:{border};">'
            f"{html.escape(label)}</div>"
            f'<div class="label-guide-description">{html.escape(description)}</div>'
            "</div>"
        )
    return '<div class="label-guide-grid">' + "".join(cards) + "</div>"


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
            f"{html.escape(text[start:end])}"
            f"<small>{html.escape(label)}</small></span>"
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


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    if column_name not in _table_columns(connection, table_name):
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def _respondent_number_from_code(code: Any) -> int | None:
    match = re.fullmatch(
        rf"{re.escape(RESPONDENT_CODE_PREFIX)}-(\d+)",
        str(code or "").strip().upper(),
    )
    return int(match.group(1)) if match else None


def _ensure_respondent_counter(connection: sqlite3.Connection) -> int:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_counters (
            counter_name TEXT PRIMARY KEY,
            counter_value INTEGER NOT NULL
        )
        """
    )
    row = connection.execute(
        "SELECT counter_value FROM app_counters WHERE counter_name='respondent_code'"
    ).fetchone()
    if row is not None:
        return int(row[0])

    existing_codes = [
        row[0]
        for row in connection.execute(
            "SELECT respondent_code FROM study_sessions WHERE respondent_code IS NOT NULL"
        ).fetchall()
    ]
    existing_codes.extend(
        row[0]
        for row in connection.execute(
            "SELECT respondent_code FROM respondent_registry WHERE respondent_code IS NOT NULL"
        ).fetchall()
    )
    existing_numbers = [
        number
        for code in existing_codes
        if (number := _respondent_number_from_code(code)) is not None
    ]
    current = max(existing_numbers, default=0)
    connection.execute(
        "INSERT INTO app_counters(counter_name, counter_value) VALUES ('respondent_code', ?)",
        (current,),
    )
    return current


def _format_respondent_code(number: int) -> str:
    return f"{RESPONDENT_CODE_PREFIX}-{number:0{RESPONDENT_CODE_WIDTH}d}"


def _next_available_respondent_code(
    connection: sqlite3.Connection,
    reserve_for_session_id: str | None = None,
) -> str:
    current = _ensure_respondent_counter(connection)
    while True:
        current += 1
        candidate = _format_respondent_code(current)
        exists = connection.execute(
            """
            SELECT 1 FROM study_sessions WHERE respondent_code = ?
            UNION ALL
            SELECT 1 FROM respondent_registry WHERE respondent_code = ?
            LIMIT 1
            """,
            (candidate, candidate),
        ).fetchone()
        if exists is None:
            break

    if reserve_for_session_id is not None:
        connection.execute(
            "UPDATE app_counters SET counter_value=? WHERE counter_name='respondent_code'",
            (current,),
        )
        connection.execute(
            """
            INSERT INTO respondent_registry (
                respondent_code, session_id, created_at
            ) VALUES (?, ?, ?)
            """,
            (candidate, reserve_for_session_id, utc_now_iso()),
        )
    return candidate


def preview_next_respondent_code(path: Path) -> str:
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        return _next_available_respondent_code(connection)


def initialise_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS study_sessions (
                session_id TEXT PRIMARY KEY,
                respondent_code TEXT NOT NULL,
                respondent_name TEXT,
                respondent_unit TEXT,
                dataset_name TEXT NOT NULL,
                dataset_signature TEXT NOT NULL,
                text_column TEXT NOT NULL,
                id_column TEXT,
                review_target INTEGER NOT NULL,
                sampling_method TEXT NOT NULL,
                review_indices_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                consent_given INTEGER NOT NULL DEFAULT 0,
                consent_version TEXT,
                consent_text TEXT,
                consented_at TEXT
            );

            CREATE TABLE IF NOT EXISTS respondent_registry (
                respondent_code TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_counters (
                counter_name TEXT PRIMARY KEY,
                counter_value INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS comment_responses (
                session_id TEXT NOT NULL,
                source_row_index INTEGER NOT NULL,
                review_order INTEGER NOT NULL,
                row_id TEXT,
                comment_text TEXT NOT NULL,
                predicted_labels_json TEXT NOT NULL,
                label_probabilities_json TEXT NOT NULL,
                predicted_spans_json TEXT NOT NULL,
                label_assessment TEXT NOT NULL,
                expected_labels_json TEXT NOT NULL,
                evidence_support TEXT NOT NULL,
                evidence_boundary TEXT NOT NULL,
                identified_patterns_json TEXT NOT NULL,
                interpretation TEXT,
                confidence INTEGER NOT NULL,
                notes TEXT,
                reviewed_at TEXT NOT NULL,
                PRIMARY KEY (session_id, source_row_index),
                FOREIGN KEY (session_id) REFERENCES study_sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS final_responses (
                session_id TEXT PRIMARY KEY,
                easiest_label TEXT NOT NULL,
                hardest_label TEXT NOT NULL,
                recurring_patterns TEXT NOT NULL,
                recurring_mismatches TEXT,
                usefulness_rating INTEGER NOT NULL,
                intended_use TEXT,
                overall_notes TEXT,
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES study_sessions(session_id)
            );
            """
        )

        # Backward-compatible migration for databases created by earlier app versions.
        _ensure_column(connection, "study_sessions", "consent_given", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "study_sessions", "consent_version", "TEXT")
        _ensure_column(connection, "study_sessions", "consent_text", "TEXT")
        _ensure_column(connection, "study_sessions", "consented_at", "TEXT")

        # Register historical respondent codes when possible. INSERT OR IGNORE keeps
        # legacy duplicate codes from breaking migration while all new IDs are reserved
        # atomically through respondent_registry.
        connection.execute(
            """
            INSERT OR IGNORE INTO respondent_registry (
                respondent_code, session_id, created_at
            )
            SELECT respondent_code, session_id, started_at
            FROM study_sessions
            WHERE respondent_code IS NOT NULL AND TRIM(respondent_code) <> ''
            """
        )
        _ensure_respondent_counter(connection)
        connection.commit()


def create_study_session(path: Path, payload: Dict[str, Any]) -> str:
    """Create a session and atomically reserve a unique respondent code."""
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        respondent_code = _next_available_respondent_code(
            connection,
            reserve_for_session_id=payload["session_id"],
        )
        connection.execute(
            """
            INSERT INTO study_sessions (
                session_id, respondent_code, respondent_name, respondent_unit,
                dataset_name, dataset_signature, text_column, id_column,
                review_target, sampling_method, review_indices_json, started_at,
                consent_given, consent_version, consent_text, consented_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                respondent_code,
                payload.get("respondent_name", ""),
                payload.get("respondent_unit", ""),
                payload["dataset_name"],
                payload["dataset_signature"],
                payload["text_column"],
                payload.get("id_column", ""),
                int(payload["review_target"]),
                payload["sampling_method"],
                json.dumps(payload["review_indices"]),
                payload["started_at"],
                1 if payload.get("consent_given") else 0,
                payload.get("consent_version", CONSENT_VERSION),
                payload.get("consent_text", CONSENT_TEXT),
                payload.get("consented_at", payload["started_at"]),
            ),
        )
        connection.commit()
        return respondent_code
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def save_comment_response(path: Path, payload: Dict[str, Any]) -> None:
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO comment_responses (
                session_id, source_row_index, review_order, row_id, comment_text,
                predicted_labels_json, label_probabilities_json, predicted_spans_json,
                label_assessment, expected_labels_json, evidence_support,
                evidence_boundary, identified_patterns_json, interpretation,
                confidence, notes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, source_row_index) DO UPDATE SET
                review_order=excluded.review_order,
                row_id=excluded.row_id,
                comment_text=excluded.comment_text,
                predicted_labels_json=excluded.predicted_labels_json,
                label_probabilities_json=excluded.label_probabilities_json,
                predicted_spans_json=excluded.predicted_spans_json,
                label_assessment=excluded.label_assessment,
                expected_labels_json=excluded.expected_labels_json,
                evidence_support=excluded.evidence_support,
                evidence_boundary=excluded.evidence_boundary,
                identified_patterns_json=excluded.identified_patterns_json,
                interpretation=excluded.interpretation,
                confidence=excluded.confidence,
                notes=excluded.notes,
                reviewed_at=excluded.reviewed_at
            """,
            (
                payload["session_id"],
                int(payload["source_row_index"]),
                int(payload["review_order"]),
                str(payload.get("row_id", "")),
                payload["comment_text"],
                json.dumps(payload["predicted_labels"], ensure_ascii=False),
                json.dumps(payload["label_probabilities"], ensure_ascii=False),
                json.dumps(payload["predicted_spans"], ensure_ascii=False),
                payload["label_assessment"],
                json.dumps(payload["expected_labels"], ensure_ascii=False),
                payload["evidence_support"],
                payload["evidence_boundary"],
                json.dumps(payload["identified_patterns"], ensure_ascii=False),
                payload.get("interpretation", ""),
                int(payload["confidence"]),
                payload.get("notes", ""),
                payload["reviewed_at"],
            ),
        )
        connection.commit()


def save_final_response(path: Path, payload: Dict[str, Any]) -> None:
    submitted_at = payload["submitted_at"]
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO final_responses (
                session_id, easiest_label, hardest_label, recurring_patterns,
                recurring_mismatches, usefulness_rating, intended_use,
                overall_notes, submitted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                easiest_label=excluded.easiest_label,
                hardest_label=excluded.hardest_label,
                recurring_patterns=excluded.recurring_patterns,
                recurring_mismatches=excluded.recurring_mismatches,
                usefulness_rating=excluded.usefulness_rating,
                intended_use=excluded.intended_use,
                overall_notes=excluded.overall_notes,
                submitted_at=excluded.submitted_at
            """,
            (
                payload["session_id"],
                payload["easiest_label"],
                payload["hardest_label"],
                payload["recurring_patterns"],
                payload.get("recurring_mismatches", ""),
                int(payload["usefulness_rating"]),
                payload.get("intended_use", ""),
                payload.get("overall_notes", ""),
                submitted_at,
            ),
        )
        connection.execute(
            "UPDATE study_sessions SET completed_at=? WHERE session_id=?",
            (submitted_at, payload["session_id"]),
        )
        connection.commit()


def session_responses_dataframe(path: Path, session_id: str) -> pd.DataFrame:
    with sqlite3.connect(path, timeout=30) as connection:
        return pd.read_sql_query(
            """
            SELECT
                s.session_id,
                s.respondent_code,
                s.respondent_name,
                s.respondent_unit,
                s.consent_given,
                s.consent_version,
                s.consented_at,
                s.dataset_name,
                c.review_order,
                c.source_row_index,
                c.row_id,
                c.comment_text,
                c.predicted_labels_json,
                c.label_probabilities_json,
                c.predicted_spans_json,
                c.label_assessment,
                c.expected_labels_json,
                c.evidence_support,
                c.evidence_boundary,
                c.identified_patterns_json,
                c.interpretation,
                c.confidence,
                c.notes,
                c.reviewed_at
            FROM comment_responses c
            JOIN study_sessions s ON s.session_id = c.session_id
            WHERE c.session_id = ?
            ORDER BY c.review_order
            """,
            connection,
            params=(session_id,),
        )


def session_final_dataframe(path: Path, session_id: str) -> pd.DataFrame:
    with sqlite3.connect(path, timeout=30) as connection:
        return pd.read_sql_query(
            """
            SELECT
                s.session_id,
                s.respondent_code,
                s.respondent_name,
                s.respondent_unit,
                s.consent_given,
                s.consent_version,
                s.consented_at,
                f.easiest_label,
                f.hardest_label,
                f.recurring_patterns,
                f.recurring_mismatches,
                f.usefulness_rating,
                f.intended_use,
                f.overall_notes,
                f.submitted_at
            FROM final_responses f
            JOIN study_sessions s ON s.session_id = f.session_id
            WHERE f.session_id = ?
            """,
            connection,
            params=(session_id,),
        )


def render_workflow(active_phase: str, completed: int, target: int) -> None:
    phases = [
        ("1", "Pahami label", active_phase in {"setup", "review", "final", "complete"}),
        ("2", "Gunakan EviSpan-PR", active_phase in {"review", "final", "complete"}),
        ("3", "Telaah komentar", active_phase in {"review", "final", "complete"}),
        ("4", "Interpretasi & pola", active_phase in {"final", "complete"}),
        ("5", "Kuesioner TTF", active_phase == "complete"),
    ]
    cards = []
    for number, label, reached in phases:
        status_class = "workflow-active" if reached else "workflow-pending"
        cards.append(
            f'<div class="workflow-card {status_class}"><span>{number}</span>{html.escape(label)}</div>'
        )
    st.markdown('<div class="workflow-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    if target:
        st.progress(min(completed / target, 1.0), text=f"Respons komentar tersimpan: {completed}/{target}")


def reset_study_state() -> None:
    for key in STUDY_STATE_KEYS:
        st.session_state.pop(key, None)


def _set_review_position(index: int, total: int) -> None:
    bounded = max(0, min(index, total - 1))
    st.session_state.review_position = bounded


def _previous_review(total: int) -> None:
    _set_review_position(int(st.session_state.review_position) - 1, total)


def _next_review(total: int) -> None:
    _set_review_position(int(st.session_state.review_position) + 1, total)


def _jump_review(total: int, widget_key: str) -> None:
    requested = int(st.session_state[widget_key]) - 1
    _set_review_position(requested, total)


def render_review_navigation(total: int, key_prefix: str) -> None:
    """Render independent top/bottom navigation widgets with unique Streamlit keys."""
    current = int(st.session_state.review_position)
    previous_col, page_col, next_col = st.columns([1, 2, 1])
    previous_col.button(
        "← Previous",
        key=f"{key_prefix}_previous",
        use_container_width=True,
        disabled=current <= 0,
        on_click=_previous_review,
        args=(total,),
    )

    page_key = f"{key_prefix}_review_page_jump"
    # The top and bottom number inputs must use different keys. Synchronize each
    # widget to the currently active record before it is instantiated.
    st.session_state[page_key] = current + 1
    page_col.number_input(
        "Komentar",
        min_value=1,
        max_value=total,
        step=1,
        key=page_key,
        on_change=_jump_review,
        args=(total, page_key),
        label_visibility="collapsed",
    )
    next_col.button(
        "Next →",
        key=f"{key_prefix}_next",
        use_container_width=True,
        disabled=current >= total - 1,
        on_click=_next_review,
        args=(total,),
    )
    st.caption(f"Komentar {current + 1} dari {total}")


def valid_questionnaire_url(url: str) -> bool:
    return url.startswith(("https://", "http://")) and "REPLACE_WITH" not in url


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
.label-guide-grid {
    display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.75rem;
    margin-top:0.35rem; margin-bottom:0.35rem;
}
.label-guide-card {
    border:1px solid rgba(16,24,40,0.10); border-radius:10px;
    padding:0.85rem 0.95rem; min-height:132px;
}
.label-guide-title {font-weight:800; font-size:1rem; margin-bottom:0.35rem;}
.label-guide-description {color:#344054; line-height:1.55; font-size:0.92rem;}
.workflow-grid {
    display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:0.55rem;
    margin:1rem 0 0.8rem 0;
}
.workflow-card {
    border-radius:10px; padding:0.70rem 0.55rem; text-align:center; font-size:0.84rem;
    border:1px solid #d0d5dd; font-weight:650;
}
.workflow-card span {
    display:inline-flex; width:1.45rem; height:1.45rem; align-items:center; justify-content:center;
    border-radius:999px; margin-right:0.35rem; font-weight:800;
}
.workflow-active {background:#eff8ff; border-color:#84caff; color:#175cd3;}
.workflow-active span {background:#175cd3; color:white;}
.workflow-pending {background:#f9fafb; color:#667085;}
.workflow-pending span {background:#eaecf0; color:#667085;}
.response-panel {
    border:1px solid #d0d5dd; border-radius:12px; padding:1rem 1.1rem; background:#f9fafb;
}
.completion-box {
    border:1px solid #75e0a7; background:#ecfdf3; border-radius:14px;
    padding:1.25rem 1.35rem; margin:1rem 0;
}
@media (max-width: 820px) {
    .label-guide-grid {grid-template-columns:1fr;}
    .label-guide-card {min-height:auto;}
    .workflow-grid {grid-template-columns:1fr;}
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("EviSpan-PR: Evidence-Grounded Multi-Label Peer Review Feedback Analysis")
st.caption(
    "EviSpan-PR membantu dosen menelaah kategori feedback pada tingkat komentar dan "
    "evidence span yang mendukung setiap prediksi. Pada sesi ini, dosen diminta memeriksa "
    "label, probabilitas, serta bukti tekstual, kemudian menyelesaikan tugas interpretasi "
    "dan identifikasi pola sebelum mengisi kuesioner TTF."
)

try:
    initialise_database(RESPONSE_DB_PATH)
    database_ready = True
    database_error = ""
except Exception as exc:
    database_ready = False
    database_error = str(exc)

prediction_candidates = [
    DEFAULT_PREDICTION_PATH.expanduser(),
    PROJECT_DIR / "artifacts" / "test_predictions.jsonl",
]
prediction_path = next(
    (candidate.resolve() for candidate in prediction_candidates if candidate.exists()),
    prediction_candidates[0].resolve(),
)

with st.sidebar:
    st.header("Konfigurasi studi")
    st.markdown("**Sumber data prediksi**")
    st.caption(f"`{prediction_path}`")
    st.caption(
        "Komentar, label, probabilitas, dan evidence span dibaca otomatis dari "
        "`test_predictions.jsonl`; dosen tidak perlu mengunggah dataset."
    )
    st.caption(f"Penyimpanan respons: `{RESPONSE_DB_PATH}`")
    if not database_ready:
        st.error(f"Database respons tidak dapat digunakan: {database_error}")

if not prediction_path.exists():
    render_workflow("setup", 0, 0)
    with st.expander("1. Pelajari kategori feedback", expanded=True):
        st.markdown(
            "Satu komentar dapat memiliki lebih dari satu label. Baca definisi berikut sebelum "
            "memulai evaluasi."
        )
        st.markdown(render_label_guide(), unsafe_allow_html=True)
    st.error(
        "File hasil prediksi tidak ditemukan. Letakkan `test_predictions.jsonl` pada folder "
        "artifact yang dikonfigurasi."
    )
    st.code(str(prediction_path), language="text")
    st.caption(
        "Lokasi dapat diubah melalui environment variable `EVISPAN_TEST_PREDICTIONS`."
    )
    st.stop()

try:
    prediction_stat = prediction_path.stat()
    test_df = read_prediction_file(str(prediction_path), prediction_stat.st_mtime_ns)
except Exception as exc:
    st.error(f"File test_predictions.jsonl tidak dapat dibaca: {exc}")
    st.code(str(prediction_path), language="text")
    st.stop()

working_df = test_df.copy().reset_index(drop=True)
text_column = "text"
id_column_selection = "row_id"

with st.sidebar:
    st.success(f"{len(working_df):,} komentar prediksi siap ditelaah.")

dataset_signature = (
    f"{prediction_path}:{prediction_stat.st_size}:{prediction_stat.st_mtime_ns}:"
    f"{len(working_df)}"
)
study_signature = dataset_signature

if st.session_state.get("active_study_signature") != study_signature:
    reset_study_state()
    st.session_state.active_study_signature = study_signature

if "study_responses" not in st.session_state:
    st.session_state.study_responses = {}

active_phase = st.session_state.get("study_phase", "setup")
review_target_for_progress = int(st.session_state.get("review_target", 0) or 0)
completed_count = len(st.session_state.get("study_responses", {}))
render_workflow(active_phase, completed_count, review_target_for_progress)

with st.expander("1. Penjelasan kategori Problem, Suggestion, Neutral, dan Appreciation", expanded=active_phase == "setup"):
    st.markdown(
        "Satu komentar dapat memperoleh lebih dari satu label. Gunakan definisi berikut sebagai "
        "acuan saat menilai kesesuaian prediksi dan evidence span."
    )
    st.markdown(render_label_guide(), unsafe_allow_html=True)

if not database_ready:
    st.warning(
        "Respons masih dapat diisi dalam sesi browser, tetapi tidak dapat disimpan permanen karena "
        "database tidak tersedia. Perbaiki lokasi `EVISPAN_RESPONSE_DB` sebelum studi dijalankan."
    )

if "study_session_id" not in st.session_state:
    st.subheader("Mulai sesi evaluasi dosen")
    st.markdown(
        "Kode responden dibuat otomatis dari database SQLite agar setiap responden memiliki ID "
        "yang unik. Nama dan unit bersifat opsional."
    )

    max_target = len(working_df)
    default_target = min(DEFAULT_REVIEW_TARGET, max_target)
    try:
        respondent_code_preview = (
            preview_next_respondent_code(RESPONSE_DB_PATH)
            if database_ready
            else "Database belum tersedia"
        )
    except Exception as exc:
        respondent_code_preview = "Tidak dapat dibuat"
        st.warning(f"Pratinjau kode responden tidak tersedia: {exc}")

    with st.form("study_setup_form"):
        st.text_input(
            "Kode responden (otomatis)",
            value=respondent_code_preview,
            disabled=True,
            help=(
                "Kode final dialokasikan secara atomik ketika sesi dimulai dan diperiksa terhadap "
                "seluruh kode yang sudah tersimpan di SQLite."
            ),
        )
        respondent_name = st.text_input("Nama dosen (opsional)")
        respondent_unit = st.text_input("Program studi/unit (opsional)")
        review_target = st.number_input(
            "Jumlah komentar yang ditelaah",
            min_value=1,
            max_value=max_target,
            value=default_target,
            step=1,
        )
        sampling_method = st.radio(
            "Pemilihan komentar",
            options=["Acak", "Berurutan dari awal dataset"],
            horizontal=True,
            help="Pemilihan acak menggunakan seed tetap untuk sesi ini sehingga urutan tidak berubah.",
        )

        st.markdown("#### Persetujuan partisipasi (consent form)")
        st.info(CONSENT_TEXT)
        consent_given = st.checkbox(
            "Saya telah membaca penjelasan di atas dan bersedia terlibat dalam penelitian ini. *",
            value=False,
        )
        begin = st.form_submit_button("Setuju dan mulai menggunakan EviSpan-PR", type="primary")

    if begin:
        if not consent_given:
            st.error("Persetujuan partisipasi wajib diberikan sebelum sesi dimulai.")
            st.stop()
        if not database_ready:
            st.error("Sesi tidak dapat dimulai sebelum database respons tersedia.")
            st.stop()

        session_id = uuid.uuid4().hex
        target = int(review_target)
        if sampling_method == "Acak":
            seed_material = f"{study_signature}:{session_id}"
            seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
            review_indices = working_df.sample(n=target, random_state=seed).index.tolist()
        else:
            review_indices = list(range(target))

        started_at = utc_now_iso()
        session_payload = {
            "session_id": session_id,
            "respondent_name": respondent_name.strip(),
            "respondent_unit": respondent_unit.strip(),
            "dataset_name": prediction_path.name,
            "dataset_signature": dataset_signature,
            "text_column": text_column,
            "id_column": id_column_selection,
            "review_target": target,
            "sampling_method": sampling_method,
            "review_indices": review_indices,
            "started_at": started_at,
            "consent_given": True,
            "consent_version": CONSENT_VERSION,
            "consent_text": CONSENT_TEXT,
            "consented_at": started_at,
        }
        try:
            respondent_code = create_study_session(RESPONSE_DB_PATH, session_payload)
        except Exception as exc:
            st.error(f"Sesi tidak dapat disimpan: {exc}")
            st.stop()

        st.session_state.study_session_id = session_id
        st.session_state.respondent_code = respondent_code
        st.session_state.respondent_name = respondent_name.strip()
        st.session_state.respondent_unit = respondent_unit.strip()
        st.session_state.review_target = target
        st.session_state.sampling_method = sampling_method
        st.session_state.review_indices = review_indices
        st.session_state.review_position = 0
        st.session_state.study_responses = {}
        st.session_state.study_phase = "review"
        st.rerun()

    st.stop()

session_id = st.session_state.study_session_id
review_indices = list(st.session_state.review_indices)
review_target = int(st.session_state.review_target)
responses: Dict[str, Dict[str, Any]] = st.session_state.study_responses
completed_count = len(responses)

with st.sidebar:
    st.divider()
    st.subheader("Sesi aktif")
    st.write(f"**Kode responden:** {st.session_state.respondent_code}")
    st.write(f"**Komentar:** {review_target}")
    st.progress(completed_count / review_target, text=f"{completed_count}/{review_target} tersimpan")
    if completed_count:
        try:
            export_df = session_responses_dataframe(RESPONSE_DB_PATH, session_id)
            st.download_button(
                "Download respons sementara",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"evispan_responses_{st.session_state.respondent_code}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.caption(f"Export sementara belum tersedia: {exc}")

if st.session_state.get("study_phase") == "complete":
    render_workflow("complete", review_target, review_target)
    st.markdown(
        '<div class="completion-box"><strong>Sesi evaluasi EviSpan-PR telah selesai.</strong><br>'
        "Respons komentar dan tugas identifikasi pola telah tersimpan. Tahap terakhir adalah "
        "mengisi kuesioner Task–Technology Fit (TTF).</div>",
        unsafe_allow_html=True,
    )
    st.write(f"**Kode penyelesaian:** `{session_id[:12].upper()}`")

    if valid_questionnaire_url(TTF_QUESTIONNAIRE_URL):
        st.link_button(
            "Buka kuesioner TTF",
            TTF_QUESTIONNAIRE_URL,
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Gunakan kode responden dan kode penyelesaian di atas apabila diminta pada kuesioner."
        )
    else:
        st.warning(
            "Link kuesioner TTF belum dikonfigurasi. Tetapkan environment variable "
            "`EVISPAN_TTF_URL` dengan URL Google Forms atau platform survei yang digunakan."
        )

    try:
        export_df = session_responses_dataframe(RESPONSE_DB_PATH, session_id)
        st.download_button(
            "Download respons per komentar",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"evispan_comment_responses_{st.session_state.respondent_code}.csv",
            mime="text/csv",
        )
        final_df = session_final_dataframe(RESPONSE_DB_PATH, session_id)
        st.download_button(
            "Download hasil interpretasi dan pola",
            data=final_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"evispan_final_response_{st.session_state.respondent_code}.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Salinan respons tidak dapat dibuat: {exc}")
    st.stop()

if st.session_state.get("study_phase") == "final":
    render_workflow("final", completed_count, review_target)
    st.subheader("Tugas interpretasi dan identifikasi pola")
    st.markdown(
        "Setelah menelaah seluruh komentar, rangkum pola yang Anda temukan pada prediksi label, "
        "probabilitas, dan evidence span EviSpan-PR."
    )

    with st.form("final_interpretation_form"):
        easiest_label = st.selectbox(
            "Label yang paling mudah diinterpretasikan *",
            options=LABELS,
        )
        hardest_label = st.selectbox(
            "Label yang paling sulit atau ambigu *",
            options=LABELS,
            index=LABELS.index("Neutral") if "Neutral" in LABELS else 0,
        )
        recurring_patterns = st.text_area(
            "Pola umum apa yang Anda identifikasi dari komentar dan hasil model? *",
            placeholder=(
                "Contoh: Appreciation cenderung memiliki ungkapan positif yang eksplisit, "
                "sedangkan Suggestion sering muncul setelah pernyataan Problem."
            ),
            height=130,
        )
        recurring_mismatches = st.text_area(
            "Ketidaksesuaian atau pola kesalahan apa yang berulang?",
            placeholder="Jelaskan label, probabilitas, atau batas evidence span yang sering kurang tepat.",
            height=110,
        )
        usefulness_rating = st.slider(
            "Seberapa membantu label, probabilitas, dan evidence span untuk menelaah peer feedback? *",
            min_value=1,
            max_value=5,
            value=4,
            help="1 = sangat tidak membantu; 5 = sangat membantu.",
        )
        intended_use = st.text_area(
            "Bagaimana EviSpan-PR dapat digunakan dalam kegiatan pembelajaran atau monitoring dosen?",
            height=100,
        )
        overall_notes = st.text_area("Catatan tambahan", height=90)
        finish = st.form_submit_button("Simpan dan selesaikan sesi", type="primary")

    if finish:
        if not recurring_patterns.strip():
            st.error("Ringkasan pola umum wajib diisi.")
        else:
            final_payload = {
                "session_id": session_id,
                "easiest_label": easiest_label,
                "hardest_label": hardest_label,
                "recurring_patterns": recurring_patterns.strip(),
                "recurring_mismatches": recurring_mismatches.strip(),
                "usefulness_rating": int(usefulness_rating),
                "intended_use": intended_use.strip(),
                "overall_notes": overall_notes.strip(),
                "submitted_at": utc_now_iso(),
            }
            try:
                save_final_response(RESPONSE_DB_PATH, final_payload)
            except Exception as exc:
                st.error(f"Tugas akhir tidak dapat disimpan: {exc}")
                st.stop()
            st.session_state.final_response = final_payload
            st.session_state.study_phase = "complete"
            st.rerun()
    st.stop()

# Review phase: run inference and collect a response for each selected comment.
st.session_state.study_phase = "review"
if "review_position" not in st.session_state:
    st.session_state.review_position = 0
_set_review_position(int(st.session_state.review_position), review_target)

st.subheader("Telaah komentar peer feedback")
st.markdown(
    "Periksa teks, label prediksi, probabilitas, dan evidence span. Setelah itu, simpan "
    "interpretasi Anda untuk komentar yang sedang ditampilkan."
)
render_review_navigation(review_target, "top")
st.divider()

review_position = int(st.session_state.review_position)
dataset_row_position = int(review_indices[review_position])
row = working_df.iloc[dataset_row_position]
source_row_index = int(row["source_row_index"])
text = str(row["text"]).strip()
row_id = str(row["row_id"])

result = {
    "text": text,
    "predicted_labels": list(row["predicted_labels"]),
    "label_probabilities": dict(row["label_probabilities"]),
    "predicted_spans": list(row["predicted_spans"]),
    "truncated": bool(row.get("truncated", False)),
}
predicted_labels = list(result["predicted_labels"])
predicted_spans = list(result["predicted_spans"])

meta_1, meta_2, meta_3, meta_4 = st.columns(4)
meta_1.metric("Urutan telaah", f"{review_position + 1}/{review_target}")
meta_2.metric("ID data", str(row_id))
meta_3.metric("Jumlah label", len(predicted_labels))
meta_4.metric("Evidence span", len(predicted_spans))

st.markdown('<div class="section-label">Teks peer feedback</div>', unsafe_allow_html=True)
st.markdown(render_plain_text(text), unsafe_allow_html=True)

label_col, prob_col = st.columns([1, 1])
with label_col:
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Prediksi label</div>', unsafe_allow_html=True)
    st.markdown(label_badges(predicted_labels), unsafe_allow_html=True)
with prob_col:
    st.markdown('<div class="section-label" style="margin-top:1.2rem">Probabilitas label</div>', unsafe_allow_html=True)
    st.dataframe(
        probabilities_dataframe(result),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f"),
            "Predicted": st.column_config.CheckboxColumn(),
        },
    )

st.markdown('<div class="section-label" style="margin-top:1.2rem">Evidence span hasil prediksi</div>', unsafe_allow_html=True)
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
    st.warning("Model tidak menghasilkan evidence span untuk komentar ini.")

if result.get("truncated"):
    st.info("Teks melebihi panjang maksimum model dan dipotong saat inferensi.")

response_key = str(source_row_index)
existing = responses.get(response_key, {})
label_assessment_options = [
    "Sesuai seluruhnya",
    "Sebagian sesuai",
    "Tidak sesuai",
    "Tidak dapat menilai",
]
evidence_support_options = [
    "Seluruh span mendukung label",
    "Sebagian span mendukung label",
    "Span tidak mendukung label",
    "Tidak ada evidence span",
    "Tidak dapat menilai",
]
evidence_boundary_options = [
    "Batas span tepat",
    "Batas span sebagian tepat",
    "Batas span tidak tepat",
    "Tidak ada evidence span",
    "Tidak dapat menilai",
]
pattern_options = [
    "Pujian atau evaluasi positif eksplisit",
    "Identifikasi kekurangan atau masalah",
    "Rekomendasi tindakan yang konkret",
    "Informasi atau deskripsi netral",
    "Gabungan beberapa fungsi feedback",
    "Label atau evidence span ambigu",
]

def option_index(options: List[str], value: Any, default: int = 0) -> int:
    return options.index(value) if value in options else default

st.divider()
st.subheader("Respons dosen untuk komentar ini")
with st.form(f"comment_response_form_{source_row_index}"):
    label_assessment = st.radio(
        "Apakah label prediksi sesuai dengan isi komentar? *",
        options=label_assessment_options,
        index=option_index(label_assessment_options, existing.get("label_assessment"), 0),
        horizontal=True,
    )
    expected_labels = st.multiselect(
        "Menurut Anda, label apa yang seharusnya diberikan? *",
        options=LABELS,
        default=existing.get("expected_labels", predicted_labels),
        help="Boleh memilih lebih dari satu label. Kosongkan hanya bila komentar tidak sesuai dengan semua kategori.",
    )
    evidence_support = st.radio(
        "Apakah evidence span benar-benar mendukung label prediksi? *",
        options=evidence_support_options,
        index=option_index(
            evidence_support_options,
            existing.get("evidence_support"),
            3 if not predicted_spans else 0,
        ),
    )
    evidence_boundary = st.radio(
        "Bagaimana ketepatan batas evidence span? *",
        options=evidence_boundary_options,
        index=option_index(
            evidence_boundary_options,
            existing.get("evidence_boundary"),
            3 if not predicted_spans else 0,
        ),
    )
    identified_patterns = st.multiselect(
        "Pola apa yang tampak pada komentar ini?",
        options=pattern_options,
        default=existing.get("identified_patterns", []),
    )
    interpretation = st.text_area(
        "Jelaskan interpretasi Anda secara singkat *",
        value=existing.get("interpretation", ""),
        placeholder="Jelaskan alasan kesesuaian atau ketidaksesuaian label dan evidence span.",
        height=105,
    )
    confidence = st.slider(
        "Tingkat keyakinan terhadap penilaian Anda *",
        min_value=1,
        max_value=5,
        value=int(existing.get("confidence", 4)),
        help="1 = sangat tidak yakin; 5 = sangat yakin.",
    )
    notes = st.text_area(
        "Catatan tambahan",
        value=existing.get("notes", ""),
        height=75,
    )
    save_only, save_next = st.columns(2)
    save_response = save_only.form_submit_button("Simpan respons", use_container_width=True)
    save_and_next = save_next.form_submit_button("Simpan & lanjut", type="primary", use_container_width=True)

if save_response or save_and_next:
    if not interpretation.strip():
        st.error("Interpretasi singkat wajib diisi sebelum respons disimpan.")
    else:
        response_payload = {
            "session_id": session_id,
            "source_row_index": source_row_index,
            "review_order": review_position + 1,
            "row_id": str(row_id),
            "comment_text": text,
            "predicted_labels": predicted_labels,
            "label_probabilities": result.get("label_probabilities", {}),
            "predicted_spans": predicted_spans,
            "label_assessment": label_assessment,
            "expected_labels": expected_labels,
            "evidence_support": evidence_support,
            "evidence_boundary": evidence_boundary,
            "identified_patterns": identified_patterns,
            "interpretation": interpretation.strip(),
            "confidence": int(confidence),
            "notes": notes.strip(),
            "reviewed_at": utc_now_iso(),
        }
        try:
            save_comment_response(RESPONSE_DB_PATH, response_payload)
        except Exception as exc:
            st.error(f"Respons tidak dapat disimpan: {exc}")
            st.stop()

        st.session_state.study_responses[response_key] = response_payload
        st.success("Respons dosen telah disimpan.")
        if save_and_next and review_position < review_target - 1:
            _set_review_position(review_position + 1, review_target)
        st.rerun()

if response_key in responses:
    st.success("Respons untuk komentar ini sudah tersimpan dan dapat diperbarui.")

render_review_navigation(review_target, "bottom")
completed_count = len(st.session_state.study_responses)
if completed_count == review_target:
    st.success("Seluruh respons komentar telah tersimpan.")
    if st.button("Lanjut ke tugas interpretasi dan identifikasi pola", type="primary", use_container_width=True):
        st.session_state.study_phase = "final"
        st.rerun()
else:
    remaining = review_target - completed_count
    st.info(f"Masih ada {remaining} komentar yang belum memiliki respons tersimpan.")
