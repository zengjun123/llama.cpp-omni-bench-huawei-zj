#!/usr/bin/env python3
"""Convert the ModelScope ``MTEB/Daily-Omni`` parquet release into the official
``qa.json`` + ``Videos/`` layout consumed by ``vllm bench serve --dataset-name daily-omni``.

Source layout (``modelscope download --dataset MTEB/Daily-Omni --local_dir SRC``)::

    SRC/data/test-0000{0..9}-of-00010.parquet

with columns ``video_id``, ``video{bytes,path}``, ``audio{bytes,path}``, ``question``,
``candidates`` (list<string>), ``answer``.

Target layout::

    DST/qa.json
    DST/Videos/{video_id}/{video_id}_video.mp4
    DST/Videos/{video_id}/{video_id}_audio.wav

Three shape mismatches this script fixes:

1. ``answer`` holds the full option text ("D. Tax laws"), but the bench's
   ``evaluate_answer_official`` compares the model's bare letter against ``Answer``
   with a strict string match. Copying the text through scores 0%, so the leading
   letter is extracted.
2. The embedded WAV is named ``{video_id}_video.wav``; the bench resolves
   ``{video_id}_audio.wav``.
3. ``Type`` / ``video_category`` do not exist upstream of this release, so the
   per-task-type and per-category accuracy breakdowns degrade to ``unknown`` unless
   ``--official-qa`` supplies them. ``video_duration`` can instead be recovered from the
   media with ``--probe-duration``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Official qa.json only ever uses these two buckets.
_DURATION_BUCKETS = (30.0, 60.0)
_LETTER_RE = re.compile(r"^\s*\(?([A-D])\)?\s*[.、:：)\-]")

def _answer_letter(answer: str, candidates: list[str] | None) -> str:
    """Reduce a ModelScope ``answer`` to the official single-letter ``Answer``."""
    a = (answer or "").strip()
    if len(a) == 1 and a.upper() in "ABCD":
        return a.upper()
    m = _LETTER_RE.match(a)
    if m:
        return m.group(1)
    # Some rows repeat the option verbatim without its prefix; locate it in the choices.
    for c in candidates or []:
        if c.strip() == a:
            m = _LETTER_RE.match(c.strip())
            if m:
                return m.group(1)
    m = re.search(r"\b([A-D])\b", a.upper())
    return m.group(1) if m else ""

def _probe_duration_bucket(video_path: Path) -> str:
    """Snap the real MP4 duration to the nearest official ``30s`` / ``60s`` bucket."""
    try:
        import av
    except ImportError:
        return ""
    try:
        with av.open(str(video_path)) as container:
            duration = None
            if container.duration is not None:
                duration = container.duration / av.time_base
            else:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
    except Exception:
        return ""
    if not duration:
        return ""
    return f"{int(min(_DURATION_BUCKETS, key=lambda b: abs(b - duration)))}s"

def _load_official_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Index the official qa.json by ``(video_id, question)`` to re-attach lost metadata."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("video_id", "")).strip(), str(row.get("Question", "")).strip())
        index[key] = row
    return index

def _blob(cell: Any, field: str) -> bytes | None:
    """Read ``bytes`` out of a struct cell that pyarrow decoded into a dict."""
    if not isinstance(cell, dict):
        return None
    value = cell.get(field)
    return value if isinstance(value, (bytes, bytearray)) else None

def convert(
    src: Path,
    dst: Path,
    official_qa: Path | None,
    probe_duration: bool,
    batch_size: int,
) -> None:
    import pyarrow.parquet as pq

    shards = sorted(src.glob("data/*.parquet")) or sorted(src.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards found under {src} (expected data/*.parquet)")

    official = _load_official_index(official_qa) if official_qa else {}
    if official_qa:
        print(f"Loaded {len(official)} official rows for metadata merge", file=sys.stderr)

    videos_root = dst / "Videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    qa_rows: list[dict[str, Any]] = []
    seen_media: set[str] = set()
    missing_letter = 0
    missing_meta = 0

    for shard in shards:
        print(f"[{shard.name}] reading", file=sys.stderr)
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                video_id = str(row.get("video_id") or "").strip()
                if not video_id:
                    continue

                # 1196 QA rows map onto 684 videos, so the media repeats; write it once.
                if video_id not in seen_media:
                    out_dir = videos_root / video_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    mp4 = out_dir / f"{video_id}_video.mp4"
                    wav = out_dir / f"{video_id}_audio.wav"
                    video_bytes = _blob(row.get("video"), "bytes")
                    audio_bytes = _blob(row.get("audio"), "bytes")
                    if video_bytes and not mp4.exists():
                        mp4.write_bytes(video_bytes)
                    if audio_bytes and not wav.exists():
                        wav.write_bytes(audio_bytes)
                    seen_media.add(video_id)

                question = str(row.get("question") or "").strip()
                candidates = [str(c) for c in (row.get("candidates") or [])]
                letter = _answer_letter(str(row.get("answer") or ""), candidates)
                if not letter:
                    missing_letter += 1

                entry: dict[str, Any] = {
                    "Question": question,
                    "Choice": candidates,
                    "Answer": letter,
                    "video_id": video_id,
                    "Type": "",
                    "video_category": "",
                    "video_duration": "",
                }

                ref = official.get((video_id, question))
                if ref is not None:
                    entry["Type"] = ref.get("Type", "")
                    entry["video_category"] = ref.get("video_category", "")
                    entry["video_duration"] = ref.get("video_duration", "")
                    for extra in ("content_parent_category", "content_fine_category"):
                        if extra in ref:
                            entry[extra] = ref[extra]
                else:
                    missing_meta += 1

                qa_rows.append(entry)

        print(f"[{shard.name}] done — {len(qa_rows)} QA rows, {len(seen_media)} videos", file=sys.stderr)

    if probe_duration:
        cache: dict[str, str] = {}
        for entry in qa_rows:
            if entry["video_duration"]:
                continue
            vid = entry["video_id"]
            if vid not in cache:
                cache[vid] = _probe_duration_bucket(videos_root / vid / f"{vid}_video.mp4")
            entry["video_duration"] = cache[vid]

    qa_path = dst / "qa.json"
    qa_path.write_text(json.dumps(qa_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {qa_path} ({len(qa_rows)} rows)", file=sys.stderr)
    print(f"Wrote {videos_root} ({len(seen_media)} videos)", file=sys.stderr)
    if missing_letter:
        print(f"WARNING: {missing_letter} rows have no parseable A-D answer letter", file=sys.stderr)
    if official_qa and missing_meta:
        print(f"WARNING: {missing_meta} rows found no official metadata match", file=sys.stderr)
    if not official_qa:
        print(
            "NOTE: Type / video_category are empty (absent from the ModelScope release); "
            "per-task-type and per-category accuracy will report as 'unknown'. "
            "Pass --official-qa to restore them.",
            file=sys.stderr,
        )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="modelscope download --local_dir target")
    parser.add_argument("--dst", required=True, type=Path, help="Output root receiving qa.json and Videos/")
    parser.add_argument(
        "--official-qa",
        type=Path,
        default=None,
        help="Official liarliar/Daily-Omni qa.json used to restore Type / video_category / video_duration",
    )
    parser.add_argument(
        "--probe-duration",
        action="store_true",
        help="Derive video_duration (30s/60s) from the extracted MP4 when not supplied by --official-qa",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Parquet rows held in memory at once")
    args = parser.parse_args()

    convert(args.src, args.dst, args.official_qa, args.probe_duration, args.batch_size)

if __name__ == "__main__":
    main()
