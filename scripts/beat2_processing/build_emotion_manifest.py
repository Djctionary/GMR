import argparse
import csv
import json
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EMOTION_RANGES = (
    ("neutral", 0, 64),
    ("happiness", 65, 72),
    ("anger", 73, 80),
    ("sadness", 81, 86),
    ("contempt", 87, 94),
    ("surprise", 95, 102),
    ("fear", 103, 110),
    ("disgust", 111, 118),
)

EMOTIONS = tuple(name for name, _, _ in EMOTION_RANGES)


@dataclass(frozen=True)
class ClipId:
    clip_id: str
    speaker_id: str
    speaker_name: str
    recording_type: str
    start_id: int
    end_id: int


def parse_clip_id(filename: str) -> ClipId:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Invalid BEAT2 clip name: {filename}")

    speaker_id = parts[0]
    speaker_name = "_".join(parts[1:-3])
    recording_type = parts[-3]
    try:
        start_id = int(parts[-2])
        end_id = int(parts[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid BEAT2 question ids in: {filename}") from exc

    return ClipId(
        clip_id=stem,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        recording_type=recording_type,
        start_id=start_id,
        end_id=end_id,
    )


def clip_id_to_emotion(filename: str) -> str:
    clip = parse_clip_id(filename)
    if clip.recording_type != "0":
        raise ValueError(
            f"{filename} is not an English Speech clip. "
            "BEAT conversation clips are officially neutral, but are excluded here."
        )

    if clip.start_id != clip.end_id:
        raise ValueError(f"Expected a single-question clip, got: {filename}")

    for emotion, start, end in EMOTION_RANGES:
        if start <= clip.start_id <= end:
            return emotion

    raise ValueError(f"Question id is outside the BEAT emotion protocol: {filename}")


def read_wav_duration_sec(wav_path: Path) -> float | None:
    if not wav_path.exists():
        return None

    with wav_path.open("rb") as file:
        header = file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None

        sample_rate = None
        block_align = None
        data_size = None
        while True:
            chunk_header = file.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)

            if chunk_id == b"fmt ":
                fmt_data = file.read(chunk_size)
                if len(fmt_data) >= 16:
                    sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                    block_align = struct.unpack("<H", fmt_data[12:14])[0]
            elif chunk_id == b"data":
                data_size = chunk_size
                file.seek(chunk_size, 1)
            else:
                file.seek(chunk_size, 1)

            if chunk_size % 2:
                file.seek(1, 1)

        if not sample_rate or not block_align or data_size is None:
            return None
        return data_size / block_align / sample_rate


def inspect_npz(npz_path: Path, max_trans_drift_m: float) -> tuple[dict, list[str]]:
    problems = []
    with np.load(npz_path, allow_pickle=True) as data:
        keys = set(data.files)
        required = {"poses", "trans", "mocap_frame_rate"}
        missing = sorted(required - keys)
        if missing:
            return {"num_frames": 0, "fps": None, "duration_sec": 0.0, "trans_drift_m": None}, [
                f"missing_keys:{','.join(missing)}"
            ]

        poses = data["poses"]
        trans = data["trans"]
        fps_raw = data["mocap_frame_rate"]
        fps = float(fps_raw.item() if getattr(fps_raw, "shape", None) == () else fps_raw)

        if poses.ndim != 2 or poses.shape[1] < 66:
            problems.append(f"bad_poses_shape:{poses.shape}")
        if trans.ndim != 2 or trans.shape[1] != 3:
            problems.append(f"bad_trans_shape:{trans.shape}")
        if poses.shape[0] != trans.shape[0]:
            problems.append(f"frame_mismatch:poses={poses.shape[0]},trans={trans.shape[0]}")
        if fps <= 0:
            problems.append(f"bad_fps:{fps}")
            duration_sec = 0.0
        else:
            duration_sec = poses.shape[0] / fps

        arrays_to_check = {"poses": poses, "trans": trans}
        if "expressions" in data:
            arrays_to_check["expressions"] = data["expressions"]
        for key, value in arrays_to_check.items():
            if not np.isfinite(value).all():
                problems.append(f"non_finite:{key}")

        if trans.ndim == 2 and trans.shape[1] == 3 and trans.shape[0] > 0:
            drift = float(np.linalg.norm(trans - trans[0], axis=1).max())
            if drift > max_trans_drift_m:
                problems.append(f"large_trans_drift:{drift:.3f}m")
        else:
            drift = None

    return {
        "num_frames": int(poses.shape[0]),
        "fps": fps,
        "duration_sec": float(duration_sec),
        "trans_drift_m": drift,
    }, problems


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(args: argparse.Namespace) -> None:
    beat2_root = Path(args.beat2_root).expanduser().resolve()
    english_root = beat2_root / "beat_english_v2.0.0"
    npz_dir = english_root / "smplxflame_30"
    wav_dir = english_root / "wave16k"
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not npz_dir.is_dir():
        raise FileNotFoundError(f"Missing BEAT2 English SMPL-X directory: {npz_dir}")

    manifest_rows = []
    problematic = {}
    all_speech_count = 0

    for npz_path in sorted(npz_dir.glob("*.npz")):
        clip = parse_clip_id(npz_path.name)
        if clip.recording_type != "0":
            continue

        all_speech_count += 1
        try:
            emotion = clip_id_to_emotion(npz_path.name)
        except ValueError as exc:
            problematic[clip.clip_id] = [str(exc)]
            continue

        audio_path = wav_dir / f"{clip.clip_id}.wav"
        npz_info, problems = inspect_npz(npz_path, args.max_trans_drift_m)
        audio_duration_sec = read_wav_duration_sec(audio_path)
        has_audio = audio_duration_sec is not None
        if not has_audio:
            problems.append("missing_audio")

        if npz_info["num_frames"] < args.min_frames:
            problems.append(f"too_short:{npz_info['num_frames']}frames")

        if problems:
            problematic[clip.clip_id] = problems
            continue

        manifest_rows.append(
            {
                "clip_id": clip.clip_id,
                "speaker_id": clip.speaker_id,
                "speaker_name": clip.speaker_name,
                "emotion": emotion,
                "num_frames": npz_info["num_frames"],
                "duration_sec": f"{npz_info['duration_sec']:.3f}",
                "npz_filename": npz_path.name,
                "audio_filename": audio_path.name,
                "has_audio": str(has_audio),
                "audio_duration_sec": f"{audio_duration_sec:.3f}",
                "trans_drift_m": f"{npz_info['trans_drift_m']:.6f}",
            }
        )

    manifest_path = output_dir / "beat2_emotion_manifest.csv"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "clip_id",
            "speaker_id",
            "speaker_name",
            "emotion",
            "num_frames",
            "duration_sec",
            "npz_filename",
            "audio_filename",
            "has_audio",
            "audio_duration_sec",
            "trans_drift_m",
        ],
    )

    group_rows = []
    rows_by_emotion = defaultdict(list)
    speaker_distribution = defaultdict(Counter)
    for row in manifest_rows:
        rows_by_emotion[row["emotion"]].append(row)
        speaker_distribution[row["emotion"]][row["speaker_id"]] += 1

    for emotion in EMOTIONS:
        rows = rows_by_emotion[emotion]
        total_duration = sum(float(row["duration_sec"]) for row in rows)
        group_rows.append(
            {
                "emotion": emotion,
                "clip_count": len(rows),
                "total_duration_sec": f"{total_duration:.3f}",
                "avg_duration_sec": f"{(total_duration / len(rows)):.3f}" if rows else "0.000",
                "speaker_count": len(speaker_distribution[emotion]),
            }
        )

    write_csv(
        output_dir / "beat2_emotion_group_stats.csv",
        group_rows,
        ["emotion", "clip_count", "total_duration_sec", "avg_duration_sec", "speaker_count"],
    )

    speaker_rows = []
    for emotion in EMOTIONS:
        for speaker_id, count in sorted(
            speaker_distribution[emotion].items(), key=lambda item: int(item[0])
        ):
            speaker_rows.append({"emotion": emotion, "speaker_id": speaker_id, "clip_count": count})
    write_csv(
        output_dir / "beat2_emotion_speaker_distribution.csv",
        speaker_rows,
        ["emotion", "speaker_id", "clip_count"],
    )

    spot_check_rows = []
    for emotion in EMOTIONS:
        rows = sorted(rows_by_emotion[emotion], key=lambda row: row["clip_id"])[: args.spot_check_per_emotion]
        for row in rows:
            spot_check_rows.append(row)
    write_csv(
        output_dir / "beat2_emotion_spot_check_samples.csv",
        spot_check_rows,
        [
            "clip_id",
            "speaker_id",
            "speaker_name",
            "emotion",
            "num_frames",
            "duration_sec",
            "npz_filename",
            "audio_filename",
            "has_audio",
            "audio_duration_sec",
            "trans_drift_m",
        ],
    )

    problem_path = output_dir / "beat2_emotion_problematic_clips.json"
    problem_path.write_text(json.dumps(problematic, indent=2, sort_keys=True), encoding="utf-8")

    print(f"English speech clips scanned: {all_speech_count}")
    print(f"Manifest clips written: {len(manifest_rows)}")
    print(f"Problematic clips: {len(problematic)}")
    print(f"Manifest: {manifest_path}")
    print(f"Group stats: {output_dir / 'beat2_emotion_group_stats.csv'}")
    print(f"Speaker distribution: {output_dir / 'beat2_emotion_speaker_distribution.csv'}")
    print(f"Spot-check samples: {output_dir / 'beat2_emotion_spot_check_samples.csv'}")
    print(f"Problematic clips: {problem_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an English BEAT2 emotion manifest for downstream retargeting."
    )
    parser.add_argument("--beat2_root", default="/home/vergil/dataset/BEAT2")
    parser.add_argument("--output_dir", default="motion_data/BEAT2/manifests")
    parser.add_argument("--min_frames", type=int, default=150)
    parser.add_argument("--max_trans_drift_m", type=float, default=5.0)
    parser.add_argument("--spot_check_per_emotion", type=int, default=3)
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
