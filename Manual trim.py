#!/usr/bin/env python3
"""
Model 2 - Manual Trimmer (single-file)

Install prerequisites:
pip install yt-dlp youtube-transcript-api requests opencv-python-headless numpy

Requires ffmpeg on PATH.

Example usage:
python model2_trimmer.py --url "https://www.youtube.com/watch?v=XXXX" --start 30 --end 75 --out outputs/model2 --srt true

Or with local file:
python model2_trimmer.py --video /path/to/input.mp4 --start 10 --end 40 --out outputs/model2 --srt true --whisper false

Options:
--url URL of YouTube video
--video local path to video (mutually exclusive with --url)
--start start time in seconds (float)
--end end time in seconds (float)
--out output directory
--srt true/false to create SRT (uses YouTube captions by default, Whisper if --whisper true)
--whisper true/false to run local whisper transcription when available
--face true/false to prioritize face-centering when cropping
--quality choose "high" or "fast" encoding presets
"""

import os
import re
import sys
import json
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

def which(bin_name: str) -> Optional[str]:
    from shutil import which as _which
    return _which(bin_name)

def run_cmd(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc

def ffprobe_duration(path: str) -> float:
    if not which("ffprobe"):
        return 0.0
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except Exception:
        return 0.0

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    if "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    cleaned = re.sub(r"[^A-Za-z0-9\-_]", "", url)
    return (cleaned[:11] or "video")

def download_youtube(url: str, out_path: str) -> str:
    out_dir = str(Path(out_path).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    template = out_path
    cmd = ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best", "-o", template, url]
    try:
        run_cmd(cmd)
    except Exception as e:
        raise RuntimeError("yt-dlp failed to download video. Ensure yt-dlp is installed and URL is valid.") from e
    if not Path(out_path).exists():
        raise RuntimeError("Downloaded video missing after yt-dlp run")
    return out_path

def fetch_youtube_vtt(url: str, outdir: str) -> Optional[str]:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    template = str(p / "%(id)s")
    cmd = [
        "yt-dlp", "--skip-download", "--no-warnings", "--no-playlist",
        "--write-auto-sub", "--write-subs", "--sub-langs", "en.*,en,en-US,en-IN,hi",
        "--sub-format", "vtt", "--convert-subs", "vtt", "-o", template, url
    ]
    try:
        run_cmd(cmd, check=False)
    except Exception:
        pass
    vtts = list(p.glob("*.vtt"))
    return str(vtts[0]) if vtts else None

def parse_vtt_to_segments(vtt_path: str) -> List[dict]:
    text = Path(vtt_path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if "-->" in b]
    segs = []
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        times = None
        texts = []
        for ln in lines:
            if "-->" in ln:
                times = ln.strip()
            elif not ln.strip().isdigit():
                texts.append(ln.strip())
        if not times:
            continue
        start_s, end_s = [t.strip() for t in times.split("-->")]
        def t2s(x: str) -> float:
            parts = x.split(":")
            if len(parts) == 3:
                h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
                return h * 3600 + m * 60 + s
            return 0.0
        segs.append({"start": t2s(start_s), "end": t2s(end_s), "text": " ".join(texts)})
    return segs

def generate_srt_from_segments(segs: List[dict], out_srt: str) -> str:
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, start=1):
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + s.get("duration", 1.0)))
            def fmt(t: float) -> str:
                h = int(t // 3600); m = int((t % 3600) // 60); s_sec = int(t % 60); ms = int((t - int(t)) * 1000)
                return f"{h:02}:{m:02}:{s_sec:02},{ms:03}"
            text = (s.get("text") or "").strip()
            f.write(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
    return out_srt

def whisper_transcribe(video_path: str, out_srt: str) -> str:
    try:
        import whisper
    except Exception:
        raise RuntimeError("Whisper requested but not installed. pip install -U openai-whisper or set --whisper false")
    model = whisper.load_model("base")
    res = model.transcribe(video_path, verbose=False)
    segments = res.get("segments", [])
    segs = []
    for seg in segments:
        segs.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()})
    return generate_srt_from_segments(segs, out_srt)

def detect_face_box(video_path: str, sec: float = 0.5) -> Optional[tuple]:
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = int(max(0.0, sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    largest = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
    x, y, w, h = largest
    return (int(x), int(y), int(x + w), int(y + h))

def build_crop_filter_9_16(video_file: str, face_box: Optional[tuple] = None) -> str:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", video_file],
                           capture_output=True, text=True)
    try:
        w, h = [int(x) for x in probe.stdout.strip().split(",")]
    except Exception:
        w, h = 1280, 720
    target_aspect = 9.0 / 16.0
    cur_aspect = float(w) / float(h)
    if cur_aspect > target_aspect:
        new_h = h
        new_w = int(h * target_aspect)
    else:
        new_w = w
        new_h = int(w / target_aspect)
    if face_box:
        fx1, fy1, fx2, fy2 = face_box
        face_cx = (fx1 + fx2) // 2
        face_cy = (fy1 + fy2) // 2
        x = max(0, int(face_cx - new_w // 2))
        y = max(0, int(face_cy - new_h // 2))
        x = min(max(0, x), max(0, w - new_w))
        y = min(max(0, y), max(0, h - new_h))
    else:
        x = max(0, (w - new_w) // 2)
        y = max(0, (h - new_h) // 2)
    return f"crop={new_w}:{new_h}:{x}:{y},scale=1080:1920"

def trim_and_crop(input_file: str, start: float, end: float, out_file: str, prioritize_face: bool = True, quality: str = "high") -> str:
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required and not found on PATH")
    tmpdir = tempfile.mkdtemp(prefix="trim_")
    try:
        tmp_clip = str(Path(tmpdir) / "clip_part.mp4")
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", input_file, "-c", "copy", tmp_clip]
        try:
            run_cmd(cmd, check=True)
        except Exception:
            cmd2 = ["ffmpeg", "-y", "-i", input_file, "-ss", str(start), "-to", str(end), "-c", "copy", tmp_clip]
            run_cmd(cmd2, check=True)
        face_box = None
        if prioritize_face:
            try:
                face_box = detect_face_box(tmp_clip, sec=max(0.2, (end - start) / 4.0))
            except Exception:
                face_box = None
        vf = build_crop_filter_9_16(tmp_clip, face_box)
        preset = "slow" if quality == "high" else "fast"
        cmd_crop = [
            "ffmpeg", "-y", "-i", tmp_clip, "-vf", vf, "-c:v", "libx264", "-crf", "18",
            "-preset", preset, "-c:a", "aac", "-movflags", "+faststart", out_file
        ]
        run_cmd(cmd_crop, check=True)
        return out_file
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def create_srt_for_trim(input_video: str, start: float, end: float, out_srt: str, use_whisper: bool = False) -> str:
    tmpdir = tempfile.mkdtemp(prefix="srt_")
    try:
        clip_tmp = str(Path(tmpdir) / "clip_for_srt.mp4")
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", input_video, "-c", "copy", clip_tmp]
        run_cmd(cmd, check=True)
        if use_whisper:
            try:
                srt = whisper_transcribe(clip_tmp, out_srt)
                return srt
            except Exception:
                pass
        vtt = fetch_youtube_vtt(input_video if input_video.startswith("http") else "", tmpdir) if input_video.startswith("http") else None
        if vtt and Path(vtt).exists():
            segs = parse_vtt_to_segments(vtt)
            adjusted = []
            for s in segs:
                if s["end"] <= start or s["start"] >= end:
                    continue
                new_start = max(0.0, s["start"] - start)
                new_end = max(0.01, s["end"] - start)
                adjusted.append({"start": new_start, "end": new_end, "text": s["text"]})
            if adjusted:
                return generate_srt_from_segments(adjusted, out_srt)
        try:
            srt = whisper_transcribe(clip_tmp, out_srt)
            return srt
        except Exception:
            words = "Automatically trimmed clip."
            with open(out_srt, "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:05,000\n" + words + "\n")
            return out_srt
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", help="YouTube URL")
    p.add_argument("--video", help="Local video file path")
    p.add_argument("--start", type=float, required=True, help="Start time in seconds")
    p.add_argument("--end", type=float, required=True, help="End time in seconds")
    p.add_argument("--out", default="outputs/model2", help="Output directory")
    p.add_argument("--srt", default="false", help="Generate SRT (true/false)")
    p.add_argument("--whisper", default="false", help="Use local Whisper for transcription if available (true/false)")
    p.add_argument("--face", default="true", help="Face-priority crop (true/false)")
    p.add_argument("--quality", default="high", choices=["high", "fast"], help="Encoding quality preset")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out)
    input_src = None
    tmpdir = tempfile.mkdtemp(prefix="model2_")
    try:
        if args.url and args.video:
            raise RuntimeError("Provide either --url or --video, not both")
        if args.url:
            vid = extract_video_id(args.url)
            target_path = str(Path(tmpdir) / f"{vid}.mp4")
            try:
                download_youtube(args.url, target_path)
                input_src = target_path
            except Exception as e:
                if Path(args.url).exists():
                    input_src = args.url
                else:
                    raise
        elif args.video:
            if not Path(args.video).exists():
                raise RuntimeError("Local video file not found")
            input_src = args.video
        else:
            raise RuntimeError("Either --url or --video must be provided")
        start = float(args.start); end = float(args.end)
        if end <= start:
            raise RuntimeError("End time must be greater than start time")
        out_file = str(Path(args.out) / f"trim_{int(start)}_{int(end)}.mp4")
        prioritized_face = True if str(args.face).lower() in ("true", "1", "yes") else False
        final = trim_and_crop(input_src, start, end, out_file, prioritize_face=prioritized_face, quality=args.quality)
        srt_path = None
        if str(args.srt).lower() in ("true", "1", "yes"):
            out_srt = str(Path(args.out) / f"trim_{int(start)}_{int(end)}.srt")
            use_whisper = True if str(args.whisper).lower() in ("true", "1", "yes") else False
            try:
                srt_path = create_srt_for_trim(args.url if args.url else input_src, start, end, out_srt, use_whisper=use_whisper)
            except Exception:
                srt_path = None
        metadata = {"source": args.url if args.url else args.video, "start": start, "end": end, "out": final, "srt": srt_path}
        meta_path = str(Path(args.out) / "metadata.json")
        Path(meta_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Trim successful. Output:", final)
        if srt_path:
            print("SRT written to:", srt_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    main()
