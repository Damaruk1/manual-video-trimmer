# Manual Video Trimmer (Face-Aware)

## Description
CLI-based video trimmer that extracts a specific time segment, converts it to vertical 9:16 format, and optionally generates subtitles.

## Features
- YouTube or local video input
- Start–end time trimming
- Face-priority smart crop using OpenCV
- Automatic SRT generation
- FFmpeg encoding presets (high / fast)

## Requirements
FFmpeg installed and added to PATH.

## How to Run
python model2_trimmer.py --video input.mp4 --start 10 --end 40 --srt true