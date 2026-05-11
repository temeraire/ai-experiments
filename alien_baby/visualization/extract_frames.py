"""
extract_frames.py — pull PNGs from a video for visual diagnostics.

Two modes (combinable):
  --even N        N evenly-spaced frames across the whole video. Use for
                  whole-episode overview ("does anything happen?").
  --burst T N FPS Dense burst: N frames at FPS hz starting at time T (seconds).
                  Use for jitter detection. E.g. --burst 3.0 10 10 = 10 frames
                  spanning 1 second around t=3s.
                  Repeatable: --burst 2 10 10 --burst 8 10 10 for two bursts.

Output: PNGs in --outdir, sorted by filename. Existing PNGs in --outdir are
removed first so the read-back is unambiguous.

Usage:
  python -m alien_baby.visualization.extract_frames video.mp4 --even 12
  python -m alien_baby.visualization.extract_frames video.mp4 --burst 5.0 10 10
  python -m alien_baby.visualization.extract_frames video.mp4 \\
      --even 6 --burst 2 10 10 --burst 8 10 10 --outdir /tmp/frames
"""
import argparse
import pathlib
import subprocess
import sys


def _probe_duration(video: pathlib.Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]).decode().strip()
    return float(out)


def _grab_frame(video: pathlib.Path, t: float, out_path: pathlib.Path) -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{t:.4f}",
        "-i", str(video),
        "-frames:v", "1",
        "-loglevel", "error",
        str(out_path),
    ], check=True)


def extract_even(video, n, outdir, duration):
    # Sample at duration * i / (n+1) for i=1..n — skips frame 0 and the very last frame.
    paths = []
    for i in range(1, n + 1):
        t = duration * i / (n + 1)
        out = outdir / f"even_{i:02d}_t{t:06.2f}s.png"
        _grab_frame(video, t, out)
        paths.append(out)
    return paths


def extract_burst(video, start, count, fps, outdir):
    dt = 1.0 / fps
    paths = []
    for i in range(count):
        t = start + i * dt
        out = outdir / f"burst_t{start:05.2f}_{i:02d}_t{t:07.3f}s.png"
        _grab_frame(video, t, out)
        paths.append(out)
    return paths


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", type=pathlib.Path)
    p.add_argument("--even", type=int, default=0,
                   help="N evenly-spaced frames across the video.")
    p.add_argument("--burst", nargs=3, action="append", metavar=("T", "N", "FPS"),
                   help="Burst: N frames at FPS hz starting at time T (seconds). "
                        "Repeatable.")
    p.add_argument("--outdir", type=pathlib.Path,
                   default=pathlib.Path("/tmp/ab_frames"))
    args = p.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")
    if args.even == 0 and not args.burst:
        sys.exit("Specify --even N and/or --burst T N FPS")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for old in args.outdir.glob("*.png"):
        old.unlink()

    duration = _probe_duration(args.video)
    print(f"Video: {args.video}")
    print(f"  duration : {duration:.2f}s")
    print(f"  outdir   : {args.outdir}")

    written = []
    if args.even:
        written += extract_even(args.video, args.even, args.outdir, duration)
    if args.burst:
        for spec in args.burst:
            t, n, fps = float(spec[0]), int(spec[1]), float(spec[2])
            if t < 0 or t + (n - 1) / fps > duration:
                print(f"  WARN: burst T={t}s + {n}frames @ {fps}fps "
                      f"extends past duration {duration:.2f}s")
            written += extract_burst(args.video, t, n, fps, args.outdir)

    print(f"\nWrote {len(written)} frames:")
    for path in sorted(written):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
