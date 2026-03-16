#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import gs_utils


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    return data


def build_single_camera(cameras):
    single_camera = {}
    for key, value in cameras.items():
        if key == "camera_to_worlds":
            if not torch.is_tensor(value) or value.shape[0] == 0:
                raise ValueError("camera_to_worlds is missing or empty in scene payload")
            single_camera[key] = value[:1]
        else:
            single_camera[key] = value
    return single_camera


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a short morph video from input GS to output GS using linear interpolation."
    )
    parser.add_argument("--scene_pt", type=str, required=True, help="Path to residual scene .pt file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for frames and video")
    parser.add_argument("--num_frames", type=int, default=48, help="Number of interpolation frames")
    parser.add_argument("--fps", type=int, default=24, help="Output video FPS")
    parser.add_argument("--skip_video", action="store_true", help="Only save PNG frames and skip video encoding")
    return parser.parse_args()


def encode_video_with_ffmpeg(frames_dir, video_path, fps):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False, "ffmpeg not found"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = proc.returncode == 0 and video_path.exists() and video_path.stat().st_size > 0
    if ok:
        return True, "encoded with ffmpeg"
    reason = proc.stderr.strip().splitlines()[-1] if proc.stderr else f"ffmpeg exit code {proc.returncode}"
    return False, reason


def encode_video_with_cv2(frames, video_path, fps, codecs):
    if cv2 is None:
        return False, "OpenCV is not installed"
    if len(frames) == 0:
        return False, "no frames to encode"

    height, width = frames[0].shape[:2]
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            continue
        for frame in frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
        writer.release()
        if video_path.exists() and video_path.stat().st_size > 0:
            return True, f"encoded with OpenCV codec {codec}"
    return False, "all OpenCV codecs failed"


def main():
    args = parse_args()
    if args.num_frames < 2:
        raise ValueError("--num_frames must be >= 2")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    scene_pt = Path(args.scene_pt)
    if not scene_pt.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_pt}")

    payload = torch.load(scene_pt, map_location="cpu")
    required_fields = ["input_gs", "output_gs", "cameras"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise KeyError(f"Missing fields in scene payload: {missing_fields}")

    input_gs = payload["input_gs"]
    output_gs = payload["output_gs"]
    cameras = payload["cameras"]
    residual_keys = payload.get("residual_keys", [])
    if len(residual_keys) == 0:
        residual_keys = sorted([key for key in input_gs.keys() if key in output_gs])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering because gs_utils uses CUDA rasterization.")
    device = torch.device("cuda")

    input_gs = move_to_device(input_gs, device)
    output_gs = move_to_device(output_gs, device)
    cameras = move_to_device(cameras, device)
    single_camera = build_single_camera(cameras)

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path_mp4 = output_dir / "morph.mp4"
    video_path_avi = output_dir / "morph.avi"
    frames = []

    with torch.no_grad():
        for frame_idx, t in enumerate(tqdm(torch.linspace(0.0, 1.0, steps=args.num_frames, device=device))):
            interp_gs = {k: v for k, v in input_gs.items()}
            for key in residual_keys:
                if key in input_gs and key in output_gs:
                    interp_gs[key] = input_gs[key] + t * (output_gs[key] - input_gs[key])

            frame_rgb, _ = gs_utils.rasterize_gaussians_to_multiimgs(interp_gs, single_camera)
            frame = torch.clamp(frame_rgb[0], 0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
            frames.append(frame)

            Image.fromarray(frame).save(frames_dir / f"frame_{frame_idx:04d}.png")

    print(f"Saved frames to {frames_dir}")
    if args.skip_video:
        print("Skipped video encoding (--skip_video).")
        return

    ok, reason = encode_video_with_ffmpeg(frames_dir, video_path_mp4, args.fps)
    if ok:
        print(f"Saved video to {video_path_mp4} ({reason})")
        return

    # Fallback for environments without ffmpeg/libx264:
    # write an MJPG AVI, which is generally easier to play than OpenCV mp4v outputs.
    ok_avi, reason_avi = encode_video_with_cv2(
        frames, video_path_avi, args.fps, codecs=["MJPG", "XVID"]
    )
    if ok_avi:
        print(f"Saved video to {video_path_avi} ({reason_avi}; mp4 unavailable: {reason})")
        return

    # Last resort: still attempt MP4 with simpler codecs.
    ok_mp4, reason_mp4 = encode_video_with_cv2(
        frames, video_path_mp4, args.fps, codecs=["mp4v", "XVID", "MJPG"]
    )
    if ok_mp4:
        print(f"Saved video to {video_path_mp4} ({reason_mp4}; ffmpeg failed: {reason})")
        return

    print(
        "Video encoding failed "
        f"(ffmpeg: {reason}; avi fallback: {reason_avi}; mp4 fallback: {reason_mp4}). "
        f"Frames are available at {frames_dir}."
    )


if __name__ == "__main__":
    main()
