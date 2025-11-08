#!/usr/bin/env python3
import sys
import time
import shutil
import core.globals

if not shutil.which('ffmpeg'):
    print('❌ ffmpeg is not installed. Please install it first.')
    quit()

# -----------------------------
#  GPU 检查
# -----------------------------
if '--gpu' not in sys.argv:
    core.globals.providers = ['CPUExecutionProvider']
elif 'ROCMExecutionProvider' not in core.globals.providers:
    import torch
    if not torch.cuda.is_available():
        quit("❌ CUDA not available, but --gpu flag used.")

# -----------------------------
# 导入模块
# -----------------------------
import glob
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import psutil
import cv2
from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames
from core.config import get_face

# -----------------------------
# CLI 参数
# -----------------------------
parser = argparse.ArgumentParser(description="Roop headless (no-GUI) version for Colab/Linux.")
parser.add_argument('-f', '--face', required=True, help='path to source face image')
parser.add_argument('-t', '--target', required=True, help='path to target video or image')
parser.add_argument('-o', '--output', help='path to save output file')
parser.add_argument('--keep-fps', action='store_true', help='keep original fps (for video)')
parser.add_argument('--keep-frames', action='store_true', help='keep extracted frames')
parser.add_argument('--gpu', action='store_true', help='use GPU if available')
parser.add_argument('--cores', type=int, help='number of CPU cores to use')
args = parser.parse_args()

# -----------------------------
# 初始化参数
# -----------------------------
if not args.cores:
    args.cores = max(1, psutil.cpu_count() - 1)

sep = "\\" if os.name == "nt" else "/"

# -----------------------------
# 核心函数
# -----------------------------
def start_processing(frame_paths):
    start_time = time.time()
    pool = mp.Pool(args.cores)
    n = len(frame_paths) // args.cores or 1
    processes = []

    for i in range(0, len(frame_paths), n):
        p = pool.apply_async(process_video, args=(args.face, frame_paths[i:i + n],))
        processes.append(p)

    for p in processes:
        p.get()
    pool.close()
    pool.join()

    print(f"✅ Processing completed in {time.time() - start_time:.2f} seconds")

# -----------------------------
# 主流程
# -----------------------------
print("\n🚀 Starting face swap process...\n")

# 检查源人脸
if not os.path.isfile(args.face):
    quit("❌ Source face not found.")
test_face = get_face(cv2.imread(args.face))
if not test_face:
    quit("❌ No face detected in source image.")

target = args.target
if not os.path.isfile(target):
    quit("❌ Target file not found.")

# 输出路径
if not args.output:
    base = os.path.splitext(os.path.basename(target))[0]
    args.output = os.path.join(os.path.dirname(target), f"{base}_swapped.mp4")

# 图片模式
if is_img(target):
    print("🖼️ Swapping face in image...")
    process_img(args.face, target, args.output)
    print(f"✅ Saved swapped image: {args.output}")
    quit()

# 视频模式
print("🎞️ Processing video...")
video_name = os.path.basename(target)
video_name = os.path.splitext(video_name)[0]
output_dir = os.path.join(os.path.dirname(target), f"{video_name}_frames")
Path(output_dir).mkdir(exist_ok=True)

fps = detect_fps(target)
if not args.keep_fps and fps > 30:
    temp_path = os.path.join(output_dir, f"{video_name}_30fps.mp4")
    set_fps(target, temp_path, 30)
    target, fps = temp_path, 30

print("📸 Extracting frames...")
extract_frames(target, output_dir)

frame_paths = sorted(glob.glob(f"{output_dir}/*.png"), key=lambda x: int(x.split(sep)[-1].replace(".png", "")))

print(f"🧠 Total frames: {len(frame_paths)}")
start_processing(frame_paths)

print("🎬 Creating video...")
create_video(video_name, fps, output_dir)

print("🔊 Adding audio...")
add_audio(output_dir, target, args.keep_frames, args.output)

print(f"\n✅ Done! Output saved at: {args.output}\n")
