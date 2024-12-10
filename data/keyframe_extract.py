import os
import cv2
import subprocess

def extract_keyframes(video_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建FFmpeg命令
    output_pattern = os.path.join(output_dir, "frame_%04d.png")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "select='eq(pict_type\,I)+eq(pict_type\,P)+eq(pict_type\,B)'",
        "-vsync", "vfr",
        output_pattern
    ]
    
    # 执行FFmpeg命令
    subprocess.run(command, check=True)

def process_videos(video_dir, output_base_dir):
    # 遍历视频目录中的所有视频文件
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                video_path = os.path.join(root, file)

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Failed to open {video_path}")
                    continue
                cap.release()

                output_dir = os.path.join(output_base_dir, os.path.splitext(file)[0])
                print(f"Processing {video_path}...")
                extract_keyframes(video_path, output_dir)
                print(f"Keyframes saved to {output_dir}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="/videgothink/goalstep", help='input folder')
    parser.add_argument('--output_folder', type=str, default="/videgothink/goalstep_keyframe", help='output folder')
    args = parser.parse_args()

    process_videos(args.input_folder, args.output_folder)

    
