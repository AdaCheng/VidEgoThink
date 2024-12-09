from moviepy.editor import VideoFileClip
import json
import os
from concurrent.futures import ThreadPoolExecutor


def trim_video(video_path, end_time, output_path):
    """
    从视频开始到end_time裁剪视频，并保存到output_path。
    """
    with VideoFileClip(video_path) as video:
        trimmed_video = video.subclip(0, end_time)  # start_time默认为0
        trimmed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

def process_video(item, args):
    """
    为单个视频条目裁剪指定的部分。
    """
    video_name = item['video_path']
    video_uid = item['video_uid']
    video_path = args.video_folder + video_uid + ".mp4"
    end_time = item['end_time']
    output_path = args.output_folder + video_name
    if not os.path.exists(output_path):
        print(f"Processing {video_path}...")
        trim_video(video_path, end_time, output_path)
        print(f"Saved trimmed video to {output_path}")
    else:
        print(f"Video {video_path} is found!")


def process_videos(json_data):
    """
    使用多线程读取JSON数据，并为每个视频条目裁剪指定的部分。
    """
    with ThreadPoolExecutor(max_workers=8) as executor:  # 可以调整max_workers以优化性能
        futures = [executor.submit(process_video, item) for item in json_data]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 等待所有视频处理完成

if main == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./final_hp_h2m.json")
    parser.add_argument("--video_folder", type=str, default="/goal_step/v2/full_scale/")
    parser.add_argument("--output_folder", type=str, default="./val_hp_video/")
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        json_data = json.load(f)

    process_videos(json_data, args)