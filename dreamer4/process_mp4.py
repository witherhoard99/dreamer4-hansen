import os
import glob
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_THREADS = 16


def process_video(mp4_file):
    dir_name = os.path.dirname(mp4_file)
    base_name = os.path.basename(mp4_file)
    temp_out = os.path.join(dir_name, f"temp_{base_name}")
    og_dir = os.path.join(dir_name, "og_videos")

    # Safely create the 'og_videos' directory if it doesn't already exist
    os.makedirs(og_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", mp4_file,
        "-vf", "scale=398:224, fps=10",
        "-threads", "2",
        temp_out
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0:
        # Move the original video into the 'og_videos' directory
        og_file_path = os.path.join(og_dir, base_name)
        os.replace(mp4_file, og_file_path)  # Using replace instead of rename prevents errors if the file already exists

        # Rename the newly processed video to the original file's name
        os.rename(temp_out, mp4_file)

        return mp4_file, True
    else:
        # Clean up the temp file if the conversion fails
        if os.path.exists(temp_out):
            os.remove(temp_out)
        return mp4_file, False


def process_all_videos(target_dir):
    mp4s = glob.glob(os.path.join(target_dir, "*.mp4"))
    mp4s = [f for f in mp4s if not os.path.basename(f).startswith("temp_")]

    if not mp4s:
        print("No MP4 files found to process.")
        return

    print(f"Found {len(mp4s)} MP4 files. Starting parallel video conversion...")

    with ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_video, mp4): mp4 for mp4 in mp4s}

        for i, future in enumerate(as_completed(futures), 1):
            file_name, success = future.result()
            base_name = os.path.basename(file_name)
            if success:
                print(f"[{i}/{len(mp4s)}] Resized and backed up original: {base_name}")
            else:
                print(f"[{i}/{len(mp4s)}] Failed: {base_name}")


def main():
    parser = argparse.ArgumentParser(description="Process zips, parquets, and videos in a target directory.")
    parser.add_argument("target_dir", help="Path to the directory containing the files")
    args = parser.parse_args()

    target_dir = args.target_dir

    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    print(f"Processing directory: {target_dir}")

    print("\nStarting video operations...")
    process_all_videos(target_dir)

    print("\nAll tasks complete.")


if __name__ == "__main__":
    main()
