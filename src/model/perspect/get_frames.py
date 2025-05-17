import subprocess
import pandas as pd
from pathlib import Path
import urllib.parse
from typing import Optional
from src.utils import load_abs_path
from data.download import YouTubeDL

from src.utils import parse_timestamp

# Match information with start and end times for each half
match_info = {
    3895113: {
        "url": "https://www.youtube.com/watch?v=GpODt842IZk",
        "first_half_start": "1:05:28",
        "first_half_end": "1:52:31",
        "second_half_start": "2:08:43",
        "second_half_end": "2:57:58"
    },
    3895158: {
        "url": "https://www.youtube.com/watch?v=JDSU8DJCALU",
        "first_half_start": "1:05:02",
        "first_half_end": "1:53:42",
        "second_half_start": "2:10:43",
        "second_half_end": "3:00:25"
    },
    3895182: {
        "url": "https://www.youtube.com/watch?v=BG9wgj_7caE",
        "first_half_start": "1:04:47",
        "first_half_end": "1:52:49",
        "second_half_start": "2:09:08",
        "second_half_end": "2:59:11"
    },
    3895202: {
        "url": "https://www.youtube.com/watch?v=JKNPeLSSs8s",
        "first_half_start": "1:04:40",
        "first_half_end": "1:53:44",
        "second_half_start": "2:10:09",
        "second_half_end": "2:58:39"
    },
    3895232: {
        "url": "https://www.youtube.com/watch?v=kI3VqQfatCM",
        "first_half_start": "1:13:02",
        "first_half_end": "1:59:35",
        "second_half_start": "2:15:52",
        "second_half_end": "3:05:29"
    },
    3895302: {
        "url": "https://www.youtube.com/watch?v=NMgJxAfv2VQ",
        "first_half_start": "1:04:40",
        "first_half_end": "1:53:47",
        "second_half_start": "2:09:06",
        "second_half_end": "2:53:29"
    },
    3895309: {
        "url": "https://www.youtube.com/watch?v=qWywGaah9w4",
        "first_half_start": "1:04:52",
        "first_half_end": "1:53:04",
        "second_half_start": "2:10:18",
        "second_half_end": "3:02:10"
    },
    3895320: {
        "url": "https://www.youtube.com/watch?v=C0uM_yFJ1H4",
        "first_half_start": "1:04:35",
        "first_half_end": "1:51:23",
        "second_half_start": "2:08:43",
        "second_half_end": "3:01:07"
    },
    3895340: {
        "url": "https://www.youtube.com/watch?v=niNTcVMUcoU",
        "first_half_start": "1:04:51",
        "first_half_end": "1:50:59",
        "second_half_start": "2:10:28",
        "second_half_end": "2:59:01"
    }
}



def clean_youtube_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    clean_query = {k: v for k, v in query.items() if k == "v"}
    new_query = urllib.parse.urlencode(clean_query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))



def extract_frame_at_timestamp(
    video_path: str,
    video_start_sec: int,
    match_timestamp_sec: float,
    output_path: Path
):
    """
    Extracts a single frame at a specific timestamp.
    video_start_sec: When the half starts in the YouTube video (in seconds)
    match_timestamp_sec: Timestamp from the CSV (in seconds, relative to the start of the half)
    """
    absolute_sec = video_start_sec + match_timestamp_sec

    cmd = [
        "ffmpeg",
        "-y",                # Overwrite existing files without asking
        "-ss", str(absolute_sec),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path)
    ]

    subprocess.run(
        cmd, 
        check=True, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL
        )


def main():
    data_path = load_abs_path() / "data"
    raw_path = data_path / "00--raw"
    csv_path = raw_path / "df_on_360.csv"
    video_dir = raw_path / "temp-videos"
    frames_dir = data_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    yt_dl = YouTubeDL(video_dir)

    for match_id, info in match_info.items():
        match_id_int = int(match_id)
        video_filename = f"{match_id}.mp4"
        video_path = video_dir / video_filename

        print(f"\n🎥 Match {match_id} - {info['url']}")
        print("📥 Downloading MP4...")
        clean_url = clean_youtube_url(info["url"])
        yt_dl.download_single_video(clean_url, output_path=video_path)

        if not video_path.exists():
            print(f"❌ Failed to download video for match {match_id}")
            continue

        

        match_df = df[df["match_id"] == match_id]
        if match_df.empty:
            print(f"⚠️ No timestamps found for match {match_id}")
            video_path.unlink(missing_ok=True)
            continue

        print(f"✅ Found {len(match_df)} timestamps")

        match_dir = frames_dir / f"match_{match_id}"
        match_dir.mkdir(parents=True, exist_ok=True)

        for period in [1, 2]:
            half_df = match_df[match_df["period"] == period]
            if half_df.empty:
                continue

            game_time = "t1" if period == 1 else "t2"
            video_start_sec = parse_timestamp(
                info["first_half_start"] if period == 1 else info["second_half_start"]
            )

            for _, row in half_df.iterrows():
                timestamp_str = row["timestamp"]
                try:
                    timestamp_sec = float(timestamp_str) if isinstance(timestamp_str, (int, float)) \
                        else parse_timestamp(timestamp_str)
                except Exception as e:
                    print(f"❌ Failed timestamp {timestamp_str}: {e}")
                    continue

                output_filename = f"{game_time}_{timestamp_sec:.3f}.jpg"
                output_path = match_dir / output_filename

                try:
                    extract_frame_at_timestamp(
                        video_path=video_path,
                        video_start_sec=video_start_sec,
                        match_timestamp_sec=timestamp_sec,
                        output_path=output_path
                    )
                except subprocess.CalledProcessError as e:
                    print(f"❌ ffmpeg failed at {timestamp_sec:.3f}s: {e}")
                    continue

        print(f"🧹 Deleting MP4 for match {match_id}")
        video_path.unlink(missing_ok=True)

    print("\n✅ Frame extraction complete!")


if __name__ == "__main__":
    main()