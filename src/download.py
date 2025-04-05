import subprocess
from typing import Optional
from pathlib import Path
from src.config import DownloadConfig

class YouTubeDL:
    def __init__(self, output_folder: Path = Path("data/00--raw/videos")):
        """Handles downloading YouTube videos directly in MP4 format."""
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def download_playlist(self, playlist_url: str):
        """Downloads YouTube videos as MP4 without AV1 and without HLS fragmentation."""
        n_to_download = DownloadConfig.numbers_to_download
        command = [
            "yt-dlp",
            "--cookies", "data/cookies.txt",
            "--playlist-end", n_to_download, # Limits to first 3 videos (remove if not needed)
            "-f", "best[ext=mp4]",           # Forces MP4 format
            "--merge-output-format", "mp4",  # Ensures the final file is MP4
            "-o", f"{self.output_folder}/%(title)s.%(ext)s",
            playlist_url
        ]
        subprocess.run(command, check=True)

    def download_single_video(self, video_url: str, output_path: Optional[Path] = None):
        """Downloads a single YouTube video as MP4 named by videoID"""
        if output_path is None:
            output_path = self.output_folder / "%(id)s.%(ext)s"
        else:
            output_path = output_path.with_suffix(".%(ext)s")
        command = [
        "yt-dlp",
        "--cookies", "data/cookies.txt",
        "-f", "best[ext=mp4]",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        video_url
        ]
        subprocess.run(command, check=True)


    def list_downloaded_videos(self):
        """Lists all downloaded MP4 files."""
        videos = list(self.output_folder.glob("*.mp4"))
        print(f"✅ Downloaded {len(videos)} MP4 videos:")
        for video in videos:
            print(f"   📁 {video.name}")

if __name__ == "__main__":
    output_folder = Path(__file__).resolve().parent.parent / "data" / "00--raw" / "videos"
    yt_dl = YouTubeDL(output_folder)

    # Bundesliga Playlist
    url = "https://www.youtube.com/watch?v=BDtadhXmuAA&list=PLsFWLnYCEXEVcFhJIJe3zhJbyGhx99e7y"

    print("📥 Downloading videos in MP4 format...")
    yt_dl.download_playlist(playlist_url=url)

    print("📂 Listing downloaded videos...")
    yt_dl.list_downloaded_videos()
