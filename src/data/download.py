import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

from src.config import DownloadConfig


class YouTubeDL:
    """Download YouTube content (single videos or playlists) as MP4."""

    def __init__(self, output_folder: Path = Path("data/00--raw/videos")) -> None:
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, output_path: Optional[Path] = None) -> None:
        """Automatically choose between playlist and single-video download."""
        if self._is_playlist(url):
            self.download_playlist(url)            # playlist: ignore output_path
        else:
            self.download_single_video(url, output_path)

    def download_playlist(self, playlist_url: str, n_to_download: int=1) -> None:
        """Download the first *n* items of a playlist as MP4."""
        command = [
            "yt-dlp",
            "--cookies", "data/cookies.txt",
            "--playlist-end", str(n_to_download),
            "-f", "best[ext=mp4]",
            "--merge-output-format", "mp4",
            "-o", f"{self.output_folder}/%(title)s.%(ext)s",
            playlist_url,
        ]
        subprocess.run(command, check=True)

    def download_single_video(
        self,
        video_url: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """Download one video as MP4 (named by video ID unless output_path given)."""
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
            video_url,
        ]
        subprocess.run(command, check=True)

    def list_downloaded_videos(self) -> None:
        """Print all MP4 files in *output_folder*."""
        videos = list(self.output_folder.glob("*.mp4"))
        print(f"✅ Downloaded {len(videos)} MP4 videos:")
        for v in videos:
            print(f"   📁 {v.name}")

    @staticmethod
    def _is_playlist(url: str) -> bool:
        """Heuristic: URL has a non-empty *list=* query parameter."""
        qs = parse_qs(urlparse(url).query)
        return bool(qs.get("list"))




if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent.parent / "data" / "00--raw" / "videos"
    yt_dl = YouTubeDL(out)

    playlist = "https://www.youtube.com/watch?v=BDtadhXmuAA&list=PLsFWLnYCEXEVcFhJIJe3zhJbyGhx99e7y"
    single   = "https://www.youtube.com/watch?v=JA0p0Bg9N1w"

    # print("📥 Downloading playlist …")
    # yt_dl.download(playlist)

    print("📥 Downloading single video …")
    yt_dl.download(single)

    yt_dl.list_downloaded_videos()