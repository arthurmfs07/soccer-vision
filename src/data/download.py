import subprocess
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs


class YouTubeDL:
    """Download YouTube content (single videos or playlists) as MP4."""

    def __init__(self, output_folder: str = "data/00--raw/videos") -> None:
        self.output_folder: Path = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, output_filename: Optional[str] = None, best_quality: bool = False) -> None:
        """Automatically choose between playlist and single-video download."""
        if best_quality:
            self._quality = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best"
        else:
            self._quality = "best[ext=mp4]"

        if self._is_playlist(url):
            self.download_playlist(url)            # playlist: ignore output_path
        else:
            self.download_single_video(url, output_filename)

    def download_playlist(self, playlist_url: str, n_to_download: int=1) -> None:
        """Download the first *n* items of a playlist as MP4."""
        command = [
            "yt-dlp",
            "--cookies", "data/cookies.txt",
            "--playlist-end", str(n_to_download),
            "-f", self._quality,
            "--merge-output-format", "mp4",
            "--no-keep-video",  # Remove arquivos temporários após mesclar
            "-o", f"{self.output_folder}/%(title)s.%(ext)s",
            playlist_url,
        ]
        subprocess.run(command, check=True)

    def download_single_video(
        self,
        video_url: str,
        output_filename: Optional[str] = None,
    ) -> None:
        """Download one video as MP4 (named by video ID unless output_path given)."""
        if output_filename is None:
            output_filename: Path = self.output_folder / "%(id)s.%(ext)s"
        else:
            output_filename: Path = self.output_folder / Path(output_filename).with_suffix(".%(ext)s")
        command = [
            "yt-dlp",
            "--cookies", "data/cookies.txt",
            "-f", self._quality,
            "--merge-output-format", "mp4",
            "--no-keep-video",  # Remove arquivos temporários após mesclar
            "-o", str(output_filename),
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
    print("📥 Downloading playlist …")
    yt_dl.download(playlist)

    # single   = "https://www.youtube.com/watch?v=JA0p0Bg9N1w"
    # print("📥 Downloading single video …")
    # yt_dl.download(single)

    # single   = "https://www.youtube.com/watch?v=JA0p0Bg9N1w"
    # print("📥 Downloading single video …")
    # yt_dl.download(single, output_filename="best_JA0p0Bg9N1w.mp4", best_quality=True)


    # single   = "https://www.youtube.com/live/DWw5BvI-13M"
    # print("📥 Downloading single video …")
    # yt_dl.download(single)

    # single   = "https://www.youtube.com/live/DWw5BvI-13M"
    # print("📥 Downloading single video …")
    # yt_dl.download(single, output_filename="best_DWw5BvI-13M.mp4", best_quality=True)


    print("📥 Downloaded videos:")
    yt_dl.list_downloaded_videos()