import subprocess
from pathlib import Path

class YouTubeDL:
    def __init__(self, output_folder: Path = Path("videos")):
        self.output_folder = output_folder

    def download_playlist(self, playlist_url):
        command = [
            "yt-dlp",
            "--playlist-end", "3",
            "--verbose",
            "-o", f"{self.output_folder}/%(title)s.%(ext)s",
            playlist_url
        ]
        subprocess.run(command)


if __name__ == "__main__":

    output_folder = Path(__file__).resolve().parent / "data" / "00--raw" / "videos"
    yt_dl = YouTubeDL(output_folder)

    # bundesliga
    url = "https://www.youtube.com/watch?v=BDtadhXmuAA&list=PLsFWLnYCEXEVcFhJIJe3zhJbyGhx99e7y"
    yt_dl.download_playlist(playlist_url=url)

# x, y, player_id, is_detected