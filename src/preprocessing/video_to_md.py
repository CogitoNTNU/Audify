from urllib.parse import urlparse, parse_qs
import subprocess
import os
from pathlib import Path


def youtube_to_markdown(youtube_url: str) -> str:
    output_dir = Path("data/markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    #extracting id's
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        video_id = parsed_url.path.lstrip("/")
    else:
        raise ValueError("Invalid YouTube URL")

    #output markdown file path
    output_path = output_dir / f"{video_id}.md"

    #skip conversion if file already exists
    if output_path.exists():
        return str(output_path)

    #run yt2doc command
    cmd = ["yt2doc", "--video", youtube_url, "-o", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt2doc failed:\n{result.stderr}")

    return str(output_path)


if __name__ == "__main__":
    file_path = youtube_to_markdown("https://www.youtube.com/watch?v=Gx5qb1uHss4")
    print("Transcript saved at:", file_path)
