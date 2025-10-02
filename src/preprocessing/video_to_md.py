from urllib.parse import urlparse, parse_qs
import subprocess 
import os 
from pathlib import Path 


def youtube_to_md(youtube_url: str)->str:
    output_dir = r"data/markdown"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    parsed = urlparse(youtube_url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [None])[0]

    elif parsed.hostname == "youtu.be":
        video_id = parsed.path.lstrip("/")
    else:
        raise ValueError("This is an invalid YouTube URL")
    
    md_path = os.path.join(output_dir, f'{video_id}.md')

    cmd = ["yt2doc", "--video", youtube_url, "-o", md_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt2doc failed:\n{result.stderr}")

    return md_path

file_path = youtube_to_md("https://www.youtube.com/watch?v=Gx5qb1uHss4")
print("Transcript saved at:", file_path)
