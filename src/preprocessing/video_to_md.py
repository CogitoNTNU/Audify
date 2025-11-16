from urllib.parse import urlparse, parse_qs
import os
from pathlib import Path

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    YOUTUBE_TRANSCRIPT_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_AVAILABLE = False


def youtube_to_markdown(youtube_url: str) -> str:
    """
    Extract transcript from YouTube video and save as markdown.
    Falls back to yt2doc if youtube_transcript_api is not available.
    """
    if not YOUTUBE_TRANSCRIPT_AVAILABLE:
        # Fallback to yt2doc if youtube-transcript-api is not installed
        return _youtube_to_markdown_yt2doc(youtube_url)
    
    output_dir = Path("data/markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract video ID
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        video_id = parsed_url.path.lstrip("/")
    else:
        raise ValueError("Invalid YouTube URL")

    if not video_id:
        raise ValueError("Could not extract video ID from URL")

    # Output markdown file path
    output_path = output_dir / f"{video_id}.md"

    # Skip conversion if file already exists
    if output_path.exists():
        return str(output_path)

    try:
        # Get transcript - use instance method
        api = YouTubeTranscriptApi()
        transcript_data = api.get_transcript(video_id)
        
        # Format transcript as markdown
        transcript_text = "\n\n".join([entry['text'] for entry in transcript_data])
        
        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# YouTube Video Transcript\n\n")
            f.write(f"Video ID: {video_id}\n\n")
            f.write(f"## Transcript\n\n")
            f.write(transcript_text)
        
        return str(output_path)
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract YouTube transcript: {str(e)}")


def _youtube_to_markdown_yt2doc(youtube_url: str) -> str:
    """Fallback method using yt2doc command-line tool."""
    import subprocess
    
    output_dir = Path("data/markdown")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract video ID
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == "youtu.be":
        video_id = parsed_url.path.lstrip("/")
    else:
        raise ValueError("Invalid YouTube URL")

    # Output markdown file path
    output_path = output_dir / f"{video_id}.md"

    # Skip conversion if file already exists
    if output_path.exists():
        return str(output_path)

    # Run yt2doc command
    cmd = ["yt2doc", "--video", youtube_url, "-o", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt2doc failed:\n{result.stderr}")

    return str(output_path)
