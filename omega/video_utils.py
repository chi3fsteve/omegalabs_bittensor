import os
import tempfile
from typing import Optional, BinaryIO

import bittensor as bt
import ffmpeg
from pydantic import BaseModel
from yt_dlp import YoutubeDL

from omega.constants import FIVE_MINUTES

import asyncio

EXISTING_IDS_FILE = "existing_ids.txt"

def load_existing_ids():
    try:
        with open(EXISTING_IDS_FILE, "r") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        print("Datasets file not found")
        return set()

existing_ids = load_existing_ids()

def seconds_to_str(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def clip_video(video_path: str, start: int, end: int) -> Optional[BinaryIO]:
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    (
        ffmpeg
        .input(video_path, ss=seconds_to_str(start), to=seconds_to_str(end))
        .output(temp_fileobj.name, c="copy")  # copy flag prevents decoding and re-encoding
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_fileobj


def skip_live(info_dict):
    """
    function to skip downloading if it's a live video (yt_dlp doesn't respect the 20 minute 
    download limit for live videos), and we don't want to hang on an hour long stream
    """
    if info_dict.get("is_live"):
        return "Skipping live video"
    return None


class YoutubeResult(BaseModel):
    video_id: str
    title: str
    description: Optional[str]
    length: int
    views: int


def search_videos(query, max_results=8):
    videos = []
    ydl_opts = {
        "format": "worst",
        "dumpjson": True,
        "extract_flat": True,
        "quiet": True,
        "simulate": True,
        "match_filter": skip_live,
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_query = f"ytsearch{max_results * 5}:{query}"  # Search for 5 times the desired number of videos
            result = ydl.extract_info(search_query, download=False)
            if "entries" in result and result["entries"]:
                unique_videos = []
                for entry in result["entries"]:
                    if entry["id"] not in existing_ids:
                        video = YoutubeResult(
                            video_id=entry["id"],
                            title=entry["title"],
                            description=entry.get("description"),
                            length=(int(entry.get("duration")) if entry.get("duration") else FIVE_MINUTES),
                            views=(entry.get("view_count") if entry.get("view_count") else 0),
                        )
                        unique_videos.append(video)
                        existing_ids.add(entry["id"])
                        if len(unique_videos) == max_results:
                            break
                
                # Fill remaining slots with videos from the search query results
                if len(unique_videos) < max_results:
                    remaining_slots = max_results - len(unique_videos)
                    for entry in result["entries"]:
                        if entry["id"] not in existing_ids and entry["id"] not in [video.video_id for video in unique_videos]:
                            video = YoutubeResult(
                                video_id=entry["id"],
                                title=entry["title"],
                                description=entry.get("description"),
                                length=(int(entry.get("duration")) if entry.get("duration") else FIVE_MINUTES),
                                views=(entry.get("view_count") if entry.get("view_count") else 0),
                            )
                            unique_videos.append(video)
                            if len(unique_videos) == max_results:
                                break
                
                videos = unique_videos

        except Exception as e:
            bt.logging.warning(f"Error searching for videos: {e}")
            return []
    return videos


def get_video_duration(filename: str) -> int:
    metadata = ffmpeg.probe(filename)
    video_stream = next((stream for stream in metadata['streams'] if stream['codec_type'] == 'video'), None)
    duration = int(float(video_stream['duration']))
    return duration


class IPBlockedException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


async def download_video(
    video_id: str, start: Optional[int]=None, end: Optional[int]=None, proxy: Optional[str]=None
) -> Optional[BinaryIO]:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    ydl_opts = {
        "format": "worst",  # Download the worst quality
        "outtmpl": temp_fileobj.name,  # Set the output template to the temporary file's name
        "overwrites": True,
        "quiet": True,
        "noprogress": True,
        "match_filter": skip_live,
    }

    if start is not None and end is not None:
        ydl_opts["download_ranges"] = lambda _, __: [{"start_time": start, "end_time": end}]

    if proxy is not None:
        ydl_opts["proxy"] = proxy

    try:
        with YoutubeDL(ydl_opts) as ydl:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ydl.download, [video_url])

        # Check if the file is empty (download failed)
        if os.stat(temp_fileobj.name).st_size == 0:
            print(f"Error downloading video: {temp_fileobj.name} is empty")
            temp_fileobj.close()
            return None

        return temp_fileobj
    except Exception as e:
        temp_fileobj.close()
        if (
            "Your IP is likely being blocked by Youtube" in str(e) or
            "Requested format is not available" in str(e)
        ):
            raise IPBlockedException(e)
        print(f"Error downloading video: {e}")
        return None


def copy_audio(video_path: str) -> BinaryIO:
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    (
        ffmpeg
        .input(video_path)
        .output(temp_audiofile.name, vn=None, acodec='copy')
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_audiofile