import os
import tempfile
from typing import Optional, BinaryIO
import time

import bittensor as bt
import ffmpeg
from pydantic import BaseModel
from yt_dlp import YoutubeDL

import redis
from datasets import load_dataset

from omega.constants import FIVE_MINUTES

# Set up Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

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

def load_existing_ids():
    # Check if the existing video IDs are already in the Redis cache
    if redis_client.exists("existing_video_ids"):
        # If the IDs are in the cache, retrieve them
        existing_ids = redis_client.smembers("existing_video_ids")
        return set(id.decode('utf-8') for id in existing_ids)
    else:
        # If the IDs are not in the cache, load them from the dataset and store them in the cache
        existing_ids = set(load_dataset('omegalabsinc/omega-multimodal')['train']['youtube_id'])
        redis_client.sadd("existing_video_ids", *existing_ids)
        return existing_ids

def search_videos(query, max_results=8, max_time=60):
    existing_ids = load_existing_ids()
    videos = []
    ydl_opts = {
        "format": "worst",
        "dumpjson": True,
        "extract_flat": True,
        "quiet": True,
        "simulate": True,
        "match_filter": skip_live,
    }
    start_time = time.time()
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_query = f"ytsearch{max_results * 10}:{query}"  # Search for 10 times the number of desired videos
            result = ydl.extract_info(search_query, download=False)
            if "entries" in result and result["entries"]:
                for entry in result["entries"]:
                    video_id = entry["id"]
                    if video_id not in existing_ids:
                        existing_ids.add(video_id)
                        redis_client.sadd("existing_video_ids", video_id)
                        videos.append(YoutubeResult(
                            video_id=video_id,
                            title=entry["title"],
                            description=entry.get("description"),
                            length=(int(entry.get("duration")) if entry.get("duration") else FIVE_MINUTES),
                            views=(entry.get("view_count") if entry.get("view_count") else 0),
                        ))
                    if len(videos) == max_results:
                        break
                    if time.time() - start_time > max_time:
                        bt.logging.warning(f"Search time limit of {max_time} seconds exceeded. Returning {len(videos)} videos.")
                        break
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


def download_video(
    video_id: str, start: Optional[int]=None, end: Optional[int]=None, proxy: Optional[str]=None
) -> Optional[BinaryIO]:
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    ydl_opts = {
        "format": "worst",  # Download the worst quality
        "outtmpl": temp_fileobj.name,  # Set the output template to the temporary file"s name
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
            ydl.download([video_url])

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
