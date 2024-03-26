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

import logging

from omega.constants import FIVE_MINUTES

# Set up Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, ssl=True)

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_existing_ids():
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            if redis_client.exists("existing_video_ids"):
                logging.info("Fetching existing video IDs from Redis...")
                existing_ids = redis_client.smembers("existing_video_ids")
                logging.info(f"Fetched {len(existing_ids)} existing video IDs from Redis.")
                return set(id.decode('utf-8') for id in existing_ids)
            else:
                logging.info("Loading existing video IDs from the dataset...")
                existing_ids = set(load_dataset('omegalabsinc/omega-multimodal')['train']['youtube_id'])
                logging.info(f"Loaded {len(existing_ids)} existing video IDs from the dataset.")
                
                logging.info("Storing existing video IDs in Redis...")
                redis_client.sadd("existing_video_ids", *existing_ids)
                logging.info("Stored existing video IDs in Redis.")
                return existing_ids
        except redis.ConnectionError as e:
            logging.error(f"Redis connection error: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Failed to connect to Redis after multiple retries.")
                raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    raise redis.ConnectionError("Failed to connect to Redis after multiple retries.")


def search_videos(query, max_results=8, max_time=60):
    try:
        existing_ids = load_existing_ids()
        logging.info(f"Loaded {len(existing_ids)} existing video IDs.")

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
                search_query = f"ytsearch{max_results * 10}:{query}"
                logging.info(f"Searching for videos with query: {search_query}")

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
                            logging.info(f"Found {len(videos)} unique videos.")
                            break
                        if time.time() - start_time > max_time:
                            logging.warning(f"Search time limit of {max_time} seconds exceeded. Returning {len(videos)} videos.")
                            break
                else:
                    logging.warning("No video entries found in the search result.")

                # If there are less than 8 unique videos, fill the remaining slots with non-unique videos
                if len(videos) < max_results:
                    logging.info(f"Filling remaining slots with non-unique videos.")
                    for entry in result["entries"]:
                        video_id = entry["id"]
                        if video_id not in [video.video_id for video in videos]:
                            videos.append(YoutubeResult(
                                video_id=video_id,
                                title=entry["title"],
                                description=entry.get("description"),
                                length=(int(entry.get("duration")) if entry.get("duration") else FIVE_MINUTES),
                                views=(entry.get("view_count") if entry.get("view_count") else 0),
                            ))
                        if len(videos) == max_results:
                            logging.info(f"Found a total of {len(videos)} videos.")
                            break
            except Exception as e:
                logging.error(f"Error searching for videos: {e}")
                return []

        logging.info(f"Returning {len(videos)} videos.")
        return videos
    except redis.ConnectionError as e:
        logging.error(f"Redis connection error: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred during video search: {e}")
        return []


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
 
