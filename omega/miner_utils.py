import os
import time
from typing import List, Tuple

import asyncio
import bittensor as bt

from omega.protocol import VideoMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH, FIVE_MINUTES
from omega import video_utils

from openai import OpenAI

client = OpenAI()

async def get_description(yt: video_utils.YoutubeDL, video_path: str) -> str:
    title = yt.title
    description = yt.description if yt.description else ""

    prompt = f"Video Title: {title}\n\nVideo Description: {description}\n\nGenerate a concise and informative description for the video:"

    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        generated_description = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        bt.logging.error(f"Error generating description using OpenAI API: {e}")
        generated_description = ""

    return generated_description


async def get_relevant_timestamps(query: str, yt: video_utils.YoutubeDL, video_path: str) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    Miner TODO: Implement logic to get the optimal start and end timestamps of a video for
    ensuring relevance to the query.
    """
    start_time = 0
    end_time = min(yt.length, MAX_VIDEO_LENGTH)
    return start_time, end_time


async def process_video(query: str, result: video_utils.YoutubeDL, imagebind: ImageBind) -> VideoMetadata:
    start = time.time()
    download_path = video_utils.download_video(
        result.video_id,
        start=0,
        end=min(result.length, FIVE_MINUTES)  # download the first 5 minutes at most
    )
    if download_path:
        clip_path = None
        try:
            result.length = video_utils.get_video_duration(download_path.name)  # correct the length
            bt.logging.info(f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
            start, end = await get_relevant_timestamps(query, result, download_path)
            description = await get_description(result, download_path)
            clip_path = video_utils.clip_video(download_path.name, start, end)
            embeddings = imagebind.embed([description], [clip_path])
            return VideoMetadata(
                video_id=result.video_id,
                description=description,
                views=result.views,
                start_time=start,
                end_time=end,
                video_emb=embeddings.video[0].tolist(),
                audio_emb=embeddings.audio[0].tolist(),
                description_emb=embeddings.description[0].tolist(),
            )
        finally:
            download_path.close()
            if clip_path:
                clip_path.close()
    return None


async def search_and_embed_videos(query: str, num_videos: int, imagebind: ImageBind) -> List[VideoMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of VideoMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int): The number of videos to return.
        imagebind (ImageBind): The ImageBind instance for embedding.

    Returns:
        List[VideoMetadata]: A list of VideoMetadata objects representing the search results.
    """
    results = video_utils.search_videos(query, max_results=int(num_videos))
    video_metas = []
    try:
        # Process videos concurrently
        tasks = [process_video(query, result, imagebind) for result in results]
        video_metas = await asyncio.gather(*tasks)
        video_metas = [meta for meta in video_metas if meta is not None]
        video_metas = video_metas[:num_videos]  # Take the first N that we need
    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")
        # Log additional information
        bt.logging.error(f"Query: {query}")
        bt.logging.error(f"Number of videos: {num_videos}")
        bt.logging.error(f"Video results: {results}")
        # Raise the exception for further investigation
        raise

    return video_metas 