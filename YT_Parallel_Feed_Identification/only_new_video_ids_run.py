import requests
import pandas as pd
from datetime import datetime
import pytz
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import concurrent.futures
import time
import yt_dlp
import os

# Replace with your YouTube Data API key
API_KEY = 'AIzaSyCtnIoQfXiZSb0ske3CBUCGuvjznNVZaTI'

# Channel ID list
channel_ids = [
    'UCt4t-jeY85JegMlZ-E5UWtA',  # Aaj Tak
    'UCmphdqZNmqL72WJ2uyiNw5w',  # ABPLIVE
    'UCRWFSbif-RFENbBrSiez1DA',  # ABP NEWS
]

# File path to save video details between runs
SAVE_FILE_PATH = 'yt_dlp_live_videos_data_test.xlsx'

# Function to map channel ID to channel name
def get_channel_name(channel_id):
    """Map channel ID to the corresponding channel name."""
    channel_mapping = {
        'UCt4t-jeY85JegMlZ-E5UWtA': 'Aaj Tak',
        'UCmphdqZNmqL72WJ2uyiNw5w': 'ABPLIVE',
        'UCRWFSbif-RFENbBrSiez1DA': 'ABP NEWS',
    }
    return channel_mapping.get(channel_id, 'Unknown Channel')

# Function to fetch live video IDs with pagination handling
def get_live_video_ids_with_pagination(channel_id, api_key):
    """Fetch live video IDs for a specific YouTube channel using pagination."""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'channelId': channel_id,
        'eventType': 'live',
        'type': 'video',
        'maxResults': 50,
        'key': api_key
    }
    
    live_video_ids = []
    next_page_token = None
    retries = 3

    while True:
        if next_page_token:
            params['pageToken'] = next_page_token

        for attempt in range(retries):
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', []):
                        video_id = item['id']['videoId']
                        published_at = item['snippet']['publishedAt']
                        live_video_ids.append((video_id, published_at))

                    # Handle pagination
                    next_page_token = data.get('nextPageToken')
                    if not next_page_token:
                        return live_video_ids
                    break  # Exit retry loop on success

                elif response.status_code == 429:  # Too Many Requests
                    print(f"Too many requests for channel {channel_id}, retrying after delay...")
                    time.sleep(10)  # Wait before retrying

                else:
                    print(f"Error fetching live videos for channel {channel_id}: {response.text}")
                    return []
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)  # Wait before retrying in case of an error

    return live_video_ids

# Check if video is live
def check_if_video_is_live(video_id, api_key):
    """Check if the video is live by querying the 'videos' endpoint."""
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'snippet,liveStreamingDetails',
        'id': video_id,
        'key': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        for item in data.get('items', []):
            live_status = item.get('snippet', {}).get('liveBroadcastContent', None)
            if live_status == 'live':
                return True
    return False

# Convert UTC date to IST
def convert_to_ist(utc_datetime):
    """Convert a UTC datetime string to IST."""
    utc_time = datetime.strptime(utc_datetime, "%Y-%m-%dT%H:%M:%SZ")
    utc_time = utc_time.replace(tzinfo=pytz.UTC)
    ist_time = utc_time.astimezone(pytz.timezone('Asia/Kolkata'))
    return ist_time.strftime('%Y-%m-%d %H:%M:%S')

# Capture frame using yt-dlp
def stream_video_to_frame(video_id, retries=3):
    """Capture a frame from a YouTube live stream using yt-dlp and OpenCV."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {'format': 'best', 'quiet': True, 'noplaylist': True}

    for attempt in range(retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_url = info.get("url")

                # Capture stream using OpenCV
                cap = cv2.VideoCapture(video_url)
                if not cap.isOpened():
                    return None

                ret, frame = cap.read()
                cap.release()

                if ret:
                    return frame
        except Exception as e:
            time.sleep(10)  # Wait before retrying in case of an error
    return None

# Compare frames using SSIM and MSE
def compare_frames(frame1, frame2):
    """Compare two frames using SSIM and MSE."""
    min_height = min(frame1.shape[0], frame2.shape[0])
    min_width = min(frame1.shape[1], frame2.shape[1])

    frame1_resized = cv2.resize(frame1, (min_width, min_height))
    frame2_resized = cv2.resize(frame2, (min_width, min_height))

    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    mse = np.mean((gray1 - gray2) ** 2)
    return score, mse

# Check for new and deleted videos
def check_for_video_changes(new_videos, old_videos):
    """Identify new and deleted video IDs by comparing current and previous lists."""
    new_ids = set(new_videos) - set(old_videos)
    deleted_ids = set(old_videos) - set(new_videos)
    return new_ids, deleted_ids

# Fetch and mark primary video, capture frames, and compare
def fetch_and_compare_live_videos(channel_ids, api_key, first_run=False):
    # Check if a previous file exists
    if not first_run and os.path.exists(SAVE_FILE_PATH):
        old_df = pd.read_excel(SAVE_FILE_PATH)
    else:
        old_df = pd.DataFrame()

    # Fetch live videos
    live_videos_with_primary = []
    for channel_id in channel_ids:
        live_video_ids = get_live_video_ids_with_pagination(channel_id, api_key)
        for video_id, published_at in live_video_ids:
            live_videos_with_primary.append((channel_id, video_id, published_at))

    df = pd.DataFrame(live_videos_with_primary, columns=['Channel_ID', 'Video_ID', 'Published_Date'])
    df['Published_Date_IST'] = df['Published_Date'].apply(convert_to_ist)
    df['Channel_Name'] = df['Channel_ID'].apply(get_channel_name)

    # Check for new and deleted videos
    if not old_df.empty:
        new_ids, deleted_ids = check_for_video_changes(df['Video_ID'], old_df['Video_ID'])
        df['New_Video'] = df['Video_ID'].isin(new_ids)
        df['Deleted_Video'] = df['Video_ID'].isin(deleted_ids)
    else:
        df['New_Video'] = True  # First run: all videos are new
        df['Deleted_Video'] = False

    # Initialize comparison-related columns
    df['SSIM'] = None
    df['MSE'] = None
    df['Is_Parallel'] = None
    df['Exception'] = None

    # Perform frame capture and comparison only for new videos
    for index, row in df.iterrows():
        if row['New_Video']:
            try:
                primary_frame, compare_frame = stream_video_to_frame(row['Primary_ID']), stream_video_to_frame(row['Video_ID'])
                if primary_frame is not None and compare_frame is not None:
                    score, mse = compare_frames(primary_frame, compare_frame)
                    df.at[index, 'SSIM'] = score
                    df.at[index, 'MSE'] = mse
                    df.at[index, 'Is_Parallel'] = 'Yes' if score > 0.85 else 'No'
                else:
                    df.at[index, 'Exception'] = 'Frame capture failed'
            except Exception as e:
                df.at[index, 'Exception'] = str(e)

    # Save the updated DataFrame to an Excel file
    df.to_excel(SAVE_FILE_PATH, index=False)

# Main execution
if __name__ == "__main__":
    first_run = not os.path.exists(SAVE_FILE_PATH)  # Check if it's the first run
    fetch_and_compare_live_videos(channel_ids, API_KEY, first_run)
