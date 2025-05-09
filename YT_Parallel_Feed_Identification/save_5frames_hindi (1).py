import os
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

# Replace with your YouTube Data API key
API_KEY = 'AIzaSyARhR9PCMuiNXdGTU9yA2sfA9EAJ7ZqMNE'


timestap = datetime.now().strftime("%Y%m%d_%H%M%S")

# Directory to save frames
SAVE_DIR = f'Hindi_captured_frames_{timestap}'

# Ensure the directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Channel ID list
channel_ids = [
        'UCt4t-jeY85JegMlZ-E5UWtA', # 'Aaj Tak',
        'UCmphdqZNmqL72WJ2uyiNw5w', # 'ABPLIVE',
        'UCRWFSbif-RFENbBrSiez1DA', # 'ABP NEWS',
        'UCttspZesZIDEwwpVIgoZtWQ', # 'IndiaTV',
        # 'UC1tnj_v8Sn-hWERFvqSjBWQ', # 'inKhabar',
        'UC9CYT9gSNLevX5ey2_6CK0Q', # 'NDTV India',
        'UCx8Z14PpntdaxCt2hakbQLQ', # 'The Lallantop',
        'UCPP3etACgdUWvizcES1dJ8Q', # 'News18 India',
        'UCsNdeLwEZf86swPD3qJJ7Dw', # 'News Nation',
        'UCMk9Tdc-d1BIcAFaSppiVkw', # 'TIMES NOW Navbharat',
        'UCKwucPzHZ7zCUIf7If-Wo1g', # 'DD News',
        'UCOutOIcn_oho8pyVN3Ng-Pg', # 'TV9 Bharatvarsh',
        'UC7wXt18f2iA3EDXeqAVuKng', # 'Republic Bharat',
        # 'UCjFKMoAk3qhRkW4eOqNm6dw', # 'Zee Hindustan',
        'UCIvaYmXn910QMdemBG3v1pQ', # 'Zee News',
        'UCuzS3rPQAYqHcLWqOFuY0pw', # 'News24',
        'UCQC1wGbOOIoC23fRGxt4kbg', # 'Good News Today',
        'UCcH1_Lw_VbQoxGgDJ2OW0LQ', # 'India News',
]

# Function to save a frame in a single folder for each comparison
def save_frame(frame, channel_id, primary_video_id, compare_video_id, frame_index, frame_type):
    """Save frames from both videos in a single folder for each comparison."""
    folder_path = os.path.join(SAVE_DIR, f"{channel_id}_{primary_video_id}_vs_{compare_video_id}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the frame with a unique name indicating its source and index
    filename = f"{frame_type}_frame_{frame_index}.jpg"
    filepath = os.path.join(folder_path, filename)
    cv2.imwrite(filepath, frame)
    print(f"Frame saved: {filepath}")

# Function to get live video IDs with pagination
def get_live_video_ids_with_pagination(channel_id, api_key):
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

    while True:
        if next_page_token:
            params['pageToken'] = next_page_token

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('items', []):
                video_id = item['id']['videoId']
                published_at = item['snippet']['publishedAt']
                live_video_ids.append((video_id, published_at))

            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                return live_video_ids
        else:
            print(f"Error fetching live videos for channel {channel_id}: {response.text}")
            return []

    return live_video_ids

# Check if a video is live
def check_if_video_is_live(video_id, api_key):
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

# Capture multiple frames from a live video
def stream_video_to_frames(video_id, num_frames=5, retries=3):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'noplaylist': True,
    }

    for attempt in range(retries):
        print(f"Capturing {num_frames} frames from: {url} (Attempt {attempt + 1}/{retries})")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_url = info.get("url")

                if not video_url:
                    print(f"Failed to get stream URL for {url}")
                    return None

                cap = cv2.VideoCapture(video_url)
                if not cap.isOpened():
                    print(f"Failed to open stream: {url}")
                    return None

                frames = []
                for _ in range(num_frames):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        break

                cap.release()
                if len(frames) == num_frames:
                    print(f"Successfully captured {num_frames} frames from: {url}")
                    return frames
        except Exception as e:
            print(f"Error capturing frames from {url}: {e}")
            time.sleep(10)
    return None

# Capture multiple frames in parallel
def capture_frames_parallel(primary_video_id, compare_video_id, channel_id, num_frames=5):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        primary_frames_future = executor.submit(stream_video_to_frames, primary_video_id, num_frames)
        compare_frames_future = executor.submit(stream_video_to_frames, compare_video_id, num_frames)

        primary_frames = primary_frames_future.result()
        compare_frames = compare_frames_future.result()

        return primary_frames, compare_frames

# Compare multiple pairs of frames using SSIM and MSE
def compare_multiple_frames(primary_frames, compare_frames):
    ssim_scores = []
    mse_scores = []

    for frame1, frame2 in zip(primary_frames, compare_frames):
        min_height = min(frame1.shape[0], frame2.shape[0])
        min_width = min(frame1.shape[1], frame2.shape[1])

        frame1_resized = cv2.resize(frame1, (min_width, min_height))
        frame2_resized = cv2.resize(frame2, (min_width, min_height))

        gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(gray1, gray2, full=True)
        mse = np.mean((gray1 - gray2) ** 2)

        ssim_scores.append(score)
        mse_scores.append(mse)

    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    print(f"Average SSIM score: {avg_ssim}, Average MSE: {avg_mse}")
    return avg_ssim, avg_mse

# Fetch and compare live videos
def fetch_and_compare_live_videos(channel_ids, api_key):
    all_live_videos = []

    for channel_id in channel_ids:
        print(f"Fetching live video IDs for channel: {channel_id}")
        live_video_ids = get_live_video_ids_with_pagination(channel_id, api_key)
        print(f"Found {len(live_video_ids)} live video IDs for channel {channel_id}")

        confirmed_live_videos = []
        for video_id, published_at in live_video_ids:
            if check_if_video_is_live(video_id, api_key):
                print(f"Video {video_id} is confirmed as live.")
                confirmed_live_videos.append((video_id, published_at))

        all_live_videos.extend([(channel_id, video_id, published_at) for video_id, published_at in confirmed_live_videos])

    if all_live_videos:
        df = pd.DataFrame(all_live_videos, columns=['Channel_ID', 'Video_ID', 'Published_Date'])
        primary_videos = df.loc[df.groupby('Channel_ID')['Published_Date'].idxmin()].copy()
        primary_videos['Primary_ID'] = primary_videos['Video_ID']
        df = df.merge(primary_videos[['Channel_ID', 'Primary_ID']], on='Channel_ID', how='left')

        df['SSIM'] = None
        df['MSE'] = None
        df['Is_Parallel'] = None
        df['Exception'] = None

        for index, row in df.iterrows():
            primary_video_id = row['Primary_ID']
            video_id = row['Video_ID']
            channel_id = row['Channel_ID']

            if video_id != primary_video_id:
                print(f"Comparing primary video {primary_video_id} with video {video_id}")
                try:
                    primary_frames, compare_frames = capture_frames_parallel(primary_video_id, video_id, channel_id, num_frames=5)

                    if primary_frames is not None and compare_frames is not None:
                        # Save all frames (5 primary and 5 comparison) in a single folder
                        for idx, (primary_frame, compare_frame) in enumerate(zip(primary_frames, compare_frames)):
                            save_frame(primary_frame, channel_id, primary_video_id, video_id, idx, 'primary')
                            save_frame(compare_frame, channel_id, primary_video_id, video_id, idx, 'compare')

                        avg_ssim, avg_mse = compare_multiple_frames(primary_frames, compare_frames)
                        df.at[index, 'SSIM'] = avg_ssim
                        df.at[index, 'MSE'] = avg_mse
                        df.at[index, 'Is_Parallel'] = 'Yes' if avg_ssim > 0.65 else 'No'
                except Exception as e:
                    df.at[index, 'Exception'] = str(e)
                    print(f"Error comparing video {video_id}: {str(e)}")

        df.to_excel(f'Hindi_yt_dlp_live_videos_with_primary_and_comparison_{timestap}.xlsx', index=False)
        print(f"Saved live video details to Excel.")
    else:
        print("No live videos found.")

# Main execution
if __name__ == "__main__":
    fetch_and_compare_live_videos(channel_ids, API_KEY)