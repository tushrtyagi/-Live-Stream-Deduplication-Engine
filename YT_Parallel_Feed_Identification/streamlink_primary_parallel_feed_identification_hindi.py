import requests
import pandas as pd
from datetime import datetime
import pytz
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import concurrent.futures
import time
import streamlink

# Replace with your YouTube Data API key
API_KEY = 'AIzaSyARhR9PCMuiNXdGTU9yA2sfA9EAJ7ZqMNE'

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

# Function to map channel ID to channel name
def get_channel_name(channel_id):
    """Map channel ID to the corresponding channel name."""
    channel_mapping = {
        'UCt4t-jeY85JegMlZ-E5UWtA': 'Aaj Tak',
        'UCmphdqZNmqL72WJ2uyiNw5w': 'ABPLIVE',
        'UCRWFSbif-RFENbBrSiez1DA': 'ABP NEWS',
        'UCttspZesZIDEwwpVIgoZtWQ': 'IndiaTV',
        'UC1tnj_v8Sn-hWERFvqSjBWQ': 'inKhabar',
        'UC9CYT9gSNLevX5ey2_6CK0Q': 'NDTV India',
        'UCx8Z14PpntdaxCt2hakbQLQ': 'The Lallantop',
        'UCPP3etACgdUWvizcES1dJ8Q': 'News18 India',
        'UCsNdeLwEZf86swPD3qJJ7Dw': 'News Nation',
        'UCMk9Tdc-d1BIcAFaSppiVkw': 'TIMES NOW Navbharat',
        'UCKwucPzHZ7zCUIf7If-Wo1g': 'DD News',
        'UCOutOIcn_oho8pyVN3Ng-Pg': 'TV9 Bharatvarsh',
        'UC7wXt18f2iA3EDXeqAVuKng': 'Republic Bharat',
        'UCjFKMoAk3qhRkW4eOqNm6dw': 'Zee Hindustan',
        'UCIvaYmXn910QMdemBG3v1pQ': 'Zee News',
        'UCuzS3rPQAYqHcLWqOFuY0pw': 'News24',
        'UCQC1wGbOOIoC23fRGxt4kbg': 'Good News Today',
        'UCcH1_Lw_VbQoxGgDJ2OW0LQ': 'India News',
        'UC_gUM8rL-Lrg6O3adPW9K1g': 'WION',
        'UCYPvAwZP8pZhSMW8qs7cVCw': 'India Today',
        'UCef1-8eOpJgud7szVPlZQAQ': 'CNN-News18',
        'UCwqusr8YDwM-3mEYTDeJHzw': 'Republic World',
        'UCZFMm1mMw0F81Z37aaEzTUA': 'NDTV',
        'UCWCEYVwSqr7Epo6sSCfUgiw': 'MIRROR NOW',
        'UC6RJ7-PaXg6TIH2BzZfTV7w': 'TIMES NOW',
        'UCytSP0M0Jdnw6qIy3Y-nTig': 'NewsX',
        'UCGDQNvybfDDeGTf4GtigXaw': 'DD INDIA',
        'UCSWSOS6YXUbNMzTH-tV7Pfw': 'Biz Tak',
        'UC3uJIdRFTGgLWrUziaHbzrg': 'BQ Prime',
        'UCQIycDaLsBpMKjOCeaKUYVg': 'CNBC Awaaz',
        'UCmRbHAgG2k2vDUvb3xsEunQ': 'CNBC-TV18',
        'UCI_mwTKUhicNzFrhm33MzBQ': 'ET NOW',
        'UChftTVI0QJmyXkajQYt2tiQ': 'moneycontrol',
        'UCkXopQ3ubd-rnXnStZqCl2w': 'Zee Business',
        'UCJEtD6JDgNKxcPJPBOAf-Tg': 'NDTV Profit',
        'UCaPHWiExfUWaKsUtENLCv5w': 'Business Today',
        'UCUI9vm69ZbAqRK3q3vKLWCQ': 'Mint',
        'UCQsob4fGjHWhYHW0OLb6rew': 'Business Standard',
        'UCmk6ZFMy1CT80orXca4tKew': 'The Financial Express',
        'UCuymYhTC78nLKDrA4z5FBSQ': 'CNBC Bajar',
        'UC5ebo42ydvAayGn2Z4Lf9XA': 'Uncut',
        'UCKsDvGQTUa8RoRARgOlQcGg': 'Astro Tak',
        'UCH6v_SxtFLtfD4Iptx2WbNg': 'MP Tak',
        'UCwPAS11WSOQvZh5RA36Wr-Q': 'Gujarat Tak',
        'UCYXd8ZkJGHy_beLGSk1OB2w': 'Life Tak',
        'UCPQLEr1W1R52H3dvxf9pXmw': 'Food Tak',
        'UCKy1T74_i4LgaZ02Tm8kBOA': 'Kids Tak - Nursery Rhymes & Kids Songs',
        'UC6ZMkiLxLEtz2P3x57nLfhg': 'Gaming Tak',
        'UCBLyYkSMFOpN0epIFSgCVKg': 'Cinema Tak',
        'UCmVsKnz_QcSRRqnjRP_BVdQ': 'Fit Tak',
        'UCnAp2J0bR9b8pM-Avp1GFOQ': 'Bihar Tak',
        'UCdfOu3dJAYCc99mUy8hyfMw': 'Aaj Tak Bangla',
        'UCSMSO8ITitwZ_fgIelqRSig': 'Dilli Tak',
        'UCpDxPj3sm40ISX5hn-TlYcw': 'Crime Tak',
        'UC2GRJOrY04aZjTmeBKr7nLA': 'Duniya Tak',
        'UCb1ScGnYiuIlc8AT5if67hg': 'Rajasthan Tak',
        'UCAUNFgpgVisKPL3yq_-Nj-Q': 'News Tak',
        'UCQ2W9GCHctD59vQpvLKVg3g': 'Mumbai Tak',
        'UCVXCo0W9pk2dDkEBNLhTt7A': 'Sports Tak',
        'UCZINqCwuTRertJyDvHUSopA': 'Bharat Tak',
        'UCJ-F1ElALMfXS74xaWja8jg': 'Haryana Tak',
        'UC_o9J_Ru7Ag4F-29C20x7ew': 'Fiiber Hindi',
        'UCskG03x3CoEW9W7s2c3IgbA': 'UP Tak',
        'UCUUVTcKl6MTdT5dqyUMrJrw': 'Jobs Tak',
        'UC6c1617Hv9xPoV2XFiA7gGw': 'Sahitya Tak',
        'UCz8QaiQxApLq8sLNcszYyJw': 'Firstpost',
        'UCENc-ImCfgkqHbCWIDhMkwg': 'News18 Debate & Interview',
        'UC3prwMn9aU2z5Y158ZdGyyA': 'CRUX',
        'UCrpEPrZCzOBIpz4XRMEaUAQ': 'India Today Conclave',
        'UCones2mLcMbwCsB2lEfJ_jA': 'Chunav Aaj Tak',
        'UC1NF71EwP41VdjAU1iXdLkw': 'Narendra Modi',
        'lW7Y3gS2SwRui9mzLv50Z1DxmNw': 'Rahul Gandhi',
        'UCB5C3oyks1QkFdKIN-7WopQ': 'India TV Aap Ki Adalat',
        'UCsXNGiYbKI21UrmnsFvSZCQ': 'Zee News HD',
        'UC-1UNkEZ_jpP3gF0gztLyCQ': 'News Nation Digital',
        'UCLKz0mTBW0ZKItzKqEgiKxA': 'TV9 Hindi News',
        'UCZbNi5DqBvaOZygMl8HfjDQ': 'Sports Today'
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

# Cross-check if a video is live using the 'videos' endpoint
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

# Fetch live video IDs for each channel and cross-check them
def fetch_live_videos_for_channels(channel_ids, api_key):
    all_live_videos = []

    for channel_id in channel_ids:
        print(f"Fetching live video IDs for channel: {channel_id}")
        live_video_ids = get_live_video_ids_with_pagination(channel_id, api_key)
        print(f"Found {len(live_video_ids)} live video IDs for channel {channel_id}")

        # Cross-check live video status
        confirmed_live_videos = []
        for video_id, published_at in live_video_ids:
            if check_if_video_is_live(video_id, api_key):
                print(f"Video {video_id} is confirmed as live.")
                confirmed_live_videos.append((video_id, published_at))

        all_live_videos.extend([(channel_id, video_id, published_at) for video_id, published_at in confirmed_live_videos])

    return all_live_videos

# Capture a frame from a live video using Streamlink and OpenCV
def stream_video_to_frame(video_id, retries=3):
    """Capture a frame from a YouTube live stream using Streamlink and OpenCV."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    for attempt in range(retries):
        print(f"Capturing frame from: {url} (Attempt {attempt + 1}/{retries})")
        try:
            streams = streamlink.streams(url)
            stream_url = streams['best'].url  # Select the best quality stream

            # Capture stream using OpenCV
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print(f"Failed to open stream: {url}")
                return None

            ret, frame = cap.read()
            cap.release()

            if ret:
                print(f"Successfully captured frame from: {url}")
                return frame  # Return the captured frame
            else:
                print(f"Failed to capture frame from: {url}")
                return None

        except Exception as e:
            print(f"Error capturing frame from {url}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Failed to capture frame after {retries} attempts.")
                return None

# Compare two frames using SSIM and MSE
def compare_frames(frame1, frame2):
    """Compare two frames using SSIM and MSE."""
    # Get the smaller dimensions
    min_height = min(frame1.shape[0], frame2.shape[0])
    min_width = min(frame1.shape[1], frame2.shape[1])

    # Resize both frames
    frame1_resized = cv2.resize(frame1, (min_width, min_height))
    frame2_resized = cv2.resize(frame2, (min_width, min_height))

    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    mse = np.mean((gray1 - gray2) ** 2)
    print(f"SSIM score: {score}, MSE: {mse}")
    return score, mse

# Function to perform parallel frame capture for both primary and comparison video
def capture_frames_parallel(primary_video_id, compare_video_id):
    """Capture frames from both the primary video and compare video in parallel."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for capturing frames in parallel
        primary_frame_future = executor.submit(stream_video_to_frame, primary_video_id)
        compare_frame_future = executor.submit(stream_video_to_frame, compare_video_id)

        # Wait for the frames to be captured
        primary_frame = primary_frame_future.result()
        compare_frame = compare_frame_future.result()

        return primary_frame, compare_frame

# Fetch and mark primary video, capture frames, and compare
def fetch_and_compare_live_videos(channel_ids, api_key):
    # Fetch live videos and mark the oldest one as primary
    live_videos_with_primary = fetch_live_videos_for_channels(channel_ids, api_key)

    if live_videos_with_primary:
        print("Final list of live videos with primary marked:")
        df = pd.DataFrame(live_videos_with_primary, columns=['Channel_ID', 'Video_ID', 'Published_Date'])
        df['Published_Date_IST'] = df['Published_Date'].apply(convert_to_ist)

        # Add the channel names to the DataFrame
        df['Channel_Name'] = df['Channel_ID'].apply(get_channel_name)

        # Find the oldest video and mark it as the primary video
        primary_videos = df.loc[df.groupby('Channel_ID')['Published_Date'].idxmin()].copy()
        primary_videos['Primary_ID'] = primary_videos['Video_ID']
        df = df.merge(primary_videos[['Channel_ID', 'Primary_ID']], on='Channel_ID', how='left')

        # Initialize columns for SSIM, MSE, and Parallel Status
        df['SSIM'] = None
        df['MSE'] = None
        df['Is_Parallel'] = None

        print(df)

        # Perform comparison of frames in parallel for each video
        for index, row in df.iterrows():
            primary_video_id = row['Primary_ID']
            video_id = row['Video_ID']

            if video_id != primary_video_id:
                print(f"Comparing primary video {primary_video_id} with video {video_id}")
                
                # Capture frames from both primary and comparison videos in parallel
                primary_frame, compare_frame = capture_frames_parallel(primary_video_id, video_id)
                
                # Compare frames if both were successfully captured
                if primary_frame is not None and compare_frame is not None:
                    score, mse = compare_frames(primary_frame, compare_frame)
                    df.at[index, 'SSIM'] = score
                    df.at[index, 'MSE'] = mse
                    if score > 0.85:
                        df.at[index, 'Is_Parallel'] = 'Yes'
                        print(f"Video {video_id} from {row['Channel_Name']} is a parallel feed (SSIM > 0.85)")
                    else:
                        df.at[index, 'Is_Parallel'] = 'No'
                        print(f"Video {video_id} from {row['Channel_Name']} is not a parallel feed (SSIM <= 0.85)")
                else:
                    print(f"Skipping comparison for video {video_id} from {row['Channel_Name']} due to frame capture failure.")

        # Save the results to an Excel file
        df.to_excel('live_videos_with_primary_and_comparison.xlsx', index=False)
        print("Saved live video details with primary and comparison to 'live_videos_with_primary_and_comparison.xlsx'.")
    else:
        print("No live videos found.")

# Main execution
if __name__ == "__main__":
    fetch_and_compare_live_videos(channel_ids, API_KEY)
