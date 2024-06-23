from pytube import YouTube
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import ffmpeg
import json
import math
import time
import os

OPENAI_API_KEY = json.loads(open("secrets.json").read()).get("OPENAI_API_KEY", None)
NUM_CHUNKS = 16
CHUNKS_LIMIT = 64

if OPENAI_API_KEY is None:
    print("Please set the OPENAI_API_KEY environment variable.")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
URL = "https://www.youtube.com/watch?v=BTv5feWd9dk"


def download_video(url, filename):
    def on_progress(stream, chunk, bytes_remaining):
        percent = (1 - bytes_remaining / stream.filesize) * 100
        os.system("cls" if os.name == "nt" else "clear")
        print(f"{percent:.1f}% Downloaded")

    def on_complete(stream, file_handle):
        print("Download completed")

    print("Deleting previous segments...")
    os.system("del output-*.mp4" if os.name == "nt" else "rm output-*.mp4")
    if not os.path.exists(filename):
        os.system(
            "del input_video-*.mp4" if os.name == "nt" else "rm input_video-*.mp4"
        )
        yt = YouTube(
            url, on_progress_callback=on_progress, on_complete_callback=on_complete
        )
        # Download both audio and video streams, then merge them
        print("Downloading video file...")
        video = yt.streams.filter(
            file_extension="mp4", res="1080p", adaptive=True
        ).first()
        video.download(filename="inputV.mp4")

        print("Downloading audio file...")
        audio = yt.streams.filter(only_audio=True).first()
        audio.download(filename="inputA.mp4")
        print("Downloaded audio file")

        # Merge the audio and video files
        input_video = ffmpeg.input("inputV.mp4", hwaccel="auto")
        input_audio = ffmpeg.input("inputA.mp4", hwaccel="auto")

        output = ffmpeg.output(
            input_video,
            input_audio,
            filename,
            shortest=None,
            preset="ultrafast",
            deadline="realtime",
            vcodec="copy",
            acodec="aac",
            strict="experimental",
        )

        print("Merging audio and video files... This may take a while.")
        output.run(quiet=False)
        print("Merged audio and video files")
        os.remove("inputV.mp4")
        os.remove("inputA.mp4")
    else:
        print("Video file already exists. Skipping download.")


def segment_video(response, filename, i):
    start_time = math.floor(float(response.get("start_time", 0)))
    end_time = math.ceil(float(response.get("end_time", 0))) + 2
    output_file = f"output-{str(i).zfill(3)}.mp4"

    stream = (
        ffmpeg.input(filename, ss=start_time, to=end_time)
        .output(output_file, acodec="copy", vcodec="copy")
        .overwrite_output()
        .run(quiet=True)
    )


def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    formatted_transcript = ""
    for entry in transcript:
        start_time = "{:.2f}".format(entry["start"])
        end_time = "{:.2f}".format(entry["start"] + entry["duration"])
        text = entry["text"]
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

    return transcript


response_obj = """[
  {
    "start_time": 97.19, 
    "end_time": 127.43,
    "duration":36
  }
]"""


def analyze_transcript(transcript):
    prompt = f"This is a transcript of a video. Please identify 2 of the most viral sections from the whole, make sure they are more than 30 seconds in duration,Make Sure you provide extremely accurate timestamps respond only in this format, provide duration in seconds and do not add a description, {response_obj}  \n Here is the Transcription:\n{transcript}"
    messages = [
        {
            "role": "system",
            "content": "You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content",
        },
        {"role": "user", "content": prompt},
    ]
    print("Prompt size: ", len(prompt))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=512,
        stream=False,
        n=1,
        stop=None,
    )

    return response.choices[0].message


def final_analysis(segments):
    prompt = f"This is a transcript of a video. Identify 3 of the most viral sections from the whole, in this format: {response_obj}  \n Here is the Transcription:\n{segments}"
    messages = [
        {
            "role": "system",
            "content": "You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=512,
        stream=False,
        n=1,
        stop=None,
    )
    return response.choices[0].message


def main(url, num_chunks=4, chunk_limit=64):
    def split_array(arr, num_splits):
        length = len(arr)
        split_size = length // num_splits
        remainder = length % num_splits
        splits = []
        start_idx = 0

        for i in range(num_splits):
            end_idx = start_idx + split_size + (1 if i < remainder else 0)
            splits.append(arr[start_idx:end_idx])
            start_idx = end_idx

        return splits

    video_id = url.split("v=")[1]

    filename = f"input_video-{video_id}.mp4"
    download_video(url, filename)

    transcript = get_transcript(video_id)
    chunks = split_array(transcript, num_chunks)

    print("Total chunks: ", len(chunks))
    segments = []

    def process_chunks(errors, num_chunks, chunks, segments):
        for i, chunk in enumerate(chunks):
            print("Analyzing transcript with GPT-3...")
            print("Length of chunk:",len(str(chunk)))
            transcript = chunk

            try:
                interesting_segment = analyze_transcript(transcript)
                print(interesting_segment)
                content = interesting_segment.content
                try:

                    parsed_content = json.loads(content)
                except:
                    print("Error parsing JSON")
                    continue

                for seg in parsed_content:
                    for comp in transcript:
                        if (
                            comp["start"] == seg["start_time"]
                            and comp["duration"] == seg["duration"]
                        ):
                            seg["text"] = comp["text"]

                print(parsed_content)
                segments.append(parsed_content)

                os.system("cls" if os.name == "nt" else "clear")
                print("Throttling for 1 second...")
                print("Chunks remaining: ", len(chunks) - i)

                time.sleep(1)

            except Exception as e:
                print("Error analyzing transcript with GPT-3")
                print(e)
                errors += 1
                return errors
        return 0

    process_errors = process_chunks(0, num_chunks, chunks, segments)
    while process_errors > 0:
        if num_chunks % 2 == 0:
            increase = 2
        else:
            increase = 1
        num_chunks += increase
        if num_chunks > chunk_limit:
            print("Max number of chunks reached, exiting...")
            return
        print("Increasing number of chunks to", num_chunks)
        chunks = split_array(transcript, num_chunks)
        process_errors = process_chunks(process_errors, num_chunks, chunks, segments)

    # Final chatgpt review
    print("Final analysis with GPT-3...")
    segments = final_analysis(segments)
    content = segments.content
    try:
        print(content)
        segments = json.loads(content)
    except Exception as e:
        print("Error parsing final JSON, exiting...")
        print(e)
        return

    for i, parsed_content in enumerate(segments):
        segment_video(parsed_content, filename, i)
        
main(URL, NUM_CHUNKS, CHUNKS_LIMIT)
