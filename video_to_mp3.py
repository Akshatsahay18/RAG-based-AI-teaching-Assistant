# Converts the videos to mp3 
import os #lets you interact with the operating system (listing folders, checking file paths, making directories, etc.).
import subprocess #lets you run external commands (like ffmpeg) directly from Python.

# 1. List all files in the "videos" folder
files = os.listdir("videos") 

# 2. Loop through each file in the folder
for file in files: 
    # Extract the tutorial number from the filename
    tutorial_number = file.split(" [")[0].split(" #")[1]

    # Extract the file name before " ｜ " (Japanese pipe symbol)
    file_name = file.split(" ｜ ")[0]

    # Debug print to check what we extracted
    print(tutorial_number, file_name)

    # 3. Run ffmpeg to convert the video into audio (mp3)
    subprocess.run([
        "ffmpeg",
        "-i", f"videos/{file}",                      # input video file
        f"audios/{tutorial_number}_{file_name}.mp3"  # output mp3 file
    ])
