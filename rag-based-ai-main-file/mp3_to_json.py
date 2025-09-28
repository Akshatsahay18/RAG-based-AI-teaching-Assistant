import whisper
import json
import os

# Load the Whisper model (large-v2 is one of the most accurate but also heavy)
model = whisper.load_model("large-v2")

# Get all file names from the "audios" folder
audios = os.listdir("audios")

# Iterate over every audio file in the folder
for audio in audios: 
    # Check if the filename contains "_" (to separate number and title)
    if("_" in audio):
        # Extract "number" from the filename before "_"
        number = audio.split("_")[0]
        
        # Extract "title" from the filename after "_", removing ".mp3" (last 4 chars)
        title = audio.split("_")[1][:-4]
        
        print(number, title)  # Debug: show which file is being processed

        # Transcribe and translate audio into English
        result = model.transcribe(
            audio = f"audios/{audio}",  # Path to the audio file
            language="hi",              # Original language is Hindi
            task="translate",           # Translate to English while transcribing
            word_timestamps=False       # Don't include timestamps for individual words
        )
        
        chunks = []  # Store segment-wise transcriptions with metadata
        
        # Loop through each segment (Whisper breaks audio into smaller parts)
        for segment in result["segments"]:
            chunks.append({
                "number": number,         # The extracted number from filename
                "title": title,           # The extracted title from filename
                "start": segment["start"],# Segment start time in seconds
                "end": segment["end"],    # Segment end time in seconds
                "text": segment["text"]   # Transcribed + translated text
            })
        """[
  {"number": "01", "title": "Intro", "start": 0.0, "end": 3.5, "text": "Hello"},
  {"number": "01", "title": "Intro", "start": 3.5, "end": 7.0, "text": "How are you?"}],"""

        # Store both segment chunks and full transcription text
        chunks_with_metadata = {
            "chunks": chunks,            # List of segments with timings
            "text": result["text"]       # Entire transcribed text (full string)
        }
        """chunkswithmetadata looks like {---act like 2d vector
  "chunks": [{----act like 1d vector which insert its o/p repetedly in 2d vec
      "number": "01","title": "Intro", "start": 0.0,"end": 3.5,"text": "Hello everyone"}, {
      "number": "01", "title": "Intro","start": 3.5, "end": 7.0,"text": "Welcome back to the session" }
      ],
  "text": "Hello everyone Welcome back to the session"
}
chunks:[ "number": "02","title": "Intro", "start": 0.0,"end": 3.5,"text": "Hello everyone"}, {
      "number": "02", "title": "Intro","start": 3.5, "end": 7.0,"text": "Welcome back to the session" }
],text="------------------------------" and so on same
        """

        # Save result as JSON in "jsons" folder with same filename + .json
        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)

"""done for every chunks_with_metadata for eg first 01_Intro.mp3 conv into jsons/01_Intro.mp3.json
{
  "chunks": [
    {"number": "01", "title": "Intro", "start": 0.0, "end": 3.5, "text": "Hello everyone"},
    {"number": "01", "title": "Intro", "start": 3.5, "end": 7.0, "text": "Welcome back"}
  ],
  "text": "Hello everyone Welcome back"
}converted into 
{"chunks": [{"number": "01", "title": "Intro", "start": 0.0, "end": 3.5, "text": "Hello everyone"}, {"number": "01", "title": "Intro", "start": 3.5, "end": 7.0, "text": "Welcome back"}], "text": "Hello everyone Welcome back"}
this is done for every chunk with diffrent number and title in chunks_with_metadata
"""