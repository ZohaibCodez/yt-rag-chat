from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled

video_id = "Gfr50f6ZBvo"
try:
  ytt_api = YouTubeTranscriptApi()
  transcript_list = ytt_api.fetch(video_id)
  print(transcript_list)
except TranscriptsDisabled:
  print("Transcripts are disabled for this video.")
except Exception as e:
  print(f"An error occurred: {e}")

transcript = " ".join(snippet.text for snippet in transcript_list)

with open("transcript.txt", "w") as f:
    f.write(transcript)