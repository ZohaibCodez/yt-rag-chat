url_or_id = "https://www.youtube.com/watch?v=Lph_zAy1Z8U&list=PLjiDjYkaJ-r62vqrmfGlvy03iqlqiNw7t&index=19"
def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from YouTube URL or return ID if already provided"""
    if 'youtube.com/watch?v=' in url_or_id:
        return url_or_id.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url_or_id:
        return url_or_id.split('youtu.be/')[1].split('?')[0]
    else:
        return url_or_id  # Assume it's already an ID

print(extract_video_id(url_or_id))  # Output: RqTEHSBrYFw