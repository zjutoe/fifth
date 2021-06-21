import subprocess

def play_mp3(source):
    return subprocess.call(["afplay", source])
