import sys
from pygame import mixer  # Load the popular external library

fin = sys.argv[1]

# mixer.init()
# mixer.music.load(fin)
# mixer.music.play()

# import playsound
# playsound.playsound(fin, True)

import subprocess
# audio_file = "/full/path/to/audio.wav"

return_code = subprocess.call(["afplay", fin])

