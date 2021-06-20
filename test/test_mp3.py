import sys
from pygame import mixer  # Load the popular external library

fin = sys.argv[1]
mixer.init()
mixer.music.load(fin)
mixer.music.play()
