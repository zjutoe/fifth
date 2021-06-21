import subprocess
import cv2
from fifth.utils import DebugOn, D, I


def play_video(video, title = 'video', loop = False):
    D(f'play video {video}')
    
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            if loop:
                D(f'rewind')
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                D('exit on end of input')
                break
        
        cv2.imshow(title, image)
        
        if cv2.waitKey(5) & 0xFF == 27: # exit on key Esc
            D('exit on Esc')
            break

    cap.release()


    
def play_video_proc(source):
    # proc = multiprocessing.Process(target=procVideoShow, args=(source,))
    proc = multiprocessing.Process(target=loop_video, args=(source,))
    proc.start()

    # time.sleep(10)
    # Terminate the process
    # proc.terminate()  # sends a SIGTERM

    return proc
    


def image_show_scaled(image, win_name):
    h, w, _ = image.shape
    image = cv2.resize(image, (int(w/2), int(h/2)))
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 600,600)
    cv2.imshow(win_name, image)


def play_video_vlc(video):
    return subprocess.Popen(['/Applications/VLC.app/Contents/MacOS/VLC',
                             video,
                             '-f',])
