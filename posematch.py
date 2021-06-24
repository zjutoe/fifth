#!/usr/bin/env python

import click
import time
import threading
import multiprocessing
import subprocess

import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic
import cv2
import mediapipe as mp
from fifth.utils import DebugOn, D, I
from fifth.common import overlay_transparent
from fifth.sound import play_mp3
from fifth.video import play_video, VideoGet, image_show_scaled

@click.group()
def execute():
    pass


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# mark1 = np.array(1000 * 33 * 4)
# mark2 = np.array(1000 * 33 * 4)


def geo_distance(len_shin, mark1, mark2):
    t = np.sqrt(((mark1 - mark2)**2).sum())
    t /= len_shin
    # D('geo_distance: %s - %s', str(mark1), str(mark2))
    return t


def anno_image(results, image):
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image
    


def landmark_remove_trivial(landmarks):
    mk = landmarks

    # remove face besides nose
    for i in range(1, 11):
        mk[i, 0] = 0
        mk[i, 1] = 0
        mk[i, 2] = 0
        mk[i, 3] = 0

    return mk



def landmark_normalize(len_shin, landmarks):
    mk = landmarks
    # return mk                   # debug

    # make the nose as the coordinate center
    nose = mk[0]
    x, y, z = nose[0], nose[1], nose[2]
    mk[:, 0] -= x
    mk[:, 1] -= y
    mk[:, 2] -= z

    # scale size
    slen = np.sqrt(np.sum((mk[27] - mk[25])**2))
    scale = len_shin / slen
    mk *= scale

    mk = landmark_remove_trivial(mk)

    return mk



def landmark_norm(landmarks):
    mk = landmarks
    len_shin = np.sqrt(np.sum((mk[27] - mk[25])**2))
    
    # return mk                   # debug

    # make the nose as the coordinate center
    nose = mk[0]
    x, y, z = nose[0], nose[1], nose[2]
    mk[:, 0] -= x
    mk[:, 1] -= y
    mk[:, 2] -= z

    # scale size
    slen = np.sqrt(np.sum((mk[27] - mk[25])**2))
    scale = len_shin / slen
    mk *= scale

    mk = landmark_remove_trivial(mk)

    return mk



def pose_match(landmarks, kf_landmarks):
    results = landmarks
    


def pose_similar(kf_landmarks, geo_dist = 10, source = 0, reference = None):
    D(f'pose_similar({kf_landmarks}, {geo_dist}, {source}, {reference})')
    success = False

    # normalize the keyframes
    std_kf = kf_landmarks[0]
    len_shin = np.sqrt(np.sum((std_kf[27] - std_kf[25])**2))
    for kf in kf_landmarks:
        landmark_normalize(len_shin, kf)

    cap = cv2.VideoCapture(source)
    ref = cv2.VideoCapture(reference) if reference else None

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        mark = np.zeros(33 * 4).reshape(33, 4)
        i = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                I("Ignoring empty camera frame.")
                continue

            if not ref.isOpened():
                break
            ref_success, ref_image = ref.read()
            while not ref_success:
                D("reference end frame")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ref_success, ref_image = ref.read()

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            if results.pose_landmarks:
                for idx, landmark in enumerate(
                        results.pose_landmarks.landmark):
                    # print(idx, landmark)
                    mark[idx][0] = landmark.x
                    mark[idx][1] = landmark.y
                    mark[idx][2] = landmark.z
                    mark[idx][3] = landmark.visibility

                landmark_normalize(len_shin, mark) # FIXME: get the users' shin 

                d = geo_distance(len_shin, mark, kf_landmarks[i])
                D('geo_distance: %f', d)
                if d < geo_dist:
                    I('keyframe %d matched', i)
                    i += 1
                    if i >= len(kf_landmarks):
                        I('all keyframes matched')
                        success = True
                        break

            image = anno_image(results, image)
            h, w, _ = ref_image.shape
            image = overlay_transparent(ref_image, image, int(w/2), int(h/2))
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                return False

    cap.release()
    return success


def load_keyframe_landmarks(keyframes):
    marks = np.zeros(len(keyframes) * 33 * 4).reshape(len(keyframes), 33, 4)

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5) as pose:
        for i, file in enumerate(keyframes):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue

            D(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            )
            for j, landmark in enumerate(results.pose_landmarks.landmark):
                # print(idx, landmark)
                marks[i][j][0] = landmark.x
                marks[i][j][1] = landmark.y
                marks[i][j][2] = landmark.z
                marks[i][j][3] = landmark.visibility

            # Draw pose landmarks on the image.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(
                '/tmp/annotated_image' +
                str(i) +
                '.png',
                annotated_image)

    return marks





def playback_video(video):
    cap = cv2.VideoCapture(video)
    # cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


   

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

        
    def start(self):
        threading.Thread(target=self.show, args=()).start()
        return self
    

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
        


def threadVideoShow(source = 0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    # cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break

        # frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        # cps.increment()


        
class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


    def run(self):
        cap = cv2.VideoCapture(source)
        (grabbed, frame) = cap.read()
        video_shower = VideoShow(frame).start()
        cps = CountsPerSec().start()

        while not self.stopped():
            (grabbed, frame) = cap.read()
            if not grabbed or video_shower.stopped:
                video_shower.stop()
                break

            frame = putIterationsPerSec(frame, cps.countsPerSec())
            video_shower.frame = frame
            cps.increment()


            
def procVideoShow(source = 0):
    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            cv2.imshow("Video", frame)




def read_video_capture(cap, is_cam):
    if not cap.isOpened():
        return False, None

    success, image = cap.read()
    if not success:
        if is_cam:
            I('Ignoring empty camera frame')
            return True, None
        else:
            I('video file ends')
            return False, None

    return success, image


def landmarks_to_array(landmark, a):
    for i, mk in enumerate(landmark):
        a[i][0] = mk.x
        a[i][1] = mk.y
        a[i][2] = mk.z
        a[i][3] = mk.visibility


def calc_shin_len(landmarks):
    return np.sqrt(np.sum((landmarks[27] - landmarks[25])**2))


def compare_keyframes(landmark, keyframes, i_kf, len_shin, threshold):
    # get the shin length
    l_shin = calc_shin_len(landmark)
    # the len_shin is updated and get more accurate
    if len_shin < l_shin:
        len_shin = l_shin
        
    landmark = landmark_normalize(len_shin, landmark)

    d = geo_distance(len_shin, landmark, keyframes)
    D('distance: %f', d)
    if d <= threshold:
        I('keyframe %d matched', i_kf)
        i_kf += 1
        if i_kf >= len(keyframe):
            D('all keyframes matched')
            return True, None

    return False, i_kf, len_shin



def match_video_with_keyframes(video, keyframes, threshold = 10, video_reference = None):
    D(f'match_video_with_keyframes({video}, keyframes, {threshold}, {video_reference}')

    cap = cv2.VideoCapture(video)
    # ref = VideoGet(video_reference).start() if video_reference else None
    
    # ref = cv2.VideoCapture(video_reference) if video_reference else None
    # proc_ref = subprocess.Popen(['/Applications/VLC.app/Contents/MacOS/VLC', video_reference])
    proc_ref = subprocess.Popen(['ffplay', video_reference, '-fs'])
    # proc_ref = play_video_vlc(video_reference)    

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        len_shin = 0

        i_kf = 0
        mark = np.zeros(33 * 4).reshape(33, 4)
        while True:
            success, image = read_video_capture(cap, type(video) is int)
            if not success:     # error happens
                break

            # if ref:
            #     ref_success, ref_image = read_video_capture(ref, False)
            #     if not ref_success:
            #         D("rewind reference video")
            #         ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            #         ref_success, ref_image = cap.read()
                    
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if results.pose_landmarks:
                # there are 33 landmarks
                landmarks_to_array(results.pose_landmarks.landmark, mark)
                match, i_kf, len_shin = compare_keyframes(mark, keyframes, i_kf, len_shin, threshold)
                if i_kf is None:
                    # all keyframes matched
                    proc_ref.terminate()
                    return True

                image = anno_image(results, image)
                image_show_scaled(image, 'MediaPipe')
                # cv2.imshow('reference', ref.frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    # ref.stop()
                    return False                

    cap.release()
    # ref.release()
    return False

    

@execute.command()
@click.option('-k', '--keyframes', required=True, help='dir containing the keyframes')
@click.option('-r', '--reference', default=None, help='the reference video file')
@click.option('-i', '--video-input', default=0, help='input video, 0 (the webcam) by default')
@click.option('-t', '--threshold', default=1.1, help='geometry distance threshold')
@click.option('-p', '--video-pass', default=None, help='pass video')
@click.option('--debug', default=False, type=bool, help='debug')
def kf(keyframes, reference, video_input, threshold, video_pass, debug):
    if debug:
        DebugOn()

    with open(f'{keyframes}/.kflist') as kflst:
        kfs = [f'{keyframes}/{x}'[:-1] for x in kflst.readlines()]
        # print(kfs)
        mkf = load_keyframe_landmarks(kfs)

        # if reference:
        #     # playback_video(reference)
        #     pb = play_video_proc(reference)

        sim = False
        # while not sim:
        # sim = pose_similar(mkf, geo_dist, video_input, reference)
        sim = match_video_with_keyframes(video_input, mkf, threshold, reference)

        # now pass
        # play the trumpian sound
        play_mp3('tmp/mofashidai.mp3')

        playback_video(video_pass)

            
            
@execute.command()
def test():
    # threadVideoShow()
    # a0 = play_video_proc(0)
    # a1 = play_video_proc('1.MP4')

    # time.sleep(10)
    # a0.terminate()
    # a1.terminate()

    keyframes = 'motions/m0'
    with open(f'{keyframes}/.kflist') as kflst:
        kfs = [f'{keyframes}/{x}'[:-1] for x in kflst.readlines()]
        # print(kfs)
        mkf = load_keyframe_landmarks(kfs)
        ok = match_video_with_keyframes(0, mkf, 5, 'motions/m0/reference.MP4')


    # play_video(1)
    
    

if __name__ == '__main__':
    execute()
