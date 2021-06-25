#!/usr/bin/env python

import click
import time
import threading

import subprocess
import importlib

import math
import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic
import cv2
import mediapipe as mp
from fifth.utils import DebugOn, D, I, E
from fifth.common import load_cfg, update_if_not_none
from fifth.sound import play_mp3
from fifth.video import play_video, VideoGet, image_show_scaled

@click.group()
def execute():
    pass


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# mark1 = np.array(1000 * 33 * 4)
# mark2 = np.array(1000 * 33 * 4)


def anno_image(results, image):
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image    



def pose_match(landmarks, kf_landmarks):
    results = landmarks
    


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



def angle_of_line(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    angle = math.atan2(y1-y2, x1-x2)
    return angle


def limbs_of_frame(f):
    lines = [
        (f[12], f[14]),       # right upper arm
        (f[14], f[16]),       # right lower arm
        (f[11], f[13]),       # left upper arm
        (f[13], f[15]),       # left lower arm

        (f[24], f[26]),       # right thigh
        (f[26], f[28]),       # right shin
        (f[23], f[25]),       # left thigh
        (f[25], f[27]),       # left shin
    ]

    return lines
    

def angles_of_limbs(limbs):
    return [ angle_of_line(*line) for line in limbs ]


def compare_frames_with_line_angles(f1, f2):
    limbs1 = limbs_of_frame(f1)
    limbs2 = limbs_of_frame(f2)

    angles1 = angles_of_limbs(limbs1)
    angles2 = angles_of_limbs(limbs2)
    dif = np.array(angles1) - np.array(angles2)
    dif = np.sum(dif**2)

    # D('angles1:%s', str(angles1))
    # D('angles2:%s', str(angles2))
    D('dif:%f', dif)

    return dif



def compare_video_with_keyframes(video, keyframes, threshold, feedback_interval,
                                 echo_play, timeout):
    ret = False
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        i = 0
        i_kf = 0
        mark = np.zeros(33 * 4).reshape(33, 4)
        while True:
            i += 1
            D('count %d before timeout %d', i, timeout)
            if i >= int(timeout): # FIXME timeout is str?
                break
                
            success, image = read_video_capture(cap, type(video) is int)
            if not success:     # error happens
                E('fail to capture video')
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if results.pose_landmarks:
                # there are 33 landmarks
                landmarks_to_array(results.pose_landmarks.landmark, mark)
                kf = keyframes[i_kf]
                dif = compare_frames_with_line_angles(mark, kf)
                if dif < threshold: # a match
                    D('matched keyframe %d', i_kf)
                    i_kf += 1
                    
                    if (i_kf+1) % feedback_interval == 0:
                        play_mp3('tmp/success.mp3')
                        
                    if i_kf >= len(keyframes): # all match
                        D('matched all keyframes')
                        ret = True
                        break
                else:
                    D('not matched keyframe %d', i_kf)

                if echo_play:
                    image = anno_image(results, image)
                    image_show_scaled(image, 'MediaPipe')
                                
                if cv2.waitKey(1) & 0xFF == 27:
                    # ref.stop()
                    break

    cap.release()
    return ret
    


# the previous video should be killed after a new one starts, that's
# why we pass it as argument
def play_scene(scene, echo_play = False, prev_proc_ref = None):
    I('play_scene: %s', str(scene))
    
    keyframes   = scene['keyframes']
    reference   = scene['reference']
    reference2  = scene['reference2']
    video_input = scene['video_input']
    threshold   = scene['threshold']
    timeout     = scene['timeout']
    feedback_interval     = scene['feedback_interval']
    
    with open(f'{keyframes}/.kflist') as kflst:
        kfs = [f'{keyframes}/{x}'[:-1] for x in kflst.readlines()]
        mkf = load_keyframe_landmarks(kfs)
            
        # proc_ref = subprocess.Popen(['ffplay', reference, '-fs', '-autoexit']) if reference else None
        proc_ref = subprocess.Popen(['ffplay', reference, '-fs']) if reference else None
        time.sleep(1)
        if prev_proc_ref: # the previous video should be killed after a new one starts
            I('killing last stage video')
            prev_proc_ref.terminate()

        succeed = compare_video_with_keyframes(video_input, mkf, threshold, feedback_interval, echo_play, timeout)
        while not succeed:
            play_mp3('tmp/fail.mp3')
            
            # start the 2nd video before killing the 1st one, to keep
            # the background desktop covered
            proc_ref2 = subprocess.Popen(['ffplay', reference2, '-fs']) if reference2 else None
            time.sleep(1)       # wait for the new video to start

            proc_ref.terminate()
            proc_ref = proc_ref2

            # try again
            succeed = compare_video_with_keyframes(video_input, mkf, threshold, feedback_interval, echo_play, timeout)


        # play_mp3('tmp/success.mp3')
        I('you pass!')
        return proc_ref




@execute.command()
@click.option('-C', '--configure', default='config.py', help='the configure file, config.py by default')
@click.option('-e', '--echo-play', default=False, help='echo play')
@click.option('-i', '--video-input', default=None, type=int, help='video input, None by default')
@click.option('-t', '--threshold', default=10, type=int, help='threshold, None by default')
@click.option('--debug', default=False, type=bool, help='debug')
def play(configure, echo_play, video_input, threshold, debug):
    if debug:
        DebugOn()
        
    cfg = load_cfg(configure)
    cfg.init(video_input, threshold)
    # cfg.video_input = update_if_not_none(video_input, cfg.video_input)

    proc_ref = None
    while True:
        for scene in cfg.playlist:
            proc_ref = play_scene(scene, echo_play, proc_ref)

    if proc_ref:
        proc_ref.terminate()
    
    
            
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
    
    



@execute.command()
@click.option('-k', '--keyframes', required=True, help='dir containing the keyframes')
@click.option('-r', '--reference', default=None, help='the reference video file')
@click.option('-R', '--reference2', default=None, help='the 2nd reference video file, with again title')
@click.option('-i', '--video-input', default=0, help='input video, 0 (the webcam) by default')
@click.option('-t', '--threshold', default=1.1, help='geometry distance threshold')
@click.option('-p', '--video-pass', default=None, help='pass video')
@click.option('-e', '--echo-play', default=False, help='echo play')
@click.option('--debug', default=False, type=bool, help='debug')
def kf(keyframes, reference, reference2, video_input, threshold, video_pass, echo_play, debug):
    if debug:
        DebugOn()

    with open(f'{keyframes}/.kflist') as kflst:
        kfs = [f'{keyframes}/{x}'[:-1] for x in kflst.readlines()]
        mkf = load_keyframe_landmarks(kfs)

        # proc_ref = subprocess.Popen(['ffplay', reference, '-fs', '-loop', '0']) if reference else None
        # proc_ref = subprocess.Popen(['ffplay', reference, '-fs']) if reference else None

        sim = compare_video_with_keyframes(video_input, mkf, threshold, echo_play, reference)
        while not sim:
            play_mp3('tmp/fail.mp3')
            sim = compare_video_with_keyframes(video_input, mkf, threshold, echo_play, reference2)

        play_mp3('tmp/success.mp3')


    
if __name__ == '__main__':
    execute()
