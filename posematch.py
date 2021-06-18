#!/usr/bin/env python

import click
import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic
import cv2
import mediapipe as mp
from fifth.utils import DebugOn, D, I

@click.group()
def execute():
    pass


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

mark1 = np.array(1000 * 33 * 4)
mark2 = np.array(1000 * 33 * 4)


def geo_distance(mark1, mark2):
    t = np.sqrt(((mark1 - mark2)**2).sum())
    return t


def anno_image_show(results, image):
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        return False

    return True


def pose_similar(kf_landmarks, video):
    # For video file input
    cap = cv2.VideoCapture(video)

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

                d = geo_distance(mark, kf_landmarks[i])
                D('geo_distance: %f', d)
                if d < 1.09:
                    I('keyframe %d matched', i)
                    i += 1
                    if i >= len(kf_landmarks):
                        I('all keyframes matched')
                        break

            if not anno_image_show(results, image):
                break

    cap.release()

    return mark


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


@execute.command()
@click.option('-k', '--keyframes', required=True, help='dir containing the keyframes')
@click.option('-i', '--video-input', default=None, help='input video')
@click.option('--debug', default=False, type=bool, help='debug')
def kf(keyframes, video_input, debug):
    if debug:
        DebugOn()

    with open(f'{keyframes}/.kflist') as kflst:
        kfs = [f'{keyframes}/{x}'[:-1] for x in kflst.readlines()]
        # print(kfs)
        mkf = load_keyframe_landmarks(kfs)

    cap = cv2.VideoCapture(video_input)
    pose_similar(mkf, video_input)


if __name__ == '__main__':
    execute()
