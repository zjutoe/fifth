import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from mediapipe.python.solutions import holistic as mp_holistic
import numpy as np

mark1 = np.array(1000 * 33 * 4)
mark2 = np.array(1000 * 33 * 4)


def geo_distance(mark1, mark2):
  t = np.sqrt(((mark1 - mark2)**2).sum())
  return t



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
        print("Ignoring empty camera frame.")
        break

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = pose.process(image)

      if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
          # print(idx, landmark)
          mark[idx][0] = landmark.x
          mark[idx][1] = landmark.y
          mark[idx][2] = landmark.z
          mark[idx][3] = landmark.visibility

        d = geo_distance(mark, kf_landmarks[i])
        # print('geo_distance:', d)
        if d < 1:
          print('keyframe {i} matched'.format(i=i))
          i += 1

          # Draw the pose annotation on the image.
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          mp_drawing.draw_landmarks(
              image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          cv2.imshow('MediaPipe Pose', image)
          if cv2.waitKey(5) & 0xFF == 27:
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
      
      print(
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
      cv2.imwrite('/tmp/annotated_image' + str(i) + '.png', annotated_image)

  return marks
  

# m1 = load_landmarks('1.MP4')

keyframes = ['pose1/pose1_0{i}.png'.format(i=i) for i in range(1, 9)]
print(keyframes)
mkf = load_keyframe_landmarks(keyframes)

cap = cv2.VideoCapture('1.MP4')
pose_similar(mkf, '1.MP4')

# # For video file input
# cap = cv2.VideoCapture('1.MP4')
# # cap = cv2.VideoCapture(0)
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = pose.process(image)
#     print('================')
#     print(results.pose_landmarks)

#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     cv2.imshow('MediaPipe Pose', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()
