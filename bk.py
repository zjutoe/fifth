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



def calc_shin_len(landmarks):
    return np.sqrt(np.sum((landmarks[27] - landmarks[25])**2))



def landmark_remove_trivial(landmarks):
    mk = landmarks

    # remove face besides nose
    for i in range(1, 11):
        mk[i, 0] = 0
        mk[i, 1] = 0
        mk[i, 2] = 0
        mk[i, 3] = 0

    # remove palm
    for i in range(17, 23):
        mk[i, 0] = 0
        mk[i, 1] = 0
        mk[i, 2] = 0
        mk[i, 3] = 0

    # remove foot
    for i in range(29, 33):
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

    # mk = landmark_remove_trivial(mk)

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




def geo_distance(len_shin, mark1, mark2):
    t = np.sqrt(((mark1 - mark2)**2).sum())
    t /= len_shin
    # D('geo_distance: %s - %s', str(mark1), str(mark2))
    return t



def compare_keyframes(landmark, keyframes, i_kf, len_shin, len_shin_old, threshold):
    # get the shin length
    l_shin = calc_shin_len(landmark)
    # the len_shin is updated and get more accurate
    if len_shin < l_shin:
        D("len_shin updated %f %f", len_shin, l_shin)
        len_shin_old = len_shin
        len_shin = l_shin

    # len_shin not stable yet
    if len_shin_old == 0 or abs(len_shin_old - len_shin)/len_shin_old > 0.1:
        return False, i_kf, len_shin, len_shin_old
        
    landmark = landmark_normalize(len_shin, landmark)

    d = geo_distance(len_shin, landmark, keyframes)
    D('distance: %f', d)
    if d <= threshold:
        I('keyframe %d matched', i_kf)
        i_kf += 1
        if i_kf >= len(keyframes):
            D('all keyframes matched')
            return True, None, len_shin, len_shin_old

    return False, i_kf, len_shin, len_shin_old

            
def match_video_with_keyframes(video, keyframes, threshold = 10, video_reference = None):
    D(f'match_video_with_keyframes({video}, keyframes, {threshold}, {video_reference}')

    # normalize the keyframes
    std_kf = keyframes[0]
    len_shin = np.sqrt(np.sum((std_kf[27] - std_kf[25])**2))
    for kf in keyframes:
        landmark_normalize(len_shin, kf)

    cap = cv2.VideoCapture(video)
    # ref = VideoGet(video_reference).start() if video_reference else None
    
    # ref = cv2.VideoCapture(video_reference) if video_reference else None
    # proc_ref = subprocess.Popen(['/Applications/VLC.app/Contents/MacOS/VLC', video_reference])
    # proc_ref = subprocess.Popen(['ffplay', video_reference, '-fs'])
    # proc_ref = play_video_vlc(video_reference)
    
    if video_reference:
        proc_ref = subprocess.Popen(['ffplay', video_reference])


    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        len_shin = 0
        len_shin_old = 0

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
            
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)        
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            if results.pose_landmarks:
                # there are 33 landmarks
                landmarks_to_array(results.pose_landmarks.landmark, mark)

                match, i_kf, len_shin, len_shin_old = compare_keyframes(mark, keyframes, i_kf, len_shin, len_shin_old, threshold)
                
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

            if ref:
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
            if ref:
                h, w, _ = ref_image.shape
                image = overlay_transparent(ref_image, image, int(w/2), int(h/2))
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                return False

    cap.release()
    return success
    
