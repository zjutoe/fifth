from fifth import configurebase
from fifth.common import update_if_not_none

class Cfg(configurebase.ConfigureBase):
    def __init__(self):
        super(Cfg, self).__init__()

        
    def init(self, video_input, threshold):
        self.video_input = update_if_not_none(video_input, self.video_input)
        self.threshold = update_if_not_none(threshold, self.threshold)

        self.playlist = [

            {'keyframes'   :'motions/m1',
             'reference'   :'motions/m1/reference.MP4',
             'reference2'  :'motions/m1/reference_again.MP4',
             'video_input' :video_input,
             'threshold'   :threshold,
             'timeout'     :400,
             'feedback_interval'   :3,
            },
            
        ]
        
