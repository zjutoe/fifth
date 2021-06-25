from fifth import configurebase
from fifth.common import update_if_not_none

class Cfg(configurebase.ConfigureBase):
    def __init__(self):
        super(Cfg, self).__init__()


    def init(self, video_input, threshold):
        self.video_input = update_if_not_none(video_input, self.video_input)
        self.threshold = update_if_not_none(threshold, self.threshold)

        self.playlist = [

            {'keyframes'   :'motions/m0',
             'reference'   :'motions/m0/reference.MP4',
             'reference2'   :'motions/m0/reference.MP4',
             'video_input' :video_input,
             'threshold'   :2,
             'timeout'     :62,
             'feedback_interval'   :1,
            },
            
            
            {'keyframes'   :'motions/m1',
             'reference'   :'motions/m1/reference.MP4',
             'reference2'  :'motions/m1/reference_again.MP4',
             'video_input' :video_input,
             'threshold'   :2,
             'timeout': 25,
             'feedback_interval'   :3,
            },
            
            {'keyframes'   :'motions/m2',
             'reference'   :'motions/m2/reference.MP4',
             'reference2'  :'motions/m2/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :4,
             'timeout': 28,
             'feedback_interval'   :3,
            },
            
            {'keyframes'   :'motions/m4',
             'reference'   :'motions/m4/reference.MP4',
             'reference2'  :'motions/m4/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :2,
             'timeout': 23,
             'feedback_interval'   :3,                          
            },

            {'keyframes'   :'motions/m5',
             'reference'   :'motions/m5/reference.MP4',
             'reference2'  :'motions/m5/reference.MP4',
             'video_input' :self.video_input,
             'threshold'   :2,
             'timeout': 47,
             'feedback_interval'   :3,                                       
            },


            {'keyframes'   :'motions/m3',
             'reference'   :'motions/m3/reference.MP4',
             'reference2'  :'motions/m3/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   : 9,
             'timeout': 33,
             'feedback_interval'   :3,             
            },

            
        ]
        
