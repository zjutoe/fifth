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
             'threshold'   :0.05,
             'timeout'     :28800, # 8 hours
             'loop'        :True,
             'feedback_interval'   :1,
             'green_channel': False,
            },
            
            
            {'keyframes'   :'motions/m1',
             'reference'   :'motions/m1/reference.MP4',
             'reference2'  :'motions/m1/reference_again.MP4',
             'video_input' :video_input,
             'threshold'   :3,
             'timeout': 23,
             'loop'        :False,
             'feedback_interval'   :3,
             'green_channel': False,
            },
            
            {'keyframes'   :'motions/m2',
             'reference'   :'motions/m2/reference.MP4',
             'reference2'  :'motions/m2/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :4,
             'timeout': 28,
             'loop'        :False,
             'feedback_interval'   :3,
             'green_channel': False,
            },
            
            {'keyframes'   :'motions/m4',
             'reference'   :'motions/m4/reference.MP4',
             'reference2'  :'motions/m4/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :2,
             'timeout': 23,
             'loop'        :False,
             'feedback_interval'   :3,
             'green_channel': False,
            },

            {'keyframes'   :'motions/m5',
             'reference'   :'motions/m5/reference.MP4',
             'reference2'  :'motions/m5/reference.MP4',
             'video_input' :self.video_input,
             'threshold'   :1,
             'timeout': 47,
             'loop'        :False,
             'feedback_interval'   :5,
             'green_channel': False,
            },


            {'keyframes'   :'motions/m3',
             'reference'   :'motions/m3/reference.MP4',
             'reference2'  :'motions/m3/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :9,
             'timeout': 33,
             'loop'        :False,
             'feedback_interval'   :3,
             'green_channel': False,
            },

            {'keyframes'   :'motions/m6',
             'reference'   :'motions/m6/reference.MP4',
             'reference2'  :'motions/m6/reference.MP4',
             'video_input' :self.video_input,
             'threshold'   :4,
             'timeout': 136,
             'loop'        :False,
             'feedback_interval'   :2,
             'green_channel': True,
            },
            
        ]
        
