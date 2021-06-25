from fifth import configurebase

class Cfg(configurebase.ConfigureBase):
    def __init__(self):
        super(Cfg, self).__init__()

        self.video_input = video_input = 1
        self.threshold = threshold = 9

        self.playlist = [

            {'keyframes'   :'motions/m0',
             'reference'   :None,
             'reference2'  :None,
             'video_input' :video_input,
             'threshold'   :5,
            },
            
            {'keyframes'   :'motions/m1',
             'reference'   :'motions/reference.MP4',
             'reference2'  :'motions/reference_again.MP4',
             'video_input' :video_input,
             'threshold'   :threshold,
            },
            
            # {'keyframes'   :'motions/m2',
            #  'reference'   :'motions/m2/reference.MP4',
            #  'reference2'  :'motions/m2/reference_again.MP4',
            #  'video_input' :self.video_input,
            #  'threshold'   :self.threshold,
            # },
            
            {'keyframes'   :'motions/m3',
             'reference'   :'motions/m3/reference.MP4',
             'reference2'  :'motions/m3/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :self.threshold,
            },

            {'keyframes'   :'motions/m4',
             'reference'   :'motions/m4/reference.MP4',
             'reference2'  :'motions/m4/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :self.threshold,
            },

            {'keyframes'   :'motions/m5',
             'reference'   :'motions/m5/reference.MP4',
             'reference2'  :'motions/m5/reference_again.MP4',
             'video_input' :self.video_input,
             'threshold'   :self.threshold,
            },
        ]
        
