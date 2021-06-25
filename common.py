import importlib
import cv2
import numpy as np


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background




def load_cfg(cfg):
    mod_cfg = importlib.import_module(cfg.replace('/', '.')[:-3]) # the surffix .py is removed

    # for local mode, we'll have to reload this module, otherwise only
    # the 1st configure file will be used again and again
    #
    importlib.reload(mod_cfg)
    return mod_cfg.Cfg()


def update_if_not_none(v, v0):
    return (v if v is not None else v0)
