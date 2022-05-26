import os
from pathlib import Path
import unittest
import time
import cv2
from PIL import Image
import glob
import logging
image_list = []
filename = 'test_images/stop.jpg'

import Object_detection


# root_path = os.path.split(lane_detector.__file__)[0]
logger = logging.getLogger(__name__)


# def io_fpath_generator(in_rpath, out_rpath, root=root_path):
#     in_path = os.path.join(root, in_rpath)
#     out_path = os.path.join(os.path.join(root, in_rpath), out_rpath)
#     logger.debug(f'Reading / writing : {in_path} / {out_path}')
#     os.makedirs(out_path, exist_ok=True)
#     pathlist = Path(in_path).glob('**/*')
#     for in_fpath in pathlist:
#         if not os.path.isdir(in_fpath) and not os.path.split(in_fpath)[1].startswith('.'):
#             out_fpath = os.path.join(out_path, os.path.split(in_fpath)[1])
#             yield str(in_fpath), str(out_fpath)
# from Line_detection import detect_lanes

class Test_Obejct_detection(unittest.TestCase):
    def test_on_image(self):
        # in_path = os.path.join(filename, 'in')
        # out_path = os.path.join('out.png', 'test_out')
        # os.makedirs(out_path, exist_ok=True)  
        # pathlist = Path(in_path).glob('test_images/stop.jpg')
        # for in_fpath in pathlist:
        #     in_fpath = str(in_fpath)
        #     print('\n'+in_fpath)
        #     orig_img = cv2.imread(in_fpath)

        #     slope, left_visible, right_visible, labeled_img = \
        #         detect_lanes(orig_img, bottom_fraction_to_analyze=.5, return_marked_image=True)

        #     out_fpath = os.path.join(out_path, os.path.split(in_fpath)[1])
        #     cv2.imwrite(out_fpath, labeled_img)
        print("test")

if __name__ == '__main__':
    unittest.main()