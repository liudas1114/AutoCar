import logging
import math
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (255, 0, 0)

line_detection_length_thold = 60/1200.

# on an image with wider single lane-line where both edges of it are marked by micro-lines we will find 2 clusters
# for the left and right side of it - x1 clustering produces two medians - centers of clusters - following param
# defines a minimum distance required between these centres to switch to 2 lane-line regime
min_distance_between_clusters = 300/1600

# relating to outlier filtering
s_outlier_filter_param = 1.4
s_outlier_filter_require_to_remain = .7

# check if one of the clusters is formed from small number of noise lines: if total vertical size is
# relatively small one cluster versus another - one of them will be noise
# computed as absolute percentage difference between two y_sizes expressed as fraction
accept_clusters_below_ydist_fraction_diff = .6

class Detect_Lane():
    def k_filter_for_black(self, img, k_thold):
        # calculate CMYK - K channel
        # img = img.astype(np.float) / 255.
        k = 255 - np.max(img, axis=2)
        _, k = cv2.threshold(k, k_thold, 255, cv2.THRESH_BINARY)
        return k
    def __init__(self,
                car = None, 
                speed = 1,
                width = 640,
                height = 480):
        self.car = car
        self.speed = speed
        self.CAMERA_WIDTH = width
        self.CAMERA_HEIGHT = height
    def detect_lanes(self, orig_img, bottom_fraction_to_analyze=.5, return_marked_image=True) -> \
            Tuple[float, bool, bool, Optional[np.ndarray]]:
        """
        Takes prefiltered image and produces and average sliope between left and right lane-lines which can be used to
        maintain in-lane steering.

        Args:
            orig_img:
            bottom_fraction_to_analyze:
            return_marked_image:

        Returns:
            tuple(final_slope, left_visible, right_visible, labeled_img)
            final_slope - returns average slope between left and right lane-lines; slope = (y2 - y1) / (x2 - x1)
            left_visible, right_visible - flags on which lane has been detected - if unexpectedly one line disappeared
                                        slope result is less confident; it also can help to get into the lane at the start
            labeled_img - if return_marked_image=True returns image with left, right and micro-lines marked
        """
        img = orig_img.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        # crop top part of the image
        img = img[-int(h * bottom_fraction_to_analyze):, :]
        gray_img = img

        # FastLineDetector is opencv-contrib-python alternative to LineSegmentDetector implemented in core opencv-python
        # which seems to be crashing in 4.5.5 on both RPI and MacOS; LineSegmentDetector suggestion taken from comments
        # on the bottom of: https://stackoverflow.com/questions/41329665/linesegmentdetector-in-opencv-3-with-python
        line_algo = cv2.ximgproc.createFastLineDetector(length_threshold=int(line_detection_length_thold * h))
        lines = line_algo.detect(gray_img)
        assert lines is not None, 'FastLineDetector did not detect lines'

        # convert numpy array into pandas DF - working with columns as vectors is slightly more intuitive/readable versus
        # unlabeled numpy array
        ldf = np.squeeze(lines, axis=1)
        ldf = pd.DataFrame(ldf, columns=['x1', 'y1', 'x2', 'y2'])

        # Note: the origin, (0, 0), is located at the top-left of
        # the image. OpenCV images are zero-indexed, where the x-values go left-to-right (column number) and y-values
        # go top-to-bottom (row number)
        # slope=inf on vertical line
        ldf['slope'] = (ldf.y2 - ldf.y1) / (ldf.x2 - ldf.x1)

        cx_l, cx_r, sl_l, sl_r, ldff = self._find_lanelines(ldf, w, h)

        labeled_img = None
        if return_marked_image:
            # note that ldff has outlier lines removed, switch to ldf to see all lines detected
            labeled_img = self._label_image(gray_img, line_algo, ldff, cx_l, sl_l, cx_r, sl_r)

        left_visible, right_visible = cx_l is not None, cx_r is not None
        if left_visible and right_visible:
            final_slope = (sl_l + sl_r) / 2
            print(f"l: {sl_l} - r: {sl_r}")
        else:
            final_slope = sl_l or sl_r
        

        return labeled_img, cx_l, sl_l, cx_r, sl_r

    def _label_image(self,gray_img, line_algo, ldf, cx_l, sl_l, cx_r, sl_r):
        # first 4 columns contain coords in same order as originally returned by FastLineDetector
        out_lines = ldf.iloc[:, :4]
        out_lines = out_lines.values[:, np.newaxis, :]
        labeled_img = line_algo.drawSegments(gray_img, out_lines)
        # left lane-line
        if cx_l:
            self._line_for_centerx_and_slope(labeled_img, cx_l, sl_l, GREEN)
        # right lane-line
        if cx_r:
            self._line_for_centerx_and_slope(labeled_img, cx_r, sl_r, BLUE)
        return labeled_img


    def _filter_raw_line_outliers(self, ldf, h):
        res = ldf
        if len(ldf) > 3:
            # filter slope outliers - lines that diverge from median slope by X standard deviations
            slope = ldf.slope.clip(-h, h)  #  vertical line slope=inf, clip it to max non inf slope possible on
                                        # the image - 1px wide vertical
            ldff = ldf.loc[(slope.median() - slope).abs() < s_outlier_filter_param * slope.std()]
            # do not apply filtering if it's removing more than ~60% of cluster - std used in filter above
            res = ldff if len(ldff) / len(ldf) > s_outlier_filter_require_to_remain else ldf
        return res

    def _find_lanelines(self,ldf, w, h):
        """

        Returns: (cx_l, cx_r, sl_l, sl_r, ldff):
            cx_l, cx_r - center x of left and right lanelines
            sl_l, sl_r - slopes of --"--
            ldff - diagnostics dataframe, a copy of ldf arg after outlier filtering and other intermediate processing cols
        """
        small_sample = len(ldf) == 1
        if not small_sample:
            # try to locate 2 means of x coordinates
            clust_alg = KMeans(n_clusters=2)
            ldf['left_right_flag'] = clust_alg.fit_predict(ldf[['x1']])
            # ldf['left_right_t2'] = clust_alg.fit_predict(ldf[['x2']])  # there is probably information in x2 that we do not use
            # is error bigger when we use 2 clusters or one?
            ldf0 = ldf.loc[ldf['left_right_flag'] == 0]
            ldf1 = ldf.loc[ldf['left_right_flag'] == 1]

            # outlier removal in case of 2 clusters
            ldf0 = self._filter_raw_line_outliers(ldf0, h)
            ldf1 = self._filter_raw_line_outliers(ldf1, h)
            ldf_2c_f = pd.concat([ldf0, ldf1])

            # outlier removal in case of 1 cluster
            ldf_1c_f = self._filter_raw_line_outliers(ldf, h)

            # mean average error for x versus center-x
            ldf0_cx, ldf1_cx = ldf0.x1.median(), ldf1.x1.median()
            e0 = (ldf0_cx - ldf0.x1).abs().sum() / len(ldf0)
            e1 = (ldf1_cx - ldf1.x1).abs().sum() / len(ldf1)
            e_single_c = (ldf_1c_f.x1.median() - ldf_1c_f.x1).abs().sum() / len(ldf_1c_f)

            # computed as absolute percentage difference between two sizes expressed as fraction
            clusters_ydist_fraction_diff = abs(1 - self._ydist(ldf0) / self._ydist(ldf1))
        else:
            ldf_1c_f = ldf

        if not small_sample \
                and abs(ldf0_cx - ldf1_cx) > min_distance_between_clusters * w \
                and clusters_ydist_fraction_diff < accept_clusters_below_ydist_fraction_diff \
                and (e0 + e1) / 2 < e_single_c:  # is error bigger when we use 2 clusters or one?
            # using 2 centers
            ldff = ldf_2c_f
            grouped = ldf_2c_f.groupby('left_right_flag')
            # note that sorting by x classifies between left/right line
            cluster_stats_df = grouped['x1'].median().sort_values().to_frame(name='center_x')
            cluster_stats_df['slope'] = grouped['slope'].median()

            cx_l, cx_r = cluster_stats_df['center_x'].astype(int).values
            sl_l, sl_r = cluster_stats_df['slope'].values
        else:
            # using 1 center
            ldff = ldf_1c_f
            cx, sl = ((ldff.x1 + ldff.x2) / 2.).median(), ldff.slope.median()
            # note that logic below depends if we're on-the-road or off-the-road at the moment, - assuming on-the-road
            cx_l, sl_l, cx_r, sl_r = None, None, None, None
            if sl > 0:
                # line \ is on the right
                cx_r, sl_r = cx, sl
            else:
                # line / is on the left
                cx_l, sl_l = cx, sl
        return cx_l, cx_r, sl_l, sl_r, ldff

    def _ydist(self, ldf):
        ydist = max(ldf.y1.max(), ldf.y2.max()) - min(ldf.y1.min(), ldf.y2.min())
        return ydist

    def _line_for_centerx_and_slope(self, labeled_img, cx, sl, color):
        h = labeled_img.shape[0]
        hh = h / 2.
        # delta_x - how much x changes during half screen height going downwards
        # we have: slope = delta_y / delta_x; hences delta_x = delta_y / slope
        delta_x = np.clip(np.divide(hh, sl), -hh, hh)  # avoid div/0 and out of bounds
        x1, y1 = int(cx - delta_x), 0
        x2, y2 = int(cx + delta_x), h
        cv2.line(labeled_img, (x1, y1), (x2, y2), color, 2)

    def compute_line(self, cx, sl, h):
        
        hh = h / 2.
        # delta_x - how much x changes during half screen height going downwards
        # we have: slope = delta_y / delta_x; hences delta_x = delta_y / slope
        delta_x = np.clip(np.divide(hh, sl), -hh, hh)  # avoid div/0 and out of bounds
        x1, y1 = int(cx - delta_x), 0
        x2, y2 = int(cx + delta_x), h
        return x1, y1, x2, y2

    def steer(self, curr_steering_angle, lane_lines, w, h):
        if len(lane_lines) == 0:
            logging.debug
            return curr_steering_angle

        new_steering_angle = self.compute_steering_angle(w, h, lane_lines)
        new_steering_angle = self.stabilize_steering_angle(curr_steering_angle, new_steering_angle, len(lane_lines))

        return new_steering_angle

    def compute_steering_angle(self, w, h, lane_lines):
        if len(lane_lines) == 0:
            logger.debug('0 lines')
            return -90

        if len(lane_lines) == 1:
            x1, _, x2, _ = lane_lines[0]
            x_offset = x2 - x1
        else:
            _, _, left_x2, _ = lane_lines[0]
            _, _, right_x2, _ = lane_lines[1]
            camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
            mid = int(w / 2 * (1 + camera_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid

        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(h / 2)

        logger.debug(f'x/y_offset: {x_offset}, {y_offset}')

        angle_to_mid_radian = math.atan(-x_offset / y_offset)  # angle (in radian) to center vertical line
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
        steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

        logger.debug('new steering angle: %s' % steering_angle)
        return steering_angle

    def stabilize_steering_angle(self, curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=5):
        """
        Using last steering angle to stabilize the steering angle
        This can be improved to use last N angles, etc
        if new angle is too different from current angle, only turn by max_angle_deviation degrees
        """
        if num_of_lane_lines == 2 :
            # if both lane lines detected, then we can deviate more
            max_angle_deviation = max_angle_deviation_two_lines
        else :
            # if only one lane detected, don't deviate too much
            max_angle_deviation = max_angle_deviation_one_lane
        
        angle_deviation = new_steering_angle - curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(curr_steering_angle
                                            + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        logger.info(f'Current: {curr_steering_angle} proposed: {new_steering_angle}, stabilized: {stabilized_steering_angle}')
        return stabilized_steering_angle
        
    def display_heading_line(self, frame, steering_angle, line_color=(0, 0, 255), line_width=5):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape

        # figure out the heading line from steering angle
        # heading line (x1,y1) is always center bottom of the screen
        # (x2, y2) requires a bit of trigonometry

        # Note: the steering angle of:
        # 0-89 degree: turn left
        # 90 degree: going straight
        # 91-180 degree: turn right 
        steering_angle_radian = steering_angle / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 2)

        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

        return heading_image






"""
def detect_lanes(orig_img, bottom_fraction_to_analyze=.5, return_marked_image=True) -> \
        Tuple[float, bool, bool, Optional[np.ndarray]]:
    """"""
    
    Takes prefiltered image and produces and average sliope between left and right lane-lines which can be used to
    maintain in-lane steering.

    Args:
        orig_img:
        bottom_fraction_to_analyze:
        return_marked_image:

    Returns:
        tuple(final_slope, left_visible, right_visible, labeled_img)
        final_slope - returns average slope between left and right lane-lines; slope = (y2 - y1) / (x2 - x1)
        left_visible, right_visible - flags on which lane has been detected - if unexpectedly one line disappeared
                                      slope result is less confident; it also can help to get into the lane at the start
        labeled_img - if return_marked_image=True returns image with left, right and micro-lines marked
    """"""
    
    img = orig_img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    # crop top part of the image
    img = img[-int(h * bottom_fraction_to_analyze):, :]
    gray_img = img

    # FastLineDetector is opencv-contrib-python alternative to LineSegmentDetector implemented in core opencv-python
    # which seems to be crashing in 4.5.5 on both RPI and MacOS; LineSegmentDetector suggestion taken from comments
    # on the bottom of: https://stackoverflow.com/questions/41329665/linesegmentdetector-in-opencv-3-with-python
    line_algo = cv2.ximgproc.createFastLineDetector(length_threshold=int(line_detection_length_thold * h))
    lines = line_algo.detect(gray_img)
    assert lines is not None, 'FastLineDetector did not detect lines'

    # convert numpy array into pandas DF - working with columns as vectors is slightly more intuitive/readable versus
    # unlabeled numpy array
    ldf = np.squeeze(lines, axis=1)
    ldf = pd.DataFrame(ldf, columns=['x1', 'y1', 'x2', 'y2'])

    # Note: the origin, (0, 0), is located at the top-left of
    # the image. OpenCV images are zero-indexed, where the x-values go left-to-right (column number) and y-values
    # go top-to-bottom (row number)
    # slope=inf on vertical line
    ldf['slope'] = (ldf.y2 - ldf.y1) / (ldf.x2 - ldf.x1)

    cx_l, cx_r, sl_l, sl_r, ldff = _find_lanelines(ldf, w, h)

    labeled_img = None
    if return_marked_image:
        # note that ldff has outlier lines removed, switch to ldf to see all lines detected
        labeled_img = _label_image(gray_img, line_algo, ldff, cx_l, sl_l, cx_r, sl_r)

    left_visible, right_visible = cx_l is not None, cx_r is not None
    if left_visible and right_visible:
        final_slope = (sl_l + sl_r) / 2
        print(f"l: {sl_l} - r: {sl_r}")
    else:
        final_slope = sl_l or sl_r
    

    return labeled_img, cx_l, sl_l, cx_r, sl_r

def _label_image(gray_img, line_algo, ldf, cx_l, sl_l, cx_r, sl_r):
    # first 4 columns contain coords in same order as originally returned by FastLineDetector
    out_lines = ldf.iloc[:, :4]
    out_lines = out_lines.values[:, np.newaxis, :]
    labeled_img = line_algo.drawSegments(gray_img, out_lines)
    # left lane-line
    if cx_l:
        _line_for_centerx_and_slope(labeled_img, cx_l, sl_l, GREEN)
    # right lane-line
    if cx_r:
        _line_for_centerx_and_slope(labeled_img, cx_r, sl_r, BLUE)
    return labeled_img

""""""
    
def _filter_raw_line_outliers(ldf, h):
    res = ldf
    if len(ldf) > 3:
        # filter slope outliers - lines that diverge from median slope by X standard deviations
        slope = ldf.slope.clip(-h, h)  #  vertical line slope=inf, clip it to max non inf slope possible on
                                       # the image - 1px wide vertical
        ldff = ldf.loc[(slope.median() - slope).abs() < s_outlier_filter_param * slope.std()]
        # do not apply filtering if it's removing more than ~60% of cluster - std used in filter above
        res = ldff if len(ldff) / len(ldf) > s_outlier_filter_require_to_remain else ldf
    return res

def _find_lanelines(ldf, w, h):
    """

"""

    Returns: (cx_l, cx_r, sl_l, sl_r, ldff):
        cx_l, cx_r - center x of left and right lanelines
        sl_l, sl_r - slopes of --"--
        ldff - diagnostics dataframe, a copy of ldf arg after outlier filtering and other intermediate processing cols
    """
"""
    
    small_sample = len(ldf) == 1
    if not small_sample:
        # try to locate 2 means of x coordinates
        clust_alg = KMeans(n_clusters=2)
        ldf['left_right_flag'] = clust_alg.fit_predict(ldf[['x1']])
        # ldf['left_right_t2'] = clust_alg.fit_predict(ldf[['x2']])  # there is probably information in x2 that we do not use
        # is error bigger when we use 2 clusters or one?
        ldf0 = ldf.loc[ldf['left_right_flag'] == 0]
        ldf1 = ldf.loc[ldf['left_right_flag'] == 1]

        # outlier removal in case of 2 clusters
        ldf0 = _filter_raw_line_outliers(ldf0, h)
        ldf1 = _filter_raw_line_outliers(ldf1, h)
        ldf_2c_f = pd.concat([ldf0, ldf1])

        # outlier removal in case of 1 cluster
        ldf_1c_f = _filter_raw_line_outliers(ldf, h)

        # mean average error for x versus center-x
        ldf0_cx, ldf1_cx = ldf0.x1.median(), ldf1.x1.median()
        e0 = (ldf0_cx - ldf0.x1).abs().sum() / len(ldf0)
        e1 = (ldf1_cx - ldf1.x1).abs().sum() / len(ldf1)
        e_single_c = (ldf_1c_f.x1.median() - ldf_1c_f.x1).abs().sum() / len(ldf_1c_f)

        # computed as absolute percentage difference between two sizes expressed as fraction
        clusters_ydist_fraction_diff = abs(1 - _ydist(ldf0) / _ydist(ldf1))
    else:
        ldf_1c_f = ldf

    if not small_sample \
            and abs(ldf0_cx - ldf1_cx) > min_distance_between_clusters * w \
            and clusters_ydist_fraction_diff < accept_clusters_below_ydist_fraction_diff \
            and (e0 + e1) / 2 < e_single_c:  # is error bigger when we use 2 clusters or one?
        # using 2 centers
        ldff = ldf_2c_f
        grouped = ldf_2c_f.groupby('left_right_flag')
        # note that sorting by x classifies between left/right line
        cluster_stats_df = grouped['x1'].median().sort_values().to_frame(name='center_x')
        cluster_stats_df['slope'] = grouped['slope'].median()

        cx_l, cx_r = cluster_stats_df['center_x'].astype(int).values
        sl_l, sl_r = cluster_stats_df['slope'].values
    else:
        # using 1 center
        ldff = ldf_1c_f
        cx, sl = ((ldff.x1 + ldff.x2) / 2.).median(), ldff.slope.median()
        # note that logic below depends if we're on-the-road or off-the-road at the moment, - assuming on-the-road
        cx_l, sl_l, cx_r, sl_r = None, None, None, None
        if sl > 0:
            # line \ is on the right
            cx_r, sl_r = cx, sl
        else:
            # line / is on the left
            cx_l, sl_l = cx, sl
    return cx_l, cx_r, sl_l, sl_r, ldff

def _ydist(ldf):
    ydist = max(ldf.y1.max(), ldf.y2.max()) - min(ldf.y1.min(), ldf.y2.min())
    return ydist

def _line_for_centerx_and_slope(labeled_img, cx, sl, color):
    h = labeled_img.shape[0]
    hh = h / 2.
    # delta_x - how much x changes during half screen height going downwards
    # we have: slope = delta_y / delta_x; hences delta_x = delta_y / slope
    delta_x = np.clip(np.divide(hh, sl), -hh, hh)  # avoid div/0 and out of bounds
    x1, y1 = int(cx - delta_x), 0
    x2, y2 = int(cx + delta_x), h
    cv2.line(labeled_img, (x1, y1), (x2, y2), color, 2)

def compute_line(cx, sl, h):
    
    hh = h / 2.
    # delta_x - how much x changes during half screen height going downwards
    # we have: slope = delta_y / delta_x; hences delta_x = delta_y / slope
    delta_x = np.clip(np.divide(hh, sl), -hh, hh)  # avoid div/0 and out of bounds
    x1, y1 = int(cx - delta_x), 0
    x2, y2 = int(cx + delta_x), h
    return x1, y1, x2, y2

def steer(curr_steering_angle, lane_lines, w, h):
    if len(lane_lines) == 0:
        logging.debug
        return curr_steering_angle

    new_steering_angle = compute_steering_angle(w, h, lane_lines)
    new_steering_angle = stabilize_steering_angle(curr_steering_angle, new_steering_angle, len(lane_lines))

    return new_steering_angle

def compute_steering_angle(w, h, lane_lines):
    if len(lane_lines) == 0:
        logger.debug('0 lines')
        return -90

    if len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0]
        _, _, right_x2, _ = lane_lines[1]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(w / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(h / 2)

    logger.debug(f'x/y_offset: {x_offset}, {y_offset}')

    angle_to_mid_radian = math.atan(-x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    logger.debug('new steering angle: %s' % steering_angle)
    return steering_angle

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=5):
    """
"""

    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """"""
    
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logger.info(f'Current: {curr_steering_angle} proposed: {new_steering_angle}, stabilized: {stabilized_steering_angle}')
    return stabilized_steering_angle
    
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image
"""
