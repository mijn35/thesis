import argparse
import copy

import cv2 as cv
import mediapipe as mp
import self

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


window_title = 'Recognition result'
trackbar_title = 'Hue offset'


class HueHelper:
    def __init__(self):
        self.mp_hands = mp_hands.Hands(static_image_mode=True,
                                       max_num_hands=2,
                                       min_detection_confidence=0.5)
        self.img_bgr = None
        self.hue_offset = 0  # Initialize hue offset

    def apply_hue_offset(self, image, hue_offset):  # 0 is no change; 0<=huechange<=180
        # convert img to hsv
        img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img_hsv)
        # shift the hue
        h = (h + hue_offset) % 180
        # combine new hue with s and v
        img_hsv = cv.merge([h, s, v])
        # convert from HSV to BGR
        return cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

    def on_trackbar_change(self, trackbar_hue_offset):
        # Update hue offset
        self.hue_offset = trackbar_hue_offset

    def recognize(self, img_bgr):
        # Apply the hue offset to the image
        img_with_hue = self.apply_hue_offset(img_bgr, self.hue_offset)
        img_rgb = cv.cvtColor(img_with_hue, cv.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_with_hue,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        else:
            print('no hands were found')
        return img_with_hue

    def run(self):
        while True:
            ret, image = cap.read()
            image = cv.flip(image, 1)  # Mirror display

            # Detection implementation #############################################################
            # Run the hue adjustment on the image
            img_with_hue = self.recognize(image)

            # Display the image
            cv.imshow(window_title, img_with_hue)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    h = HueHelper()
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #video_path = 'C:\gesture_recognition_by_image\\sign_language3.mp4'
    #cap = cv.VideoCapture(video_path)

    # Create a window
    cv.namedWindow(window_title)

    # Create trackbar and set callback function
    cv.createTrackbar(trackbar_title, window_title, 0, 179, h.on_trackbar_change)

    while True:
        ret, image = cap.read()
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        # Run the hue adjustment on the image
        img_with_hue = h.recognize(image)

        # Display the image
        cv.imshow(window_title, img_with_hue)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

#HUE 35 is a possibility but still not perfectly
