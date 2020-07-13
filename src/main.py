

from face_detection import FaceDetector
from facial_landmarks_detection import LandmarksDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from input_feeder import InputFeeder
from mouse_controller import MouseController

import logging as log
from argparse import ArgumentParser
import cv2
import numpy as np

DEVICES = ['CPU', 'GPU', 'FPGA', 'VPU']

def build_argparser():

    parser = ArgumentParser()

    parser.add_argument('-i', '--input', default=0, help="Path to the input video")

    #models
    parser.add_argument('-m_fd', '--model_fd', \
        default='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001', required=False, help='Face detection model name path')
    parser.add_argument('-m_ld', '--model_ld', \
        default='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009', required=False, help='Landmarks detection model name path')
    parser.add_argument('-m_hpe', '--model_hpe', \
        default='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001', required=False, help='Head pose estimation model name path')
    parser.add_argument('-m_ge', '--model_ge', \
        default='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002', required=False, help='Gaze estimation model name path')


    #devices
    parser.add_argument('-d_fd', '--device_fd', choices=DEVICES, default='CPU', help='Face detection device')
    parser.add_argument('-d_ld', '--device_ld', choices=DEVICES, default='CPU', help='Landmarks detection device')
    parser.add_argument('-d_hpe', '--device_hpe', choices=DEVICES, default='CPU', help='Head pose estimation device')
    parser.add_argument('-d_ge', '--device_ge', choices=DEVICES, default='CPU', help='Gaze estimation device')

    #extensions
    parser.add_argument('-e_fd', '--ext_fd', help='Face detection model extension')
    parser.add_argument('-e_ld', '--ext_ld', help='Landmarks detection model extension')
    parser.add_argument('-e_hpe', '--ext_hpe', help='Head pose estimation model extension')
    parser.add_argument('-e_ge', '--ext_ge', help='Gaze estimation model extension')


    return parser

def main(args):

    face_detector = FaceDetector(args.model_fd, args.device_fd, args.ext_fd)
    face_detector.load_model()

    landmarks_detector = LandmarksDetector(args.model_ld, args.device_ld, args.ext_ld)
    landmarks_detector.load_model()

    head_pose_estimator = HeadPoseEstimator(args.model_hpe, args.device_hpe, args.ext_hpe)
    head_pose_estimator.load_model()

    gaze_estimator = GazeEstimator(args.model_ge, args.device_ge, args.ext_ge)
    gaze_estimator.load_model()

    mouse_controller = MouseController()


    log.info("Loading the input video...")

    if args.input == 0:
        input_feeder = InputFeeder('cam', args.input)
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        input_feeder = InputFeeder('image', args.input)
    else:
        input_feeder = InputFeeder('video', args.input)
    

    input_feeder.load_data()
    
    init_w = int(input_feeder.cap.get(3))
    init_h = int(input_feeder.cap.get(4))

    for flag, frame in input_feeder.next_batch():
        
        if not flag:
            break

        outputs = face_detector.predict(frame)
        result = face_detector.preprocess_output(outputs, init_w, init_h)[0]

        face = result.getFaceCrop(frame)
        
        outputs = landmarks_detector.predict(face)
        landmarks = landmarks_detector.preprocess_output(outputs)[0]




        scale = np.array([face.shape[1], face.shape[0]])

        left_eye_x, left_eye_y = np.array(landmarks.left_eye, dtype=np.float64) * scale
        right_eye_x, right_eye_y = np.array(landmarks.right_eye, dtype=np.float64) * scale

        left_eye_x, left_eye_y = int(left_eye_x), int(left_eye_y)
        right_eye_x, right_eye_y = int(right_eye_x), int(right_eye_y)

        left_eye, right_eye = landmarks.getEyesCrop(face)

        outputs = head_pose_estimator.predict(face)
        head_pose = head_pose_estimator.preprocess_output(outputs)

        head_pose_input = [head_pose.yaw, head_pose.pitch, head_pose.roll]

        outputs = gaze_estimator.predict(left_eye, right_eye, head_pose_input)
        gaze = gaze_estimator.preprocess_output(outputs)[0]
        
        print(gaze.x, gaze.y, gaze.z)

        mouse_controller.move(gaze.x*100, gaze.y*100)


    input_feeder.close()


if __name__ == '__main__':

    #log.getLogger().setLevel(log.INFO)


    log.info('Start...')

    args = build_argparser().parse_args()

    main(args)

    log.info('End...')
