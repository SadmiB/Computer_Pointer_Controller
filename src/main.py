

from face_detection import FaceDetector
from facial_landmarks_detection import LandmarksDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from input_feeder import InputFeeder
from mouse_controller import MouseController
from visualizer import Visualizer

import logging as log
from argparse import ArgumentParser
import cv2
import numpy as np
import logging as log

import time

FACE_DETECTION_MODEL = 'models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
LANDMARKS_MODEL = 'models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'
HEAD_POSE_MODEL = 'models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'
GAZE_MODEL = 'models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'

DEVICES = ['CPU', 'GPU', 'FPGA', 'VPU']

def build_argparser():

    parser = ArgumentParser()

    parser.add_argument('-i', '--input', default=0, help="Path to the input video")

    #models
    parser.add_argument('-m_fd', '--model_fd', \
        default=FACE_DETECTION_MODEL, required=False, help='Face detection model name path')
    parser.add_argument('-m_ld', '--model_ld', \
        default=LANDMARKS_MODEL, required=False, help='Landmarks detection model name path')
    parser.add_argument('-m_hpe', '--model_hpe', \
        default=HEAD_POSE_MODEL, required=False, help='Head pose estimation model name path')
    parser.add_argument('-m_ge', '--model_ge', \
        default=GAZE_MODEL, required=False, help='Gaze estimation model name path')


    #devices
    parser.add_argument('-d_fd', '--device_fd', choices=DEVICES, default='CPU', help='Face detection device')
    parser.add_argument('-d_ld', '--device_ld', choices=DEVICES, default='CPU', help='Landmarks detection device')
    parser.add_argument('-d_hpe', '--device_hpe', choices=DEVICES, default='CPU', help='Head pose estimation device')
    parser.add_argument('-d_ge', '--device_ge', choices=DEVICES, default='CPU', help='Gaze estimation device')

    #extensions (openvino 2020 adds extensions automatically)
    parser.add_argument('-e_fd', '--ext_fd', help='Face detection model extension')
    parser.add_argument('-e_ld', '--ext_ld', help='Landmarks detection model extension')
    parser.add_argument('-e_hpe', '--ext_hpe', help='Head pose estimation model extension')
    parser.add_argument('-e_ge', '--ext_ge', help='Gaze estimation model extension')


    #visualization
    parser.add_argument('-v_fd', '--vis_fd', default=True,required=False, action='store_true', help='Face detection visualization')
    parser.add_argument('-v_ld', '--vis_ld', default=True ,required=False,action='store_true', help='Landmarks detection visualization')
    parser.add_argument('-v_hpe', '--vis_hpe', default=True , required=False,action='store_true', help='Head pose estimation visualization')
    parser.add_argument('-v_ge', '--vis_ge', default=True ,required=False,action='store_true', help='Gaze estimation visualization')


    return parser

def main(args):

    fd_time, le_time, hp_time, ge_time = 0 ,0 ,0 ,0

    face_detector = FaceDetector(args.model_fd, args.device_fd, args.ext_fd)
    start = time.time()
    face_detector.load_model()
    log.info("Face detection laoding time        :{:.4f}".format(time.time() - start ))

    landmarks_detector = LandmarksDetector(args.model_ld, args.device_ld, args.ext_ld)
    
    start = time.time()
    landmarks_detector.load_model()
    log.info("Landmarks estimation loading time: :{:.4f}".format(time.time() - start ))

    head_pose_estimator = HeadPoseEstimator(args.model_hpe, args.device_hpe, args.ext_hpe)
    start = time.time()
    head_pose_estimator.load_model()
    log.info("Head pose estimation loading time: :{:.4f}".format(time.time() - start ))

    gaze_estimator = GazeEstimator(args.model_ge, args.device_ge, args.ext_ge)
    start = time.time()
    gaze_estimator.load_model()
    log.info("Gaze estimation loading time:     :{:.4f}".format(time.time() - start ))

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


    frames = 0

    for flag, frame in input_feeder.next_batch():
        
        if not flag:
            break

        frames +=1

        key = cv2.waitKey(60)
        try:
            start = time.time()

            outputs = face_detector.predict(frame)
            result = face_detector.preprocess_output(outputs, init_w, init_h)[0]
 
            fd_time += time.time() - start


            face = result.getFaceCrop(frame)
            
            start = time.time()

            outputs = landmarks_detector.predict(face)
            landmarks = landmarks_detector.preprocess_output(outputs)[0]

            le_time += time.time() - start

            left_eye, right_eye = landmarks.getEyesCrop(face)

            start = time.time()

            outputs = head_pose_estimator.predict(face)
            head_pose = head_pose_estimator.preprocess_output(outputs)
            
            hp_time += time.time() - start

            head_pose_input = head_pose.get_angles()

            
            start = time.time()
            
            outputs = gaze_estimator.predict(left_eye, right_eye, head_pose_input)
            gaze = gaze_estimator.preprocess_output(outputs)[0]
            
            ge_time += time.time() - start


            real_landmraks = landmarks.getRealEyesCoord(face)    

            log.info("Face detection            :{:.4f}ms".format(fd_time/frames))    
            log.info("Landmarks estimation      :{:.4f}ms".format(le_time/frames))     
            log.info("Head pose estimation      :{:.4f}ms".format(hp_time/frames))     
            log.info("Gaze estimation           :{:.4f}ms".format(ge_time/frames))     

            visualizer = Visualizer(face, real_landmraks, head_pose_input, gaze)

            if args.vis_ld:
                visualizer.draw_landmarks()
            if args.vis_hpe:
                visualizer.draw_head_pose()
            if args.vis_ge:
                visualizer.draw_gazes()
            if args.vis_fd or args.vis_ge or args.vis_hpe or args.vis_ld:    
                visualizer.show()


            mouse_x, mouse_y = gaze.getMouseCoord(head_pose.roll)

            mouse_controller.move(mouse_x, mouse_y)

        except Exception as e:
            log.error("Error: {}".format(e))
        finally:
            if key == 27:
                break

    input_feeder.close()


if __name__ == '__main__':

    log.getLogger().setLevel(log.INFO)

    log.info('Start...')

    args = build_argparser().parse_args()

    main(args)

    log.info('End...')
