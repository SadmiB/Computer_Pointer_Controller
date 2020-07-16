

import cv2
import numpy as np
import math

class Visualizer:


    def __init__(self, face, eyes_landmarks, head_pose, gaze):
        self.face = face
        self.eyes_landmarks = eyes_landmarks
        self.head_pose = head_pose
        self.gaze = gaze


    def draw_landmarks(self):

        lx, ly, rx, ry = self.eyes_landmarks

        cv2.rectangle(self.face, (lx-30,ly-30),(lx+30,ly+30),(0,255,0),1)
        cv2.rectangle(self.face, (rx-30,ry-30),(rx+30,ry+30),(0,255,0),1)
    

    def draw_gazes(self):
        
        arrow_lenght = 0.4 * self.face.shape[1]

        gx = self.gaze.x
        gy = - self.gaze.y

        gaze_arrow = np.array([gx, gy], dtype=np.float64) * arrow_lenght

        lx, ly, rx, ry = self.eyes_landmarks
        gx, gy = gaze_arrow.astype(int)

        cv2.arrowedLine(self.face, (lx, ly), (lx + gx, ly + gy), (0, 255,255), 1)
        cv2.arrowedLine(self.face, (rx, ry), (rx + gx, ry + gy), (0,255,255), 1)




        

    def draw_head_pose(self):
        
        yaw = self.head_pose[0] * math.pi / 180 
        pitch = self.head_pose[1] * math.pi / 180 
        roll = self.head_pose[2] * math.pi / 180 


        cv2.putText(self.face, "yaw={:.3f}\n pitch={:.3f}, roll={:.3f}"\
            .format(yaw, pitch, roll), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        #cv2.line(self.face, (cx, cy), ())
        


    def show(self):
        cv2.imshow('face', self.face)
        