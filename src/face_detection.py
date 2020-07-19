from module import Module
import logging as log
import cv2

class FaceDetector(Module):
    '''
    Class for the Face Detection Model.
    '''

    class Result:

        def __init__(self, outputs):

            self.points = outputs 

            get_point = lambda i: self[i]

            self.xmin = get_point(0)
            self.ymin = get_point(1)
            self.xmax = get_point(2)
            self.ymax = get_point(3)

        def __getitem__(self, i):
            return self.points[i]

        def getFaceCrop(self, frame):
            return frame[self.ymin:self.ymax, self.xmin:self.xmax]


    def __init__(self, model_name, device='CPU', extension=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        
        Module.__init__(self, model_name, device, extension)  


    def preprocess_output(self, outputs, _w, _h):
        '''
            Preprocess the output.
        '''
        log.info('Preprocessing output...')
        results = [FaceDetector.Result([int(out[3] * _w), int(out[4] * _h), int(out[5] * _w), int(out[6] * _h)]) \
                                        for out in outputs[0][0]]

        return results


