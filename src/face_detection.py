'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

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
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        log.info('Preprocessing output...')
        results = [FaceDetector.Result([int(out[3] * _w), int(out[4] * _h), int(out[5] * _w), int(out[6] * _h)]) \
                                        for out in outputs[0][0]]

        return results


