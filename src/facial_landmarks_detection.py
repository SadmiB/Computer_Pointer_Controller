'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from module import Module
import numpy as np

class LandmarksDetector(Module):
    '''
    Class for the Face Detection Model.
    '''

    class Result:

        def __init__(self, outputs):
            self.points = outputs

            get_point = lambda i: self[i]

            self.left_eye = get_point(0)  
            self.right_eye = get_point(1)  

        def __getitem__(self, i):
            return self.points[i]

        def getEyesCrop(self, face):
            scale = np.array([face.shape[1], face.shape[0]])

            left_eye_x, left_eye_y = np.array(self.left_eye, dtype=np.float64) * scale
            right_eye_x, right_eye_y = np.array(self.right_eye, dtype=np.float64) * scale

            left_eye_x, left_eye_y = int(left_eye_x), int(left_eye_y)
            right_eye_x, right_eye_y = int(right_eye_x), int(right_eye_y)

            left_eye_crop = face[left_eye_y - 30: left_eye_y + 30, left_eye_x - 30:left_eye_x + 30]
            right_eye_crop = face[right_eye_y - 30: right_eye_y + 30, right_eye_x - 30:right_eye_x + 30]

            return left_eye_crop, right_eye_crop



    def __init__(self, model_name, device='CPU', extension=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Module.__init__(self, model_name, device, extension)  


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        results = [LandmarksDetector.Result(output.reshape((-1, 2)))\
                                             for output in outputs]

        return results
