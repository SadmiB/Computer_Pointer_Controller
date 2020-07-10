'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from module import Module

class HeadPoseEstimator(Module):
    '''
    Class for the Face Detection Model.
    '''
    class Result:

        def __init__(self, outputs):

            self.angles = outputs

            get_angle = lambda i: self[i]

            self.yaw = get_angle(0)
            self.pitch = get_angle(1)
            self.roll = get_angle(2)

        def __getitem__(self, i):
            return self.angles[i]

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
        results = [HeadPoseEstimator.Result([output['angle_y_fc'], output['angle_p_fc'], output['angle_r_fc']]) for output in outputs]
        
        return results