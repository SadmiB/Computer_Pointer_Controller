'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from module import Module
import logging as log



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


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        log.info("Inference...")
        
        input_image = self.preprocess_input(image)

        input_dict = {self.input_name: input_image}

        self.net.infer(input_dict)

        outputs = self.net.requests[0].outputs

        return outputs

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        results = HeadPoseEstimator.Result([outputs['angle_y_fc'].tolist()[0][0],
                                            outputs['angle_p_fc'].tolist()[0][0], 
                                            outputs['angle_r_fc'].tolist()[0][0]])
        
        return results