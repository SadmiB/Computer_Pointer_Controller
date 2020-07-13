'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from module import Module

class GazeEstimator(Module):
    '''
    Class for the Face Detection Model.
    '''

    class Result:

        def __init__(self, outputs):

            self.coordinates = outputs

            get_coord = lambda i: self[i]
            self.x = get_coord(0)
            self.y = get_coord(1)
            self.z = get_coord(2)
            
        def __getitem__(self, i):
            return self.coordinates[i]

    def __init__(self, model_name, device='CPU', extension=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        Module.__init__(self, model_name, device, extension)
        self.input_shape =  self.model.inputs['left_eye_image'].shape 

    def predict(self, left_eye, right_eye, head_pose):
        
        
        left_eye = self.preprocess_input(left_eye)
        right_eye = self.preprocess_input(right_eye)

        input_dict = {'left_eye_image':left_eye, 'right_eye_image':right_eye, 'head_pose_angles':head_pose}

        self.net.infer(input_dict)

        outputs = self.net.requests[0].outputs[self.output_name]

        return outputs

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        results = [GazeEstimator.Result(output) for output in outputs]

        return results
        

