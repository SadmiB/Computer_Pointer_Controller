'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
from openvino.inference_engine import IECore
import logging as log

class Module:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extension=None):
        '''
        TODO: Use this to set your instance variables.
        '''

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extension
       
        self.core = IECore()

        if 'CPU' in self.device and self.extension:
            log.info("Loading the extension {}...".format(self.extension))
            self.core.add_extension(extension, 'CPU')

        try:
            log.info("Loading the model...")
            self.model = self.core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?", e)
        
        self.net = None
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
        return self.net

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        log.info("Inference...")
        
        input_image = self.preprocess_input(image)

        input_dict = {self.input_name: input_image}

        self.net.infer(input_dict)

        outputs = self.net.requests[0].ouputs[self.output_name]

        return outputs

    def check_model(self):
        log.info("Checking the model support...")
        if self.device == 'CPU':
            supported_layers = self.core.query_network(self.net, self.device)
            not_supported_layers = [l for l in self.net.layers.keys() \
                                    if l not in supported_layers]

            if len(not_supported_layers) != 0:
                log.error("The device {} does not support the following layers:\n {}"\
                    .format(self.device, not_supported_layers))
                log.error("Use extensions to handle these unsupported layers")

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        log.info("Preprocessing input...")

        n, c, h, w = self.input_shape
        
        img = cv2.resize(image, (w, h))
        img = img.transpose((2,0,1))
        img = img.reshape((n, c, h, w))
        
        return img
    
    def preprocess_output(self, outputs, init_w, init_h):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        log.info("Preprocessing output...")
        pass






