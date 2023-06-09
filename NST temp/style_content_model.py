from libraries import *
from utilities import *
from vgg19_model import vgg_layers

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name : value for content_name, value in zip (self.content_layers, content_outputs)}
        style_dict = {style_name : value for style_name , value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

content_layers = ['block5_conv2']
style_layers = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']


def get_style_and_content_targets(style_image, content_image):
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']
    return style_targets, content_targets

def get_extractor():
    return StyleContentModel(style_layers, content_layers)