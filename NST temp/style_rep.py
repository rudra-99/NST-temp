from load_image import load_img
from vgg19_model import vgg_layers

def get_style(style_paths : list):
    style_images = [load_img(file) for file in style_paths]

    style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

    style_extractor = vgg_layers(style_layers)
    style_outputs = [style_extractor(img*255) for img in style_images]

    generalised_style = (style_outputs[1] + 100*style_outputs[0])
    generalised_style = [i/2 for i in generalised_style]

    style_images_averaged = style_images[0]
    for i in range(1,len(style_images)):
        style_images_averaged += style_images[i]
    style_images_averaged = style_images_averaged / len(style_images)

    return generalised_style,style_images_averaged
