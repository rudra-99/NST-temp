from libraries import *
from utilities import load_img, tensor_to_image, imshow, gram_matrix
from one_punch_image import get_style
from vgg19_model import vgg_layers
from style_content_model import get_style_and_content_targets, get_extractor
import sys
inp = sys.argv[1]

content_path = '/Users/rudra_sarkar/Documents/Mtech Second Sem/Deep Learning/images/secret_of_steve_feature3-1130x480@2x.jpeg'
content_image = load_img(content_path, 'content')
style_image = get_style(str(inp), True)

style_targets , content_targets = get_style_and_content_targets(style_image, content_image)

image = tf.Variable(content_image)
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def style_content_loss(outputs):
    style_weight = 0.09
    content_weight = 1e4
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    num_style_layers = 5
    num_content_layers = 1
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
                          for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers
    
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
                            for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = 1400*style_loss + 700*content_loss
    return loss

extractor = get_extractor()
@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        
        outputs = extractor(image)
        loss = style_content_loss(outputs)
    
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

import time
start = time.time()

epochs = 1
steps_per_epoch = 100

step = 0
display.display(tensor_to_image(image))

for n in range(epochs):
    print(f"Epoch : {n}")
    temp = 0
    for m in range(steps_per_epoch):
        step += 1
        print(f"{int(temp/steps_per_epoch*100)}%",end = "\r")
        print('=======', end='>')
        temp +=1
        train_step(image)
#         print(".", end='', flush=False)
        imshow(image)
        plt.close("all")
        
        
    temp = 0
    display.clear_output(wait=True)
    display.display(tensor_to_image(image))
    
    
    # print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
imshow(image)
