from tensorflow import keras
import tensorflow as tf
from PIL import Image
from wgan import WeightConstraint
import sys


generator = keras.models.load_model('%s/gen'%sys.argv[1],compile=False)
validator = keras.models.load_model('%s/val'%sys.argv[1],compile=False,custom_objects={"WeightConstraint":WeightConstraint})

def best_dim(num):
    target_ratio = 3/2 #width/height
    ratio_diff = float('inf')
    best_dim = (None,None)
    for i in range(1,num+1):
        if num % i == 0:
            ratio = (num//i) / i
            if abs(target_ratio - ratio) < ratio_diff:
                ratio_diff = abs(target_ratio - ratio)
                best_dim = (num//i,i)
    return best_dim
                
while True:
    inpt = input('type "quit" to stop. ')
    if inpt == 'quit': break
    if inpt == 'reload':
        generator = keras.models.load_model('%s/gen'%sys.argv[1],compile=False)
        validator = keras.models.load_model('%s/val'%sys.argv[1],compile=False,custom_objects={"WeightConstraint":WeightConstraint})
    else:
        static = tf.random.normal((1,generator.input_shape[-1]))
        img = generator(static)
        valid = validator(img)

        print('logits: %0.4f'%tf.reduce_mean(valid))
        img *= 127.5
        img += 127.5
        img = tf.cast(img,tf.uint8).numpy()
        png = Image.fromarray(img[0])
        png.save('test_bush.png',format='PNG')