from gan import GAN
from wgan import WGAN
import tensorflow as tf
import numpy as np
import os, sys

if __name__ == '__main__':
    img_dir = sys.argv[1]
    model_dir = sys.argv[2]
    x_names = os.listdir(img_dir)
    points = len(x_names)
    train_x = np.empty((points,64,64,3),dtype=np.float32)
    for i in range(points):
        try:
            name = x_names[i]
            img = open("%s/%s"%(img_dir,name),'rb').read()
            tensor = tf.io.decode_image(img,3)  
            tensor = tf.image.resize_with_pad(tensor,64,64)
            train_x[i] = tensor
        except:
            print(name)
    # Normalize data to range [-1,1]
    train_x -= 127.5
    train_x /= 127.5

    data = tf.data.Dataset.from_tensors(train_x)

    if '-l' in sys.argv:
        print('loading model from directory: %s'%model_dir)
        generator = tf.keras.models.load_model('%s/gen'%model_dir,compile=False)
        validator = tf.keras.models.load_model('%s/val'%model_dir,compile=False)
        model = WGAN(validator,generator)
    else:
        model = WGAN()
    model.generator.summary()
    model.validator.summary()
    model.compile()

    i = 1

    while True:
        print('-- Epoch %i --'%i)
        model.fit(data,workers=3,use_multiprocessing=True)
        if i % 1 == 0:
            model.generator.save('%s/gen'%model_dir,include_optimizer=False,save_format='h5')
            model.validator.save('%s/val'%model_dir,include_optimizer=False,save_format='h5')
        i += 1
