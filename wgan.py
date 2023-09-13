import tensorflow as tf
from tensorflow import keras

__author__ = "Tristan Gaeta"

"""
This is an implementation of a WGAN-GP, using Wasserstein loss and Gradient 
penalty (inspired by https://keras.io/examples/generative/wgan_gp/). This 
loss function is supposed to prevent mode collapse and speed up convergence.
Unlike the original WGAN paper, which uses weight clipping to prevent 
runaway values, gradient penalty is more stable allowing us to make use of the
Adam optimizer, which the original WGAN could not.

The overall network architecture is the same as the GAN.
"""

LAMBDA = 10 #scalar for gradient penalty. value from this paper:  https://arxiv.org/abs/1704.00028

class WGAN(keras.Model):
    def __init__(self,val=None,gen=None):
        super(WGAN, self).__init__()
        self.latent_dim = 100
        self.generator = gen or self.__create_generator__()
        self.validator = val or self.__create_validator__()
        self.generator.compile(optimizer=keras.optimizers.Adam(2e-4,0.5,0.9))
        self.validator.compile(optimizer=keras.optimizers.Adam(2e-4,0.5,0.9))
        self.generator_tracker = keras.metrics.Mean()
        self.validator_tracker = keras.metrics.Mean()

    def __create_generator__(self):
        layers = 1024
        x = keras.Input(shape=(self.latent_dim,))
        y = keras.layers.Dense(4*4*layers)(x)
        y = keras.layers.LeakyReLU(0.2)(y)
        y = keras.layers.Reshape((4,4,layers))(y)
        for _ in range(3):
            layers /= 2
            y = keras.layers.Conv2DTranspose(layers,4,2,'same')(y)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.LeakyReLU(0.01)(y)
        y = keras.layers.Conv2DTranspose(3,7,2,padding='same',activation='tanh')(y)
        return keras.Model(inputs=x,outputs=y,name='Generator')

    def __create_validator__(self):
        x = keras.Input(shape=(64,64,3))
        layers = 64
        y = keras.layers.Conv2D(layers,4,2,'same')(x)
        y = keras.layers.LeakyReLU(0.1)(y)
        for _ in range(3):
            layers *= 2
            y = keras.layers.Conv2D(layers,3,2,'same')(y)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.LeakyReLU(0.1)(y)
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dropout(0.4)(y)
        y = keras.layers.Dense(1)(y)
        return keras.Model(inputs=x,outputs=y,name='Validator')

    # this function calculates the gradient penalty used at each step
    def gradient_penalty(self,real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.normal([batch_size, 1, 1, 1])
        interpolated = real_images + alpha*(fake_images - real_images)
        grads = tf.gradients(self.validator(interpolated), [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return LAMBDA * gp

    def train_step(self,train_images):
        batch_size = tf.shape(train_images)[0]
        # When using Wasserstein loss, train the validator more than the generator.
        for _ in range(3):
            static = tf.random.normal(shape=(batch_size,self.latent_dim))
            fake_images = self.generator(static,training=True)

            # Train on validator
            with tf.GradientTape() as tape:
                real_pred = self.validator(train_images,training=True)
                fake_pred = self.validator(fake_images,training=True)
                loss_v = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                loss_v += self.gradient_penalty(train_images,fake_images)
            gradients = tape.gradient(loss_v,self.validator.trainable_variables)
            self.validator.optimizer.apply_gradients(zip(gradients,self.validator.trainable_variables))

        static = tf.random.normal(shape=(batch_size,self.latent_dim))
        # Train on Generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(static,training=True)
            predictions = self.validator(fake_images,training=True)
            loss_g = -tf.reduce_mean(predictions) 
        gradients = tape.gradient(loss_g,self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients,self.generator.trainable_variables))

        # update metrics
        self.generator_tracker.update_state(loss_g)
        self.validator_tracker.update_state(loss_v)
        return {
            'loss_g':self.generator_tracker.result(),
            'loss_v':self.validator_tracker.result()
        }