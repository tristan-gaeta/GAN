import tensorflow as tf
from tensorflow import keras

__author__ = "Tristan Gaeta"

"""
This model is a simple deep convolutional Generative-Adversarial-Network for 64x64 pixel rgb 
images. The architecture of this model is similar to the DC-GAN with a few distinctions:
    *   The generator is fed a 128 dim vector of Gaussian noise, and uses a 
        fully connected dense layer with leaky-ReLU activation to upsample. 
        Convolutional layers are not fractional, and use a leaky-ReLU activation.
    *   The validator uses a dropout layer before the output layer, which
        is a dense layer that uses a linear activation.
    *   Instead of mean squared error, this model uses binary crossentropy from logits.
One sided label flipping is used to impair the validator durring early training when generated 
and real images are easily distinguishable, to prevent mode collapse, and to prompt convergence 
later in training (https://www.mathworks.com/help/deeplearning/ug/monitor-gan-training-progress-and-identify-common-failure-modes.html).
Both the validator and generator are trained using the Adam optimizer and a binary-crossentropy 
loss function.
"""
class GAN(keras.Model):
    def __init__(self,val=None,gen=None):
        super(GAN, self).__init__()
        self.latent_dim = 128
        self.generator = gen or self.__create_generator__()
        self.validator = val or self.__create_validator__()
        self.generator.compile(optimizer=keras.optimizers.Adam(3e-4,0.5))
        self.validator.compile(optimizer=keras.optimizers.Adam(3e-4,0.5))
        self.generator_tracker = keras.metrics.Mean()
        self.validator_tracker = keras.metrics.Mean()
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # create and return network generator
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

    # create and return network validator
    def __create_validator__(self):
        x = keras.Input(shape=(64,64,3))
        layers = 64
        y = keras.layers.Conv2D(layers,4,2,'same')(x)
        y = keras.layers.LeakyReLU(0.2)(y)
        for _ in range(3):
            layers *= 2
            y = keras.layers.Conv2D(layers,3,2,'same')(y)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.LeakyReLU(0.2)(y)
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dropout(0.4)(y)
        y = keras.layers.Dense(1)(y)
        return keras.Model(inputs=x,outputs=y,name='Validator')

    # This method is used for one-sided label flipping
    def ones_flipped(self,shape,rate):
        rnd = tf.random.uniform(shape)
        return tf.math.less(rnd,1-rate)

    # Overrides training step so model can be trined using fit()
    def train_step(self,train_images):
        batch_size = tf.shape(train_images)[0]
        static = tf.random.normal(shape=(batch_size,self.latent_dim))
        fake_images = self.generator(static,training=True)  # training=True is required for batch-norm

        # Train on validator
        with tf.GradientTape() as tape:
            real_pred = self.validator(train_images,training=True)
            fake_pred = self.validator(fake_images,training=True)
            ones = self.ones_flipped(tf.shape(real_pred),0.05)  # One sided label flipping
            zeros = tf.zeros_like(fake_pred,tf.bool)
            real_loss = self.loss_fn(ones,real_pred)
            fake_loss = self.loss_fn(zeros,fake_pred)
            loss_v = (real_loss+fake_loss)*0.5
        # find gradients and apply optimizer
        gradients = tape.gradient(loss_v,self.validator.trainable_variables)
        self.validator.optimizer.apply_gradients(zip(gradients,self.validator.trainable_variables))

        static = tf.random.normal(shape=(batch_size,self.latent_dim))
        # Train on Generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(static,training=True)
            predictions = self.validator(fake_images,training=True)
            ones = tf.ones_like(predictions,tf.bool)
            loss_g = self.loss_fn(ones, predictions)
        # find gradients and apply optimizer
        gradients = tape.gradient(loss_g,self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients,self.generator.trainable_variables))

        # update metrics
        self.generator_tracker.update_state(loss_g)
        self.validator_tracker.update_state(loss_v)
        return {
            'loss_g':self.generator_tracker.result(),
            'loss_v':self.validator_tracker.result()
        }