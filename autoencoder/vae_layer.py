from keras.layers import *
from keras import backend as K
from keras.losses import binary_crossentropy

class VAELayer(Layer):

    def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        xent_loss = binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        # x, z_decoded, z_mu, z_log_sigma
        assert len(inputs) >= 4
        x = inputs[0]
        z_decoded = inputs[1]
        z_mu = inputs[2]
        z_log_sigma = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
        self.add_loss(loss, inputs=inputs)
        return z_decoded
