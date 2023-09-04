import tensorflow as tf


class Sampler_Z(tf.keras.layers.Layer):

    def call(self , inputs):
        mean , logvar = inputs
        sd = tf.exp(logvar * .5)
        epsilon = tf.random.normal(shape=(mean.shape[1] ,) , mean=0. , stddev=1.)
        z_sample = mean + sd * epsilon
        return z_sample , sd


class Encoder(tf.keras.layers.Layer):

    def __init__(self , dim_z , name="encoder" , **kwargs):
        super(Encoder , self).__init__(name=name , **kwargs)

        self.dim_x = (128 , 128 , 1)
        self.dim_z = dim_z
        self.conv_layer1 = tf.keras.layers.Conv2D(filters=16 , kernel_size=3 , strides=(2 , 2) ,
                                                  padding='same' , activation='relu')
        self.conv_layer2 = tf.keras.layers.Conv2D(filters=16 , kernel_size=3 , strides=(2 , 2) ,
                                                  padding='same' , activation='relu')
        self.conv_layer3 = tf.keras.layers.Conv2D(filters=64 , kernel_size=3 , strides=(2 , 2) ,
                                                  padding='same' , activation='relu')
        self.conv_layer4 = tf.keras.layers.Conv2D(filters=128 , kernel_size=3 , strides=(2 , 2) ,
                                                  padding='same' , activation='relu')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_mean = tf.keras.layers.Dense(self.dim_z , activation=None , name='z_mean')
        self.dense_logvar = tf.keras.layers.Dense(self.dim_z , activation=None , name='logvar')
        self.Sampler_Z = Sampler_Z()

    def call(self , x_input):
        z = self.conv_layer1(x_input)
        z = self.conv_layer2(z)
        z = self.conv_layer3(z)
        z = self.conv_layer4(z)
        z = self.flatten_layer(z)
        mean = self.dense_mean(z)
        logvar = self.dense_logvar(z)
        z_sample , sd = self.Sampler_Z((mean , logvar))
        return z_sample , mean , sd


class Decoder(tf.keras.layers.Layer):

    def __init__(self , dim_z , name="encoder" , **kwargs):
        super(Decoder , self).__init__(name=name , **kwargs)

        self.dim_z = dim_z
        self.dense1 = tf.keras.layers.Dense(units=8 * 8 * 128 , activation='relu')
        self.reshape_layer = tf.keras.layers.Reshape((8 , 8 , 128))
        self.conv_transpose_layer1 = tf.keras.layers.Conv2DTranspose(filters=128 , kernel_size=3 , strides=(2 , 2) ,
                                                                     padding='same' , activation='relu')
        self.conv_transpose_layer2 = tf.keras.layers.Conv2DTranspose(filters=64 , kernel_size=3 , strides=(2 , 2) ,
                                                                     padding='same' , activation='relu')
        self.conv_transpose_layer3 = tf.keras.layers.Conv2DTranspose(filters=16 , kernel_size=3 , strides=(2 , 2) ,
                                                                     padding='same' , activation='relu')
        self.conv_transpose_layer4 = tf.keras.layers.Conv2DTranspose(filters=16 , kernel_size=3 , strides=(2 , 2) ,
                                                                     padding='same' , activation='relu')
        self.conv_transpose_layer5 = tf.keras.layers.Conv2DTranspose(filters=1 , kernel_size=3 , strides=1 ,
                                                                     padding='same')

    def call(self , z):
        x_output = self.dense1(z)
        x_output = self.reshape_layer(x_output)
        x_output = self.conv_transpose_layer1(x_output)
        x_output = self.conv_transpose_layer2(x_output)
        x_output = self.conv_transpose_layer3(x_output)
        x_output = self.conv_transpose_layer4(x_output)
        x_output = self.conv_transpose_layer5(x_output)
        return x_output


class deep_vae(tf.keras.Model):

    def __init__(self , dim_z , name="autoencoder" , **kwargs):
        super(deep_vae , self).__init__(name=name , **kwargs)
        self.dim_x = (128 , 128 , 1)
        self.dim_z = dim_z
        self.encoder = Encoder(dim_z=self.dim_z)
        self.decoder = Decoder(dim_z=self.dim_z)

    def call(self , x_input):
        z_sample , mean , sd = self.encoder(x_input)
        x_recons_logits = self.decoder(z_sample)
        kl_divergence = - 0.5 * tf.math.reduce_sum(
            1 + tf.math.log(tf.math.square(sd)) - tf.math.square(mean) - tf.math.square(sd) , axis=1)
        kl_divergence = tf.math.reduce_mean(kl_divergence)
        self.add_loss(kl_divergence)
        return x_recons_logits



def vae_loss(x_true, x_recons_logits):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true, logits=x_recons_logits, name=None)
    neg_log_likelihood = tf.math.reduce_sum(cross_entropy, axis=[1, 2, 3])
    neg_log_lik = tf.math.reduce_mean(neg_log_likelihood)
    return neg_log_lik

