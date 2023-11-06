#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install matplotlib  


# In[6]:


pip install tensorflow-datasets


# In[5]:


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[6]:


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# In[7]:


ds = tfds.load('fashion_mnist', split='train', shuffle_files=True)


# In[8]:


ds.as_numpy_iterator().next()['label']


# In[9]:


import numpy as np


# In[10]:


dataiterator = ds.as_numpy_iterator()


# In[11]:


dataiterator.next()


# In[12]:


fig, ax = plt.subplots(ncols=4, figsize= (20, 20))
for idx in range(4):
    sample = dataiterator.next()
    ax[idx].imshow(np.squeeze(sample['image']))
    ax[idx].title.set_text(sample['label'])


# In[13]:


def scale_image(data):
    image = data['image']
    return image / 225


# In[14]:


ds = tfds.load('fashion_mnist', split='train')
ds = ds.map(scale_image)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(128)
ds = ds.prefetch(64)


# In[15]:


ds.as_numpy_iterator().next().shape


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D


# In[17]:


def build_generator():
    model = Sequential()
    
    #Beginnings 
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    #Upsampling block 1
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    #Upsampling block 2
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    #Convolutional block 1
    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(0.2))

    #Convolutional block 2
    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(0.2))

    #Convolutional layer to get to 1 channel
    model.add(Conv2D(1, 4, padding="same", activation='sigmoid'))


    return model


# In[18]:


generator = build_generator()


# In[19]:


generator.summary()


# In[20]:


#Generate new fashion
img = generator.predict(np.random.randn(4, 128, 1))

fig, ax = plt.subplots(ncols=4, figsize= (20, 20))
for idx, img in enumerate(img):
    sample = dataiterator.next()
    ax[idx].imshow(np.squeeze(img))
    ax[idx].title.set_text(idx)


# In[21]:


img.shape


# In[22]:


def build_discriminator():
    model = Sequential()

    #First Conv Block
    model.add(Conv2D(32, 5, input_shape=(28,28,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))


    return model


# In[23]:


discriminator = build_discriminator()


# In[24]:


discriminator.summary()


# In[25]:


img = np.reshape(img, (-1, 28, 28, 1))


# In[26]:


discriminator.predict(img)


# In[27]:


# Adam as optimizer for both generator and discriminator
from tensorflow.keras.optimizers import Adam

# Binary cross entropy as loss function for both generator and discriminator
from tensorflow.keras.losses import BinaryCrossentropy


# In[28]:


g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()


# In[29]:


# Importing the base model class to subclass our training step
from tensorflow.keras.models import Model


# In[30]:


class fashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class
        super().__init__(*args, **kwargs)

        # Create attributes for generator and discriminator
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        # Compile with base class
        super().compile(*args, **kwargs)

        # Create attributes for generator and discriminator optimizers and losses
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss
    
    def train_step(self, batch):
        # Get the data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create the labels for real and fake images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss - BINARYCROSSENTROPY   
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # Apply backpropagation - nn learns from its mistakes
        dgrads = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrads, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate some new images
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            # Create the predicted labels for the new images
            predicted_labels = self.discriminator(gen_images, training=False)

            # Calculate loss - trick the discriminator into thinking the images are real
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply backpropagation
        ggrads = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrads, self.generator.trainable_variables))

        # Return the losses as a dictionary to be monitored by keras
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}


# In[31]:


# Create an instance of the subclassed model
fashgan = fashionGAN(generator, discriminator)


# In[32]:


# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)


# In[33]:


import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback


# In[34]:


class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
    
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
        generated_images = self.model.generator(random_latent_vectors, training=False)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('GANImages', f'generated_img_{epoch}_{i}.png'))


# In[35]:


hist = fashgan.fit(ds, epochs=20, callbacks=[ModelMonitor()])


# In[37]:


plt.suptitle('loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()


# In[38]:


imgs = generator.predict(tf.random.normal((4, 128, 1)))


# In[39]:


fig, ax = plt.subplots(ncols=4, figsize= (20, 20))
for idx, img in enumerate(imgs):
    ax[idx].imshow(np.squeeze(img))
    ax[idx].title.set_text(idx)


# In[40]:


generator.save('generator.h5')
discriminator.save('discriminator.h5')


# In[ ]:




