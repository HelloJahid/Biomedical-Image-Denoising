#!/usr/bin/env python3
from gan import GAN
from generator import Generator
from discriminator import Discriminator
from oct_dataset import build_dataset
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy





class Trainer:
    def __init__(self, x_train=None, x_test=None, x_train_noisy=None, x_test_noisy=None):
        self.W = 256
        self.H = 256
        self.C = 3
        self.EPOCHS = 100000
        self.BATCH = 32
        self.CHECKPOINT = 50
        

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)

        self.X_train = x_train
        self.x_test = x_test
        self.X_train_noisy = x_train_noisy
        self.x_test_noisy = x_test_noisy

    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_temp = deepcopy(self.X_train)
            X_train_noisy_temp = deepcopy(self.X_train_noisy)
            while len(X_train_temp)>self.BATCH:
                # Keep track of Batches
                b=b+1

                # Train Discriminator
                # Make the training batch for this model be half real, half noise
                # Grab Real Images for this training batch
                if self.flipCoin():
                    count_real_images = int(self.BATCH)
                    starting_index = randint(0, (len(X_train_temp)-count_real_images))
                    real_images_raw = X_train_temp[ starting_index : (starting_index + count_real_images) ]
                    #self.plot_check_batch(b,real_images_raw)
                    # Delete the images used until we have none left
                    X_train_temp = np.delete(X_train_temp,range(starting_index,(starting_index + count_real_images)),0)
                    x_batch = real_images_raw.reshape( count_real_images, self.W, self.H, self.C )
                    y_batch = np.ones([count_real_images,1])
                else:
                    # Grab Generated Images for this training batch
                    count_noise_images = int(self.BATCH)
                    starting_index = randint(0, (len(X_train_noisy_temp)-count_noise_images))
                    noise_images_raw = X_train_noisy_temp[ starting_index : (starting_index + count_noise_images) ]
                    #self.plot_check_batch(b,real_images_raw)
                    # Delete the images used until we have none left
                    X_train_noisy_temp = np.delete(X_train_noisy_temp,range(starting_index,(starting_index + count_noise_images)),0)
                   
                    x_batch = self.generator.Generator.predict(noise_images_raw)
                    y_batch = np.zeros([self.BATCH,1])

                # Now, train the discriminator with this batch
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch,y_batch)[0]
            
                # In practice, flipping the label when training the generator improves convergence
                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH,1])
                else:
                    y_generated_labels = np.zeros([self.BATCH,1])

                count_noise_images = int(self.BATCH)
                starting_index = randint(0, (len(X_train_noisy_temp)-count_noise_images))
                noise_images_raw = X_train_noisy_temp[ starting_index : (starting_index + count_noise_images) ]
                    #self.plot_check_batch(b,real_images_raw)
                    # Delete the images used until we have none left
                X_train_noisy_temp = np.delete(X_train_noisy_temp,range(starting_index,(starting_index + count_noise_images)),0)

                generator_loss = self.gan.gan_model.train_on_batch(noise_images_raw,y_generated_labels)
    
                print ('Batch: '+str(int(b))+', [Discriminator :: Loss: '+str(discriminator_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                if b % self.CHECKPOINT == 0 :
                    label = str(e)+'_'+str(b)
                    self.plot_checkpoint(label)

            print ('Epoch: '+str(int(e))+', [Discriminator :: Loss: '+str(discriminator_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                        
            if e % self.CHECKPOINT == 0 :
                self.plot_checkpoint(e)
        return

    def flipCoin(self,chance=0.5):
        return np.random.binomial(1, chance)


    def plot_checkpoint(self,e):
        filename = "/data/sample_"+str(e)+".png"

        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)
        
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.C==1:
                image = images[i, :, :]
                image = np.reshape(image, [self.H,self.W])
                image = (255*(image - np.min(image))/np.ptp(image)).astype(int)
                plt.imshow(image,cmap='gray')
            elif self.C==3:
                image = images[i, :, :, :]
                image = np.reshape(image, [self.H,self.W,self.C])
                image = (255*(image - np.min(image))/np.ptp(image)).astype(int)
                plt.imshow(image)
            
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return

    def plot_check_batch(self,b,images):
        filename = "/data/batch_check_"+str(b)+".png"
        subplot_size = int(np.sqrt(images.shape[0]))
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(subplot_size, subplot_size, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.H,self.W,self.C])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return




if __name__ == "__main__":
    x_train, x_test, x_train_noisy, x_test_noisy = build_dataset()
    gan_trainer = Trainer(x_train, x_test, x_train_noisy, x_test_noisy)
    gan_trainer.train()
