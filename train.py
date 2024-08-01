import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import DataProcessing, GAN3dDataset
from gan_models import Discriminator_3DGAN, Generator_3DGAN
import matplotlib.pyplot as plt
import numpy as np


#vector_dim = 200
#noise_vector = torch.normal(mean = torch.zeros(vector_dim), std = torch.ones(vector_dim))
#print(noise_vector.shape)
#noise_vector = torch.reshape(noise_vector, (200, 1, 1, 1))
#noise_vector2 = torch.randn((200, 1, 1, 1))



def generate_noise_vector(vector_dim, normal = False):
    if normal:
        return torch.randn(vector_dim)
    else:
        return torch.rand(vector_dim)



#dataprocess = DataProcessing("D:\\Github repos\\GANs-and-Roses\\Datasets\\3DShapeNets\\volumetric_data")
#gan3d_train_dataset = GAN3dDataset(dataprocess, "airplane")
#gan3d_test_dataset = GAN3dDataset(dataprocess, "airplane", train = False)


#gan3d_train_dataloader = DataLoader(dataset = gan3d_train_dataset, batch_size = 32, shuffle = True)
#gan3d_test_dataloader = DataLoader(dataset = gan3d_test_dataset, batch_size = 32) 

'''
x, y = next(iter(gan3d_train_dataloader))
print(x.shape, y.shape)
'''

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
    
    
    def forward(self, dis_real_preds, dis_fake_preds):
        assert dis_real_preds.shape[0] == dis_fake_preds.shape[0], "Batch sizes of real and fake predictions must match"
        loss = -(torch.log(dis_real_preds) + torch.log(1 - dis_fake_preds))
        return loss.mean()



class GeneratorLoss(nn.Module):
    def __init__(self) -> None:
        super(GeneratorLoss, self).__init__()


    def forward(self, dis_fake_preds): 
        loss = -torch.log(dis_fake_preds)
        return loss.mean()



class GANTraining:
    '''
    GAN training class contains methods for training models, set device, transfer the model to the chosen device
    setting train dataloader and etc.
    '''
    def __init__(self, dis_model, gen_model, dis_loss_func = None, gen_loss_func = None, dis_opt = None, gen_opt = None):
        self.dis_model = dis_model
        self.gen_model = gen_model
        self.dis_loss_func = DiscriminatorLoss() if dis_loss_func is None else dis_loss_func
        self.gen_loss_func = GeneratorLoss() if gen_loss_func is None else gen_loss_func
        self.dis_opt = torch.optim.Adam(self.dis_model.parameters()) if dis_opt is None else dis_opt
        self.gen_opt = torch.optim.Adam(self.gen_model.parameters()) if gen_opt is None else gen_opt
        self.train_loss_history = {'train_dis_loss': [], 'train_gen_loss':[]}
        self.test_loss_history = []


    def set_device(self, device):
        self.device = device


    def model_to_device(self):
        self.gen_model.to(device = self.device)
        self.dis_model.to(device = self.device)


    def set_train_dataloader(self, train_dataloader):
        self.train_dataloader = train_dataloader


    def train(self, epoch_num, batch_size, noise_vector_dim, dis_iterations_num = 1):
        iter_data = {'iterations_num': dis_iterations_num, 'epochs': []}
        train_dataloader_iter = iter(self.train_dataloader)
        for epoch in range(epoch_num):
            for _ in range(dis_iterations_num):
                self.dis_opt.zero_grad()
                self.gen_opt.zero_grad()
                noise_vector_batch = generate_noise_vector((batch_size,) + noise_vector_dim).to(device = self.device) #(batch_size, 200, 1, 1, 1)
                real_inputs_batch = next(train_dataloader_iter)
                real_inputs_batch = real_inputs_batch.to(device = self.device)
                fake_images_batch = self.gen_model(noise_vector_batch)
                dis_real_preds_batch = self.dis_model(real_inputs_batch)
                dis_fake_preds_batch = self.dis_model(fake_images_batch)
                dis_loss = self.dis_loss_func(dis_real_preds_batch, dis_fake_preds_batch)
                self.train_loss_history['train_dis_loss'].append(dis_loss.item())
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dis_model.parameters(), max_norm = 2.0)
                self.dis_opt.step()


            noise_vector_batch = generate_noise_vector((batch_size,) + noise_vector_dim).to(device = self.device)
            fake_images_batch = self.gen_model(noise_vector_batch)
            gen_loss = self.gen_loss_func(self.dis_model(fake_images_batch))
            print(f'Epoch: {epoch + 1}/{epoch_num}, Generator Loss: {gen_loss.item()}, Discriminator Loss: {dis_loss.item()}')
            self.train_loss_history['train_gen_loss'].append(gen_loss.item())
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen_model.parameters(), max_norm = 2.0)
            self.gen_opt.step()
            iter_data['epochs'].append(epoch)

    
        fig, axs = plt.subplots(1, 2)
        epochs = iter_data['iterations_num'] * np.array(iter_data['epochs']) if dis_iterations_num != 1 else iter_data['epochs']
        axs[0].plot(epochs, self.train_loss_history['train_dis_loss'])
        axs[0].set_title('Train discriminator loss')
        axs[1].plot(iter_data['epochs'], self.train_loss_history['train_gen_loss'])
        axs[1].set_title('Train generator loss')
        plt.show()



'''
batch_size = (32, 1, 1, 1, 1)  
dis_real_preds = torch.rand(batch_size) 
dis_fake_preds = torch.rand(batch_size)
criterion = DiscriminatorLoss()
loss = criterion(dis_real_preds, dis_fake_preds)
squeezed_loss = torch.squeeze(loss, (1, 2, 3, 4)) 
print(squeezed_loss, squeezed_loss.shape, squeezed_loss.mean(), sum(squeezed_loss)/len(squeezed_loss))
print(loss, loss.shape, loss.mean())
'''


if __name__ == "__main__":
    batch_size = 32
    dir_path = "D:\\Github repos\\GANs-and-Roses\\Generated images\\GAN3d Generated images"
    dataprocess = DataProcessing("D:\\Github repos\\GANs-and-Roses\\Datasets\\3DShapeNets\\volumetric_data")
    gan3d_train_dataset = GAN3dDataset(dataprocess, "airplane")
    gan3d_train_dataloader = DataLoader(dataset = gan3d_train_dataset, batch_size = batch_size, shuffle = True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dis_model = Discriminator_3DGAN()
    gen_model = Generator_3DGAN()
    
    dis_loss = DiscriminatorLoss()
    gen_loss = GeneratorLoss()

    dis_opt = torch.optim.Adam(dis_model.parameters(), lr = 0.00001) 
    gen_opt = torch.optim.Adam(gen_model.parameters(), lr = 0.000025, betas=(0.5, 0.999))

    
    training = GANTraining(dis_model = dis_model, gen_model = gen_model,
                           dis_loss_func = dis_loss, gen_loss_func = gen_loss,
                           dis_opt = dis_opt, gen_opt = gen_opt)
    training.set_device(device = device)
    training.model_to_device()
    training.set_train_dataloader(gan3d_train_dataloader)
    training.train(epoch_num = 80, batch_size = batch_size, noise_vector_dim = (200, 1, 1, 1))
    
    #Testing generating ability
    for i in range(1):
        noise_vector = generate_noise_vector((200, 1, 1, 1)).to(device = device)
        res_image = gen_model(noise_vector)
        voxels = dataprocess.tensor_to_voxels_image(res_image)
        dataprocess.save_images(voxels, dir_path = dir_path)