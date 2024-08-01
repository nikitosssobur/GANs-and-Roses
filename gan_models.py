import torch 
import torch.nn as nn
from torchsummary import summary



class Discriminator_3DGAN(nn.Module):
    def __init__(self): 
        #input image size (64, 64, 64)
        super(Discriminator_3DGAN, self).__init__()
        self.channels = (1, 64, 128, 256, 512, 1)
        self.kernel_size = 4
        self.strides = (2, 2, 2, 2, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv3d(in_channels = self.channels[0], out_channels = self.channels[1], 
                               kernel_size = self.kernel_size, stride = self.strides[0], padding = 1)
        self.conv2 = nn.Conv3d(in_channels = self.channels[1], out_channels = self.channels[2], 
                               kernel_size = self.kernel_size, stride = self.strides[1], padding = 1)
        self.conv3 = nn.Conv3d(in_channels = self.channels[2], out_channels = self.channels[3], 
                               kernel_size = self.kernel_size, stride = self.strides[2], padding = 1)
        self.conv4 = nn.Conv3d(in_channels = self.channels[3], out_channels = self.channels[4], 
                               kernel_size = self.kernel_size, stride = self.strides[3], padding = 1)
        self.conv5 = nn.Conv3d(in_channels = self.channels[4], out_channels = self.channels[5], 
                               kernel_size = self.kernel_size, stride = self.strides[4], padding = 'valid')
        
        self.batchnorm1 = nn.BatchNorm3d(self.channels[1]) 
        self.batchnorm2 = nn.BatchNorm3d(self.channels[2])
        self.batchnorm3 = nn.BatchNorm3d(self.channels[3])
        self.batchnorm4 = nn.BatchNorm3d(self.channels[4])
        #self.batchnorm5 = nn.BatchNorm3d(self.channels[4])


    def forward(self, x):
        x = x.view(-1, *x.shape) if len(x.shape) == 4 else x
        x = self.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x        


class Generator_3DGAN(nn.Module):
    def __init__(self):
        super(Generator_3DGAN, self).__init__()
        self.channels = (200, 512, 256, 128, 64, 1)
        self.kernel_size = (4, 4, 4)
        self.strides = (1, 2, 2, 2, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.input_vector_dim = 200
        self.conv1_3d_transpose = nn.ConvTranspose3d(in_channels = self.channels[0], out_channels = self.channels[1],
                                                     kernel_size = self.kernel_size, stride = self.strides[0], padding = 0)
        self.conv2_3d_transpose = nn.ConvTranspose3d(in_channels = self.channels[1], out_channels = self.channels[2],
                                                     kernel_size = self.kernel_size, stride = self.strides[1], padding = 1)
        self.conv3_3d_transpose = nn.ConvTranspose3d(in_channels = self.channels[2], out_channels = self.channels[3],
                                                     kernel_size = self.kernel_size, stride = self.strides[2], padding = 1)
        self.conv4_3d_transpose = nn.ConvTranspose3d(in_channels = self.channels[3], out_channels = self.channels[4],
                                                     kernel_size = self.kernel_size, stride = self.strides[3], padding = 1)
        self.conv5_3d_transpose = nn.ConvTranspose3d(in_channels = self.channels[4], out_channels = self.channels[5],
                                                     kernel_size = self.kernel_size, stride = self.strides[4], padding = 1)
        self.batchnorm1 = nn.BatchNorm3d(self.channels[1])
        self.batchnorm2 = nn.BatchNorm3d(self.channels[2])
        self.batchnorm3 = nn.BatchNorm3d(self.channels[3])
        self.batchnorm4 = nn.BatchNorm3d(self.channels[4])
        


    def forward(self, x):
        x = x.view(-1, *x.shape) if len(x.shape) == 4 else x
        x = self.relu(self.batchnorm1(self.conv1_3d_transpose(x)))
        x = self.relu(self.batchnorm2(self.conv2_3d_transpose(x)))
        x = self.relu(self.batchnorm3(self.conv3_3d_transpose(x)))
        x = self.relu(self.batchnorm4(self.conv4_3d_transpose(x)))
        x = self.sigmoid(self.conv5_3d_transpose(x))
        return x 


'''
class GAN3D(nn.Module):
    def __init__(self):
        super(GAN3D, self).__init__()
'''

'''
tensor1 = torch.rand((1, 64, 64, 64))
tensor2 = torch.rand((200, 1, 1, 1))

generator = Generator_3DGAN()
descriminator = Discriminator_3DGAN()

print(descriminator(tensor1).shape)
print('-----------------------------')
print(generator(tensor2).shape)
'''

#print(summary(generator))
#print(summary(descriminator))



class AgeCGANEncoder(nn.Module):
    def __init__(self):
        super(AgeCGANEncoder, self).__init__()
        self.channels = (32, 64, 128, 256)
        self.linear_neurons_num = (4096, 100)
        self.kernel_size = 5
        self.stride = 2
        self.padding = 'same'
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.02)
        self.conv1 = nn.Conv2d(in_channels = self.channels[0], out_channels = self.channels[1], 
                               kernel_size = self.kernel_size, padding = self.padding)
        self.conv2 = nn.Conv2d(in_channels = self.channels[1], out_channels = self.channels[2], 
                               kernel_size = self.kernel_size, padding = self.padding)
        self.conv3 = nn.Conv2d(in_channels = self.channels[2], out_channels = self.channels[3], 
                               kernel_size = self.kernel_size, padding = self.padding)
        self.conv4 = nn.Conv2d(in_channels = self.channels[3], out_channels = self.channels[3], 
                               kernel_size = self.kernel_size, padding = self.padding)
        
        self.batchnorm1 = nn.BatchNorm2d(self.channels[1]) 
        self.batchnorm2 = nn.BatchNorm2d(self.channels[2]) 
        self.batchnorm3 = nn.BatchNorm2d(self.channels[3]) 
        self.batchnorm4 = nn.BatchNorm2d(self.linear_neurons_num[0])
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear('''in_features = ''', out_features = self.linear_neurons_num[0])
        self.linear2 = nn.Linear(*self.linear_neurons_num)


    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batchnorm1(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm2(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm3(self.conv4(x)))
        x = self.flatten(x)
        x = self.leaky_relu(self.batchnorm4(self.linear1(x)))
        x = self.linear2(x)
        return x
    

'''
class AgeCGANGenerator(nn.Module):
    def __init__(self):
        super(AgeCGANGenerator, self).__init__()
    

    def forward(self, x):
'''


'''
class AgeCGANDiscriminator(nn.Module):
    def __init__(self):
        super(AgeCGANDiscriminator, self).__init__()


    def forward(self, x):
'''