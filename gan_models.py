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
    


class AgeCGANGenerator(nn.Module):
    def __init__(self):
        super(AgeCGANGenerator, self).__init__()
        '''
        Input vector size: 100, conditional variable size: 6    
        '''
        self.noize_vec_dim, self.cond_var_dim = 100, 6 
        self.linear1 = nn.Linear(in_features = self.noize_vec_dim + self.cond_var_dim, out_features = 2048)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(in_features = 2048, out_features = 256 * 8 * 8)
        self.batchnorm1 = nn.BatchNorm2d(num_features = 256 * 8 * 8)
        self.reshape = torch.reshape            #(8, 8, 256)
        self.upsample = nn.Upsample((2, 2))
        self.kernel_size = 5
        self.padding = 'same'
        self.conv1 = nn.Conv2d(in_channels = 256, out_channels = 128, 
                              kernel_size = self.kernel_size, padding = self.padding)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 64, 
                              kernel_size = self.kernel_size, padding = self.padding)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 3, 
                              kernel_size = self.kernel_size, padding = self.padding)
        self.batchnorm2 = nn.BatchNorm2d(num_features = 128, momentum = 0.8)
        self.batchnorm3 = nn.BatchNorm2d(num_features = 64, momentum = 0.8)
        self.tanh = nn.Tanh()


    def forward(self, noize_vector, conditional_variable_vector):
        x = torch.cat(noize_vector, conditional_variable_vector)
        x = self.dropout(self.leaky_relu(self.linear1(x)))
        x = self.dropout(self.leaky_relu(self.batchnorm1(self.linear2(x))))
        
        x = self.reshape(x, (256, 8, 8))
        x = self.leaky_relu(self.batchnorm2(self.conv1(self.upsample(x))))
        x = self.leaky_relu(self.batchnorm3(self.conv2(self.upsample(x))))
        x = self.tanh(self.conv3(self.upsample(x)))
        return x


class AgeCGANDiscriminator(nn.Module):
    def __init__(self):
        super(AgeCGANDiscriminator, self).__init__()
        '''
        Input image shape: (3, 64, 64), conditional vector shape: (6,)
        '''
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)  #(64, 32, 32)
        self.conv2 = nn.Conv2d(in_channels = 70, out_channels = 128, kernel_size = 3, stride = 2, padding = 1) #(128, 16, 16)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1) #(256, 8, 8)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1) #(512, 4, 4)

        self.batchnorm1 = nn.BatchNorm2d(num_features = 128)
        self.batchnorm2 = nn.BatchNorm2d(num_features = 256)
        self.batchnorm3 = nn.BatchNorm2d(num_features = 512)

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(in_features = 512 * 4 * 4, out_features = 1)
        self.sigmoid = nn.Sigmoid()


    def expand_cond_vector(self, x):
        x = x.unsqueeze(1)
        x = x.unsqueeze(2)
        x = x.expand(-1, 32, 32)
        return x


    def forward(self, img, cond_vector):
        reshaped_cond_vector = self.expand_cond_vector(cond_vector)
        x = self.leaky_relu(self.conv1(img))
        x = torch.cat(x, reshaped_cond_vector) 
        x = self.leaky_relu(self.batchnorm1(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm2(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm3(self.conv4(x)))
        x = self.flatten(x)
        x = self.sigmoid(self.linear(x))
        return x
    
