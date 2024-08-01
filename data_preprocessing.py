import scipy.io as io
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import os 
from torch.utils.data import Dataset
import torch

#voxels = io.loadmat("path to file.mat")['instance']

#voxels = np.pad(voxels, (1, 1), 'constant', constant_values = (0, 0))

#voxels = nd.zoom(voxels, (2, 2, 2), mode = 'constant', order = 0)

'''
Image visualization block 

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.set_aspect('equal')

ax.voxels(voxels, edgecolor="red")
plt.show()
plt.savefig(file_path)
'''

'''
voxels = io.loadmat("D:\Github repos\GANs-and-Roses\Datasets\\3DShapeNets\\volumetric_data\\airplane\\30\\test\\1c8fa980dc87bd3a16af5d7a0b735543_1.mat")
voxels = np.pad(voxels['instance'], (1, 1), 'constant', constant_values = (0, 0))
voxels = nd.zoom(voxels, (2, 2, 2), mode = 'constant', order = 0)
print(voxels.shape)

ax = plt.figure().add_subplot(projection = '3d')
ax.set_aspect('equal')
ax.voxels(voxels, facecolors = "red", edgecolor = "red")
plt.show()
'''

class DataProcessing:
    def __init__(self, general_volume_data_path):
        self.general_volume_data_path = general_volume_data_path #os.path.normpath(general_volume_data_path)
        self.folder_names = self.get_folders_names()
        self.folder_pathes = self.get_folders_pathes()
        self.images_pathes_dict = self.get_pathes_dict()
        

    def get_folders_names(self):
        return os.listdir(self.general_volume_data_path)
    

    def get_folders_pathes(self):
        folder_pathes = []
        for folder in os.listdir(self.general_volume_data_path):
            folder_pathes.append(os.path.join(self.general_volume_data_path, folder))

        return folder_pathes


    def get_pathes_dict(self):
        '''
        {'category': {'train': [list of train pathes images], 'test': [list of test pathes images]}}
        '''
        pathes_dict = {}
        for folder in self.folder_pathes:
            temp_list = ["test", "train"]
            test_folder_path = os.path.join(folder + "\\30", temp_list[0])
            train_folder_path = os.path.join(folder + "\\30", temp_list[1])
            test_pathes = [os.path.join(test_folder_path, matfile) for matfile in os.listdir(test_folder_path)]
            train_pathes = [os.path.join(train_folder_path, matfile) for matfile in os.listdir(train_folder_path)]
            pathes_dict[folder.split("\\")[-1]] = {"test": test_pathes, "train": train_pathes}
        
        return pathes_dict
    
    
    def image_voxels(self, image_path):
        voxels = io.loadmat(image_path)
        voxels = np.pad(voxels['instance'], (1, 1), 'constant', constant_values = (0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode = 'constant', order = 0)
        return voxels



class GAN3dDataset(Dataset):
    def __init__(self, data_processing, cathegory_name, train = True, return_labels = False):
        self.data_processing = data_processing
        self.cathegory_name = cathegory_name
        self.set_type = "train" if train else "test"
        self.pathes_list = self.data_processing.images_pathes_dict[self.cathegory_name][self.set_type]
        self.return_labels = return_labels


    def __len__(self):
        return len(self.pathes_list)


    def __getitem__(self, index):
        x = self.data_processing.image_voxels(self.pathes_list[index])
        x = torch.reshape(torch.from_numpy(x), (1, 64, 64, 64))
        x = x.to(dtype = torch.float32)
        if self.return_labels:
            return x, torch.tensor([1.])    
        else:
            return x

'''
dataprocessing = DataProcessing("D:\\Github repos\\GANs-and-Roses\\Datasets\\3DShapeNets\\volumetric_data")
print(dataprocessing.folder_names)
print(dataprocessing.images_pathes_dict['airplane']['train'][:2])
'''
#print(dataprocessing.images_pathes_dict.keys())
#print(dataprocessing.folder_pathes)
'''
dataprocessing = DataProcessing("D:\\Github repos\\GANs-and-Roses\\Datasets\\3DShapeNets\\volumetric_data")
gan_dataset = GAN3dDataset(dataprocessing, "airplane")
tensor_image = gan_dataset.__getitem__(5)
print(tensor_image.shape, tensor_image.dtype)
'''
