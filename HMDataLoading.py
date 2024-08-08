
import torch
from torch.utils.data import Dataset
import skimage as ski
import numpy as np
from preprocess import pad_image


class HMDataset(Dataset):

  def __init__(self, imageList, transformations=None):
    self.transforms = transformations
    self.imageDict = {}
    with open(imageList, "r") as image_files:
      for i, image_file in enumerate(image_files):
        self.imageDict[i] = image_file.strip()

  def __getitem__(self, item):
    image = np.load(self.imageDict[item])
    image = torch.from_numpy(image)
    if self.transforms is not None:
      image = self.transforms(image)
    return image, self.imageDict[item]

  def __len__(self):
    return len(self.imageDict)


class HMDatasetWithPreProc(Dataset):

  def __init__(self, imageList, pad_size=(1750,3494), image_size=None, transformations=None):
    self.padSize = pad_size
    self.imageSize = image_size
    self.transforms = transformations
    self.imageDict = {}
    with open(imageList, "r") as image_files:
      for i, image_file in enumerate(image_files):
        self.imageDict[i] = image_file.strip()

  def __getitem__(self, item):
    image = ski.io.imread(self.imageDict[item])
    #image = pad_image(image)
    image = pad_image(image, self.padSize[0], self.padSize[1])
    image = ski.util.img_as_float(image)
    if self.imageSize is not None:
      image = ski.transform.resize(image, self.imageSize)
    image = torch.from_numpy(image).float().permute(2,0,1)
    if self.transforms is not None:
      image = self.transforms(image)
    return image

  def __len__(self):
    return len(self.imageDict)


