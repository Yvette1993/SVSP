from __future__ import print_function

import torch.utils.data as data
import os
from PIL import Image
from utils import preprocess
import torchvision.transforms as tramsforms


class DepthEstimation(data.Dataset):


  def __init__(self, root, train=True, transform=None, target_transform=None, crop_size=None,download=False):
    self.root = root
    _depth_root = os.path.join(self.root, 'data/')
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
  

    if download:
      self.download()

    if self.train:
      #_list_f = os.path.join(_depth_root, 'train.txt')
      _list_f = os.path.join(_depth_root, 'test.txt')

    else:
      _list_f = os.path.join(_depth_root, 'val.txt')
    self.images = []
    self.masks = []
    with open(_list_f, 'r') as lines:
      for line in lines:
        _image = _depth_root + line.split()[0]
        _mask = _depth_root + line.split()[1]
        assert os.path.isfile(_image)
        assert os.path.isfile(_mask)
        self.images.append(_image)
        self.masks.append(_mask)

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')
    #print('_img=',_img)
    _target = Image.open(self.masks[index])
    #print('_target=',_target)
    
    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=self.crop_size,
                               )


   
    if self.transform is not None:
      _img = self.transform(_img)
      #print('_img=',_img.size())

    if self.target_transform is not None:
      _target = self.target_transform(_target)
      #print('_target=',_target.size())

    return _img, _target

  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')

