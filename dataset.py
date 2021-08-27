from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gc
import torchvision.transforms as transforms
import os
import torchvision
import torch.nn.functional as F
import numpy as np



class UnetDataset(Dataset):

    def __init__(self, img_path, mask_path, transform = None) -> None:
        super(UnetDataset, self).__init__()

        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        
    def __len__(self):

        return len(os.listdir(self.img_path))

    def __getitem__(self, index):

        img = Image.open(self.img_path+"/"+os.listdir(self.img_path)[index])
        mask = Image.open(self.mask_path+"/"+os.listdir(self.mask_path)[index])
        #img = torchvision.io.read_image(self.img_path+"/"+os.listdir(self.img_path)[index])
        #image = torchvision.io.decode_png(image)
        
        #mask = torchvision.io.read_image(self.mask_path+"/"+os.listdir(self.mask_path)[index])
        #mask = torchvision.io.decode_png(mask)
        masks = []
        mask = np.asarray(mask)
#         mask = Image.open(self.label_arr[index])
#         mask = mask.resize((256, 256))
#         mask = np.asarray(mask)
        
        for i in range(13):
            cls_mask = np.where(mask == i, 255, 0)
            cls_mask = cls_mask.astype('float')
            masks.append(cls_mask[:,:,0]/255)
            
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks = transforms.Resize((512,512))(masks)
        
        
        if self.transform:
            img = self.transform(img)
            #masks = self.transform(mask)

        #mask = torch.max(mask, dim = 0, keepdim = True)
        #mask = mask[1].type(torch.LongTensor)

        return img, masks