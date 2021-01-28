import os
import random
import torch
from torchvision import transforms
from config import options
from torch.utils.data import Dataset
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(options.gpu_used)
device = torch.device("cuda" if options.cuda else "cpu")


class MICCAI(Dataset):
    def __init__(self, mode="train", input_size=(256, 256)):
        # self.imgs_dir = options.dataset  # Path to AMR dataset
        self.imgs_dir = '/home/cougarnet.uh.edu/sdpatiba/Desktop/MICCAI/ABMR_dataset/AMR_raw_gloms/'
        self.mode = mode
        self.input_size = input_size
        self.label_mode = options.label_mode
        self.val = options.test_fold_val
        self.label_code = {"Non-AMR": 0, "AMR": 1, "Inconclusive": 2, "None": 3}
        self.patients = {}
        self.batchsize = options.batch_size
        self.labels = None
        self.images = None

        glom_files = sorted(os.listdir(self.imgs_dir))
        print("Total length: ",len(glom_files))

        for file in glom_files:
            patient_name_list = file.split('-')
            patient_name = "-".join(patient_name_list[:-1])
            if patient_name in self.patients:
                self.patients[patient_name].append(file)
            else:
                self.patients[patient_name] = [file]

        for patient in self.patients:
            default_images = len(self.patients[patient])
            images_needed = 0
            if (default_images // self.batchsize) < 1:
                images_needed = self.batchsize - default_images
            elif default_images % self.batchsize == 0:
                images_needed = 0
            else:
                images_needed = (((self.batchsize // self.batchsize) + 1) * self.batchsize) - default_images

            # Randomly sample an image.
            print("Images needed: ", images_needed)
            for i in range(images_needed):
                original = random.sample(self.patients[patient], 1)
                # Removes file extension
                file_name = original[0][:-4]
                original = Image.open(self.imgs_dir + original[0])
                copyImage = original.copy()
                copyfileName = file_name + '-stylized-' + str(i + 1) + '.jpg'
                print("copyfileName: ",copyfileName)
                copyImage.save(self.imgs_dir + copyfileName)
                self.patients[patient].append(copyfileName)
                break
            break
        self.images = self.patients[list(self.patients.keys())[0]]  # List of all images for the first patient
        if self.images[0][0] == 'A':
            self.labels = ['AMR']*len(self.images)
        else:
            self.labels = ['Non-AMR']*len(self.images)

        print("Images: ", self.images)
        print("Labels: ", self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Open image
        image_name = self.images[index]
        img = Image.open(self.imgs_dir + image_name)  # Are other conversions needed?
        target_label = self.labels[index]  # label_bunch for the image indexed
        if self.mode == 'train' and options.label_mode == 'samp':
            target_label = random.choice(target_label)

        # Min-max scaling
        img = transforms.ToTensor()(img)
        # img = img.div(255.)
        img = (img - img.min()) / (img.max() - img.min())

        if self.mode == 'train':
            # normalization & augmentation
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)  # Bilinear resizing?
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomVerticalFlip()(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            # img = transforms.RandomResizedCrop(self.input_size[0], scale=(0.7, 1.))(img)
            # img = transforms.RandomRotation(90, resample=PIL.Image.BICUBIC)(img)
            # img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05)(img)

        if self.mode == 'val':
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

        if self.mode == 'test':
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

        return img, self.label_code[target_label]
