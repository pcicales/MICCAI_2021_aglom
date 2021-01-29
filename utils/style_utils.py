<<<<<<< d1c5680939a62d1e44c8685d03673cfd1a2d569c
from PIL import Image
import numpy as np


def load_image(filename, size=256, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.BILINEAR)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)  # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

    # From pytorch gatys method
    # features = y.view(b * ch, h * w)
    # G = torch.mm(features, features.t())
    # return G.div(b * ch * h * w)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)  # new_tensor for same dimension of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)  # back to tensor within 0, 1
    return (batch - mean) / std
=======
from PIL import Image
import numpy as np


def load_image(filename, size=256, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.BILINEAR)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)  # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

    # From pytorch gatys method
    # features = y.view(b * ch, h * w)
    # G = torch.mm(features, features.t())
    # return G.div(b * ch * h * w)


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)  # new_tensor for same dimension of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)  # back to tensor within 0, 1
    return (batch - mean) / std
>>>>>>> first commit
