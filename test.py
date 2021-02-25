from ABMR_dataloader import MICCAI
from config import options
from torch.utils.data import DataLoader

train_dataset = MICCAI(mode='train', input_size=(768, 1024))
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=False, num_workers=options.num_workers, drop_last=False)

for i, data in enumerate(train_loader):
    x, y = data
    print(data)
    break;