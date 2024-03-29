import os
import PIL
from datetime import datetime
from AMRDataset import AMRDataset
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision import transforms
import torch.backends.cudnn as cudnn

from models import *
from utils.eval_utils import compute_accuracy
from utils.logger_utils import Logger
from config import options
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(options.gpu_used)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Train command
# python train_classifier.py --epochs 500 --lr 1e-3 --single-optimizer Adam --batch-size 8 --log-interval 75
# --val_freq 150 --save-dir ./save/classifier --gpu-used 0 --label-mode samp --native_labels 2 --num_classes 2
# --img_h 512 --img_w 512 --loo 0


# Table of contents for classifier save folders
# (label mode was cons)
# 20201212_182753 - old dataloader                                              (val acc 80.73%)
# 29219195_231225 - new dataloader, lr 1e-3, SGD optimizer, batch size 16, label mode cons          (val acc 74.83%)
# 20210106_100618 - new dataloader, lr 1e-2, SGD optimizer, batch size 16, label mode cons          (val acc 77.89%)
# 20210106_100825 - new dataloader, lr 1e-3, Adam optimizer, batch size 16, label mode cons         (val acc 79.93%)
# 20210107_141639 - sanity check, same experiment                                                   (val acc 79.93%)

# (Changed AMR dataloader, now has min-max scaling, commented out imagenet normalization in this code)
# 20210108_082652 - new dataloader, lr 1e-3, Adam optimizer, batch size 16, label mode cons         (val acc 77.89%)

# (New dataloader, label sampling and can model 2 or 3 classes)
# 20210110_075034 - label mode samp, native labels 2, num_classes 2, Adam, lr 1e-3                  (val acc 86.83%)
# 20210110_111017 - label mode cons, native labels 2, num_classes 2, Adam, lr 1e-3                  (val acc 84.64%)

# (Added learning rate decay)
# 20210110_155329 - label mode samp, native labels 2, num_classes 2, Adam, lr 1e-3                  (val acc 86.12%)*Bug

# (Changed learning rate decay to calculate off of test_loss instead of test_acc. Also fixed
#   min-max scaling to be per image within the batch)
# 20210110_222910 - label mode samp, native labels 2, num_classes 2, Adam, lr 1e-3                  (val acc 64.41%)

# Removed min-max scaling from before classifier, uncommented min-max scaling in dataloader
# 20210111_093700 - label mode samp, native labels 2, num_classes 2, Adam, lr 1e-3                  (val acc 86.12%)

# Native labels 1 experiment
# 20210114_120952 - label mode samp, native labels 1, num_classes 2, Adam, lr 1e-3                  (val acc 85.26%)

# (512x512 experiments)
# 20210115_080801 - label sampling, test fold val 0, 512x512, batch size 8, lr 1e-3, test fold 0    (val acc 86.12%)
#  - label sampling, test fold val 0, 512x512, batch size 8, lr 1e-4, test fold 0    (val acc )


def train():
    log_string('Optimizer is ' + str(options.single_optimizer))
    log_string('lr is ' + str(options.lr))
    log_string('GPU Used: ' + str(options.gpu_used))
    log_string('Label mode: ' + str(options.label_mode))
    log_string('Native label num: ' + str(options.native_labels))
    log_string('Num_classes: ' + str(options.num_classes))
    log_string('Test fold val: ' + str(options.test_fold_val))

    global_step = 0
    best_loss = 100
    best_acc = 0

    for epoch in range(options.epochs):
        log_string('**' * 30)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()

        train_loss = 0
        targets, outputs = [], []

        for batch_id, (data, target) in enumerate(train_loader):
            global_step += 1

            data = data.cuda()
            target = target.cuda()

            # Min-max scaling
            # for i in range(len(data)):
            #     data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())  # Min-max scaling

            # Extra classifier transforms
            data = classifier_transforms(data)

            # Forward pass
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            train_loss += batch_loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (batch_id + 1) % options.log_interval == 0:
                train_loss /= options.log_interval
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))
                log_string("epoch: {0}, step: {1}, train_loss: {2:.4f} train_accuracy: {3:.02%}"
                           .format(epoch+1, batch_id+1, train_loss, train_acc))
                info = {'loss': train_loss,
                        'accuracy': train_acc}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                train_loss = 0
                targets, outputs = [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 30)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(best_loss=best_loss, best_acc=best_acc, global_step=global_step)
                net.train()


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    net.eval()
    test_loss = 0
    targets, outputs = [], []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            output = net(data)
            batch_loss = criterion(output, target)
            targets += [target]
            outputs += [output]
            test_loss += batch_loss

        test_loss /= (batch_id + 1)
        test_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs))
        scheduler.step(test_loss)

        # check for improvement
        loss_str, acc_str = '', ''
        if test_loss <= best_loss:
            loss_str, best_loss = '(improved)', test_loss
        if test_acc >= best_acc:
            acc_str, best_acc = '(improved)', test_acc

        # display
        log_string("validation_loss: {0:.4f} {1}, validation_accuracy: {2:.02%}{3}"
                   .format(test_loss, loss_str, test_acc, acc_str))

        # write to TensorBoard
        info = {'loss': test_loss,
                'accuracy': test_acc}
        for tag, value in info.items():
            test_logger.scalar_summary(tag, value, global_step)

        # save checkpoint model
        """
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        save_path = os.path.join(model_dir, '{}.ckpt'.format(global_step))
        torch.save({
            'global_step': global_step,
            'loss': test_loss,
            'acc': test_acc,
            'save_dir': model_dir,
            'state_dict': state_dict},
            save_path)
        log_string('Model saved at: {}'.format(save_path))
        """
        log_string('--' * 30)
        return best_loss, best_acc


if __name__ == '__main__':
    ##################################
    # Initialize saving directory
    ##################################
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Create subfolder in save directory for current run, named by current time
    save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)

    LOG_FOUT = open(os.path.join(save_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(options) + '\n')

    model_dir = os.path.join(save_dir, 'models')
    logs_dir = os.path.join(save_dir, 'tf_logs')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # backup of model def
    os.system('cp {}/models/densenet.py {}'.format(BASE_DIR, save_dir))
    # backup of train procedure
    os.system('cp {}/train_classifier.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/AMRDataset.py {}'.format(BASE_DIR, save_dir))

    ##################################
    # Creating the classifier model
    ##################################
    net = None
    if options.classifier_model == 'resnet':
        net = resnet.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
    elif options.classifier_model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
    elif options.classifier_model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
    elif options.classifier_model == 'densenet':
        net = densenet.densenet121(pretrained=True, drop_rate=0.2)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)

    log_string('{} model Generated.'.format(options.classifier_model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    classifier_transforms = transforms.Compose([
        transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.)),
        transforms.RandomRotation(90, fill=(0,)),  # resample=PIL.Image.BICUBIC
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.cuda()
    net = nn.DataParallel(net)
    ##################################
    # Loss and Optimizer
    ##################################
    criterion = nn.CrossEntropyLoss()  # Good for classification problems with distributions of output classes
    if options.single_optimizer == "Adam":
        optimizer = Adam(net.parameters(), lr=options.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if options.num_classes > 1 else 'max',
                                                         factor=0.5, patience=100)
    elif options.single_optimizer == "SGD":
        optimizer = SGD(net.parameters(), lr=options.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if options.num_classes > 1 else 'max',
                                                         factor=0.5, patience=100)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################

    train_dataset = AMRDataset(mode='train', input_size=(options.img_h, options.img_w), LOG_FOUT=LOG_FOUT)
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                              shuffle=True, num_workers=options.num_workers, drop_last=False)
    test_dataset = AMRDataset(mode='test', input_size=(options.img_h, options.img_w), LOG_FOUT=LOG_FOUT)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                             shuffle=False, num_workers=options.num_workers, drop_last=False)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()

