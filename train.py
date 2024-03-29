import os
import PIL
from datetime import datetime
from ABMR_dataloader import ABMR_Dataset
from AMRDataset import AMRDataset
from GN_dataloader import GN_Dataset
import torch
import torch.nn as nn
from prettytable import PrettyTable
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torchvision import transforms
import torch.backends.cudnn as cudnn

from models import *
from efficientnet_pytorch import EfficientNet
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

def train():
    log_string('Training Initiated. Listing Training Configurations:')
    table = PrettyTable(['Name', 'Value'])
    table.add_row(['Parameters', sum(param.numel() for param in net.parameters())])
    table.add_row(['Log/Checkpoint Files', save_dir])
    for i in range(len(list(options.__dict__.keys()))):
        if list(options.__dict__.keys())[i] == 'load_model_path':
            continue
        else:
            table.add_row([list(options.__dict__.keys())[i], str(list(options.__dict__.values())[i])])
    log_string(str(table))

    global_step = 0
    best_loss = 100
    best_acc = 0

    for epoch in range(options.epochs):
        log_string('**' * 40)
        log_string('Training Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
        net.train()

        train_loss = 0
        targets, outputs = [], []

        for batch_id, (data, target) in enumerate(train_loader):
            len_batch = len(data)
            global_step += 1

            if options.classifier_model == 'morphset':
                # Reshape stacks into batch, [options.batchsize * options.stack_size, 3, options.img_h, options.img_w]
                data = data.view(len_batch * options.stack_size, 3, options.img_h, options.img_w)

            if options.cuda:
                data = data.cuda()
                target = target.cuda()

            # Forward pass
            output = net(data)
            if options.num_classes == 2:
                batch_loss = criterion(output, target.type_as(output).unsqueeze(1))
            else:
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
                train_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs), options.num_classes)
                log_string("epoch: {0}, step: {1}, train_loss: {2:.4f} train_accuracy: {3:.02%}"
                           .format(epoch+1, batch_id+1, train_loss, train_acc))
                info = {'loss': train_loss,
                        'accuracy': train_acc}
                for tag, value in info.items():
                    train_logger.scalar_summary(tag, value, global_step)
                train_loss = 0
                targets, outputs = [], []

            if (batch_id + 1) % options.val_freq == 0:
                log_string('--' * 40)
                log_string('Evaluating at step #{}'.format(global_step))
                best_loss, best_acc = evaluate(best_loss=best_loss, best_acc=best_acc, global_step=global_step)
                net.train()


def evaluate(**kwargs):
    best_loss = kwargs['best_loss']
    best_acc = kwargs['best_acc']
    global_step = kwargs['global_step']

    net.eval()

    with torch.no_grad():
        net_out = []
        net_loss = []
        if options.classifier_model == 'morphset':
            for _ in range(options.val_iters):
                test_loss = 0
                targets, outputs = [], []
                for batch_id, (data, target) in enumerate(test_loader):
                    len_batch = len(data)
                    data = data.view(len_batch * options.stack_size, 3, options.img_h, options.img_w)
                    if options.cuda:
                        data, target = data.cuda(), target.cuda()
                    output = net(data)
                    if options.num_classes == 2:
                        batch_loss = criterion(output, target.type_as(output).unsqueeze(1))
                    else:
                        batch_loss = criterion(output, target)
                    targets += [target]
                    outputs += [output]
                    test_loss += batch_loss
                net_out += [torch.cat(outputs)]
                net_loss += [test_loss]

            output_stack = torch.stack(net_out, dim=0).permute(1, 0, 2)
            loss_stack = torch.stack(net_loss)
            targets = torch.cat(targets)

            if options.num_classes == 2:
                output_stack = nn.functional.sigmoid(output_stack)
            else:
                output_stack = torch.softmax(output_stack, dim=2)  # Row-wise softmax

            out_mean = torch.mean(output_stack, dim=1)
            test_loss = torch.mean(loss_stack, dim=0)
            test_acc = compute_accuracy(targets, out_mean, options.num_classes)
        else:
            test_loss = 0
            targets, outputs = [], []
            for batch_id, (data, target) in enumerate(test_loader):
                if options.cuda:
                    data, target = data.cuda(), target.cuda()
                output = net(data)
                if options.num_classes == 2:
                    batch_loss = criterion(output, target.type_as(output).unsqueeze(1))
                else:
                    batch_loss = criterion(output, target)
                targets += [target]
                outputs += [torch.sigmoid(output)]
                test_loss += batch_loss

            test_loss /= (batch_id + 1)
            test_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs), options.num_classes)

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
        if options.efficient_save:
            if test_acc >= best_acc:
                state_dict = net.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()
                save_path = os.path.join(model_dir, 'TOP_MODEL_FOLD_{}.ckpt'.format(options.test_fold_val))
                torch.save({
                    'global_step': global_step,
                    'loss': test_loss,
                    'acc': test_acc,
                    'save_dir': model_dir,
                    'state_dict': state_dict},
                    save_path)
                log_string('Model saved at: {}'.format(save_path))
                log_string('--' * 40)
        else:
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
            log_string('--' * 40)
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
    os.system('cp {}/train.py {}'.format(BASE_DIR, save_dir))
    os.system('cp {}/ABMR_dataloader.py {}'.format(BASE_DIR, save_dir))

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
    elif 'efficientnet' in options.classifier_model:
        if options.num_classes == 2:
            outputs = 1
        else:
            outputs = options.num_classes
        net = EfficientNet.from_pretrained(options.classifier_model, include_top=True, num_classes=outputs)
    elif options.classifier_model == 'morphset':
        if options.encoder == 'resnet50':
            enc = resnet.resnet50(pretrained=True)
            enc = nn.Sequential(*(list(enc.children())[:-2]))
        elif 'efficientnet' in options.encoder:
            enc = EfficientNet.from_pretrained(options.encoder, include_top=True)
            a = list(enc.children())[0]
            b = list(enc.children())[1]
            c = list(enc.children())[2]
            cx = nn.Sequential(*list(c.children()))
            d = list(enc.children())[3]
            e = list(enc.children())[4]
            enc = nn.Sequential(*[a, b, cx, d, e])
        net = MorphSet(options.img_c, options.num_classes, enc, mode=options.ablation_mode)

    log_string('{} model Generated.'.format(options.classifier_model))
    log_string("Number of trainable parameters: {}".format(sum(param.numel() for param in net.parameters())))

    ##################################
    # Use cuda
    ##################################
    if options.cuda:
        cudnn.benchmark = True
        net.cuda()
        net = nn.DataParallel(net)
    ##################################
    # Loss and Optimizer
    ##################################
    if options.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if options.single_optimizer == "Adam":
        optimizer = Adam(net.parameters(), lr=options.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if options.num_classes > 1 else 'max',
                                                         factor=0.5, patience=100)
    elif options.single_optimizer == "SGD":
        optimizer = SGD(net.parameters(), lr=options.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if options.num_classes > 1 else 'max',
                                                         factor=0.5, patience=100)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # Load dataset
    ##################################
    if options.dataset == 'ABMR':
        if options.classifier_model == 'morphset':
            train_dataset = ABMR_Dataset(mode='train', input_size=(options.img_h, options.img_w))
            train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                                      shuffle=True, num_workers=options.num_workers, drop_last=False)
            test_dataset = ABMR_Dataset(mode='val', input_size=(options.img_h, options.img_w))
            test_loader = DataLoader(test_dataset, batch_size=options.val_batch_size,
                                     shuffle=False, num_workers=options.num_workers, drop_last=False)
        else:
            train_dataset = AMRDataset(mode='train', input_size=(options.img_h, options.img_w), LOG_FOUT=LOG_FOUT)
            train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                                  shuffle=True, num_workers=options.num_workers, drop_last=False)
            test_dataset = AMRDataset(mode='val', input_size=(options.img_h, options.img_w), LOG_FOUT=LOG_FOUT)
            test_loader = DataLoader(test_dataset, batch_size=options.val_batch_size,
                                 shuffle=False, num_workers=options.num_workers, drop_last=False)
    elif options.dataset == 'GN':
        train_dataset = GN_Dataset(mode='train', input_size=(options.img_h, options.img_w))
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                                  shuffle=True, num_workers=options.num_workers, drop_last=True)
        test_dataset = GN_Dataset(mode='val', input_size=(options.img_h, options.img_w))
        test_loader = DataLoader(test_dataset, batch_size=options.val_batch_size,
                                 shuffle=False, num_workers=options.num_workers, drop_last=True)

    ##################################
    # TRAINING
    ##################################
    log_string('')
    log_string('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
               format(options.epochs, options.batch_size, len(train_dataset), len(test_dataset)))
    train_logger = Logger(os.path.join(logs_dir, 'train'))
    test_logger = Logger(os.path.join(logs_dir, 'test'))

    train()
