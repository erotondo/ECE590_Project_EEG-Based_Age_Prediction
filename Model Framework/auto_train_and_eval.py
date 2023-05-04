import argparse
import os
from collections import OrderedDict
from glob import glob

import cv2
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import CNNDataset, LSTMDataset
from metrics import mae_score, male_score
from utils import AverageMeter, str2bool

import time

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                       metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='CNNAgePrediction',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: CNNAgePrediction)')
    
    # loss
    parser.add_argument('--loss', default='MALELoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: MALELoss)')
    
    # dataset
    parser.add_argument('--group',
                        help='participant diagnosis')
    parser.add_argument('--condition', default='restEC',
                        help='restEC or restEO (Default: restEC)')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),  
                  'mae_score': AverageMeter(),
                  'male_score': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    #print("length of dataloader from inside the function is " + str(len(train_loader)))
    #print(train_loader)
    for input, target, meta in train_loader:
        #print(input.shape)
        #print(target.shape)
        output = model(input)
        #print(output.shape)
        loss = criterion(output, target)
        mae = mae_score(output, target)
        male = male_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['mae_score'].update(mae, input.size(0))
        avg_meters['male_score'].update(male, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('mae_score', avg_meters['mae_score'].avg),
            ('male_score', avg_meters['male_score'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('mae_score', avg_meters['mae_score'].avg),
                        ('male_score', avg_meters['male_score'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'mae_score': AverageMeter(),
                  'male_score': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, meta in val_loader:
            # compute output
            output = model(input)
            loss = criterion(output, target)
            mae = mae_score(output, target)
            male = male_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['mae_score'].update(mae, input.size(0))
            avg_meters['male_score'].update(male, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('mae_score', avg_meters['mae_score'].avg),
                ('male_score', avg_meters['male_score'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('mae_score', avg_meters['mae_score'].avg),
                        ('male_score', avg_meters['male_score'].avg)])

def cust_collate_fn(batch):
    if (len(batch) == 1):
        return torch.from_numpy(np.float32(batch[0][0])), torch.from_numpy(np.float32(batch[0][1])), batch[0][2]

    data = np.concatenate([batch[0][0], batch[1][0]],axis=0)
    for i in range(len(batch)-2):
        data = np.concatenate([data, batch[i+2][0]],axis=0)

    targets = np.concatenate([batch[0][1], batch[1][1]],axis=0)
    for i in range(len(batch)-2):
        targets = np.concatenate([targets, batch[i+2][1]],axis=0)
    
    labels = batch[0][2]

    return torch.from_numpy(np.float32(data)), torch.from_numpy(np.float32(targets)), labels

def main_func(modelName, fileName, train_set, val_set, cond, metadata):
    config = vars(parse_args())
    config['name'] = modelName
    fw = open('batch_results_train/'+ fileName, 'w')
    print('config of group is ' + str(config['group']))
    fw.write('config of group is ' + str(config['group']) + '\n')    
    if config['name'] is None:
        config['name'] = '%s_%s_woDS' % (config['group'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    fw.write('-' * 20 + '\n')
    for key in config:
        print('%s: %s' % (key, config[key]))
        fw.write('%s: %s' % (key, config[key]) + '\n')
    print('-' * 20)
    fw.write('-' * 20 + '\n')
    #TODO print parameters manually i think, all imports to function

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    criterion = losses.__dict__[config['loss']]()#.cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')   
    model = archs.__dict__[config['arch']]()

    model = model#.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    data_dir = os.path.join('inputs', config['group'] + '_group')

    if config['arch'] == 'CNNAgePrediction':
        dataset_type = CNNDataset
    elif config['arch'] == 'LSTMAgePrediction':
        dataset_type = LSTMDataset
    
    train_dataset = dataset_type(
        participants_ids=train_set,
        preproc_eeg_data_dir=data_dir,
        condition=cond,
        partipant_metadata=metadata)
    val_dataset = dataset_type(
        participants_ids=val_set,
        preproc_eeg_data_dir=data_dir,
        condition=cond,
        partipant_metadata=metadata)

    #print("length of train dataset is " + str(len(train_dataset)))
    #print("length of val dataset is " + str(len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        collate_fn=cust_collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        collate_fn=cust_collate_fn)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('mae_score', []),
        ('male_score', []),
        ('val_loss', []),
        ('val_mae_score', []),
        ('val_male_score', []),
    ])

    trigger = 0
    best_loss = np.inf
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        fw.write('Epoch [%d/%d]' % (epoch, config['epochs']) + '\n')    

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - mae_score %.4f - male_score %.4f - val_loss %.4f - val_mae_score %.4f - val_male_score %.4f'
              % (train_log['loss'], train_log['mae_score'], train_log['male_score'], val_log['loss'], val_log['mae_score'], val_log['male_score']))
        fw.write('loss %.4f - mae_score %.4f - male_score %.4f - val_loss %.4f - val_mae_score %.4f - val_male_score %.4f'
              % (train_log['loss'], train_log['mae_score'], train_log['male_score'], val_log['loss'], val_log['mae_score'], val_log['male_score']) + '\n')

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['mae_score'].append(train_log['mae_score'])
        log['male_score'].append(train_log['male_score'])
        log['val_loss'].append(val_log['loss'])
        log['val_mae_score'].append(val_log['mae_score'])
        log['val_male_score'].append(val_log['male_score'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1
        if val_log['loss'] < best_loss:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_loss = val_log['loss']
            print("=> saved best model")
            fw.write("=> saved best model" + '\n')
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            fw.write("=> early stopping" + '\n')
            break

        #torch.cuda.empty_cache()

def perform_validation(modelName, fileName, test_set, cond, metadata):
    fw = open('batch_results_test/' + fileName, 'w') 
    #with open('models/%s/config.yml' % args.name, 'r') as f:
    with open('models/%s/config.yml' % modelName, 'r') as f:   
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    fw.write('-'*20 + '\n')
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
        fw.write('%s: %s' % (key, str(config[key])) + '\n')
    print('-'*20)
    fw.write('-'*20 + '\n')

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    fw.write("=> creating model %s" % config['arch'] + '\n')
    model = archs.__dict__[config['arch']]()

    model = model#.cuda()

    data_dir = os.path.join('inputs', config['group'] + '_group')

    if config['arch'] == 'CNNAgePrediction':
        dataset_type = CNNDataset
    elif config['arch'] == 'LSTMAgePrediction':
        dataset_type = LSTMDataset

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    test_dataset = dataset_type(
        participants_ids=test_set,
        preproc_eeg_data_dir=data_dir,
        condition=cond,
        partipant_metadata=metadata)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        collate_fn=cust_collate_fn)


    mae_avg_meter = AverageMeter()
    male_avg_meter = AverageMeter()

    os.makedirs(os.path.join('outputs', config['name']), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            # compute output
            output = model(input)
            mae = mae_score(output, target)
            male = male_score(output, target)

            mae_avg_meter.update(mae, input.size(0))
            male_avg_meter.update(male, input.size(0))

    print('MAE: %.4f' % mae_avg_meter.avg)
    fw.write('MAE: %.4f' % mae_avg_meter.avg)
    print('MALE: %.4f' % male_avg_meter.avg)
    fw.write('MALE: %.4f' % male_avg_meter.avg)

    #torch.cuda.empty_cache()

def main():
    params = vars(parse_args())

    meta = pd.read_csv('./inputs/'+params['group']+'_group/'+params['group']+'_subjects.csv')
    split_lists = []
    for i in range(5):
        split_lists.append(np.load('./inputs/'+params['group']+'_group/split_'+str(i+1)+'.npy'))

    split_lists = np.array(split_lists, dtype=object)

    for i in range(5):
        for j in range(5):
            if i == j:
                continue
                
            test_set = split_lists[i]
            val_set = split_lists[j]
            train_set = np.concatenate(np.delete(split_lists, [i,j]))

            modelName = params['name'] + '_Val_' + str(j) + '_Test_' + str(i)
            trainFileName = params['name'] + '_' + params['condition'] + '_trainingResult_Val_' + str(j) + '_Test_' + str(i)
            testFileName = params['name'] + '_' + params['condition'] + '_testResult_Val_' + str(j) + '_Test_' + str(i)
            main_func(modelName, trainFileName, train_set, val_set, params['condition'], meta)
            perform_validation(modelName, testFileName, test_set, params['condition'], meta)

if __name__ == '__main__':
    # python auto_train_and_eval.py --name CNN_healthy_only --epochs 50 --batch_size 2 --arch CNNAgePrediction --group healthy --condition restEC
    # python auto_train_and_eval.py --name LSTM_healthy_only --epochs 50 --batch_size 2 --arch LSTMAgePrediction --group healthy --condition restEC
    
    # OSError: [Errno 24] Too many open files -> ulimit -n 4096
    main()