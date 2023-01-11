import math
import torch.backends.cudnn as cudnn
import random
import warnings
from tensorboardX import SummaryWriter
import torch.nn as nn
import time
from sklearn.metrics import roc_auc_score

import models
from opts import parser
from utils import *
from load_extract_data import MyData
from losses import IB_FocalLoss, IBLoss
from get_loaders import get_combo_loader
import torch.nn.functional as F

def main(args):
    args.store_name = '_'.join([args.dataset,str(args.batch_size),str(args.mixup_a),str(args.mixup_b),args.subset_index,str(args.block_index),str(args.lr),str(args.T_max),args.scheduler,args.exp_str])
    print(args.store_name)
    prepare_folders(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    if args.subset_index == 'all':
        for i in range(10):
            print('*'*20+f' Start subset{i}_block{args.block_index}! '+ '*'*20)
            main_worker(args, test_subset='subset'+str(i))
    else:
        print('*'*20+f' Start subset{args.subset_index}_block{args.block_index}! '+ '*'*20)
        main_worker(args)

def main_worker(args):
    global best_auc
    best_auc = 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = args.num_classes
    model = models.__dict__[args.arch](num_classes=num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:
        warnings.warn('Optimizer is not listed')

    if args.dataset is not None:
        train_dataset = MyData(os.path.join('/home/haokexin/lungs_classification/imbalance_experience/data',args.dataset,str(args.subset_index),str(args.block_index),'train.csv'))
        val_dataset = train_dataset
        test_dataset = MyData(os.path.join('/home/haokexin/lungs_classification/imbalance_experience/data',args.dataset,str(args.subset_index),str(args.block_index),'test.csv'))
    else:
        warnings.warn('Dataset is not listed')
        return
    
    args.cls_num_list = train_dataset.get_cls_num_list()
    print(args.cls_num_list)

    train_sampler = None

    # data_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers=args.workers)
    if args.balanced_mixup:
        # balanced-mixup
        train_loader = get_combo_loader(train_loader)

    if args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max, eta_min=1e-4)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=4,T_mult=args.T_mult,eta_min=1e-4)
    else:
        warnings.warn('error scheduler')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True 
    
    # init log for training
    log_valing = open(os.path.join(args.root_log, args.store_name, 'log_val.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')

    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'IBReweight':
            train_sampler = None
            per_cls_weights = 1.0 / np.array(args.cls_num_list)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')
    
        criterion_ib = None
        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'IB':
                criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)
                criterion_ib = IBLoss(weight=per_cls_weights, alpha=1000).cuda(args.gpu)
        elif args.loss_type == 'IBFocal':
            criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)
            criterion_ib = IB_FocalLoss(weight=per_cls_weights, alpha=4, gamma=1).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        train(train_loader, model, criterion, criterion_ib, optimizer, epoch, args, log_valing, tf_writer)
        
        # validate trainset
        losses, auc = validate(val_loader, model, criterion, criterion_ib, epoch, args, log_valing, tf_writer)
        scheduler.step()

        is_best = auc > best_auc
        best_auc = max(auc, best_auc)

        tf_writer.add_scalar('acc/train_best_auc', best_auc, epoch)
        output_best = 'Best AUC: %.3f\n' % (best_auc)
        print(output_best)
        log_valing.write(output_best + '\n')
        log_valing.flush()

        if epoch > args.save_epoch:
            # validate testset
            losses, auc = validate(test_loader, model, criterion, criterion_ib, epoch, args, log_testing, tf_writer, flag = 'test')
            
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_auc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch)

def train(train_loader, model, criterion, criterion_ib, optimizer, epoch, args, log, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    outputs_list = []
    targets_list = []

    # switch to train mode
    model.train()

    end = time.time()
    for data in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        if args.balanced_mixup:
            lam = np.random.beta(a=args.mixup_a, b=args.mixup_b)
            imgs, targets, _ = data[0]
            balanced_imgs, balanced_targets, _ = data[1]
            if args.gpu is not None:
                imgs, targets = imgs.cuda(args.gpu, non_blocking=True), targets.cuda(args.gpu, non_blocking=True)
                balanced_imgs, balanced_targets = balanced_imgs.cuda(args.gpu, non_blocking=True), balanced_targets.cuda(args.gpu, non_blocking=True)
            input = (1-lam) * balanced_imgs + lam * imgs
            target = (1-lam) * F.one_hot(balanced_targets, 2) + lam * F.one_hot(targets, 2)
            del balanced_imgs
            del balanced_targets
        else:
            if args.gpu is not None:
                input = data[0].cuda(args.gpu, non_blocking=True)
                target = data[1].cuda(args.gpu, non_blocking=True)
        
        # compute output
        if 'IB' in args.loss_type and epoch >= args.start_ib_epoch:
            output, features = model(input)
            loss = criterion_ib(output, target, features)
        else:
            output, _ = model(input)
            loss = criterion(output, target)
        
        # record loss
        losses.update(loss.item(), input.size(0))
        outputs_list.extend(output.cpu())
        targets_list.extend(target.cpu())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    lr = optimizer.param_groups[-1]['lr']
    tf_writer.add_scalar('lr', lr, epoch)


def validate(val_loader, model, criterion, criterion_ib, epoch, args, log=None, tf_writer=None, flag='val'):
    losses = AverageMeter('Loss', ':.4e')

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for input, target, _ in val_loader:
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            
            # compute output
            if 'IB' in args.loss_type and epoch >= args.start_ib_epoch:
                output, features = model(input)
                loss = criterion_ib(output, target, features)
                # Note that while the loss is computed using target, the target is not used for prediction.
            else:
                output, _ = model(input)
                loss = criterion(output, target)
            
            losses.update(loss.item(), input.size(0))

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        TP = cf[1,1]
        TN = cf[0,0]
        FP = cf[0,1]
        FN = cf[1,0]

        auc = roc_auc_score(all_targets, all_preds)
        accuracy = (TP+TN)/float(TP+TN+FP+FN)
        Sensitivity = TP/float(TP+FN)
        Specificity = TN/float(TN+FP)

        output = ('Epoch: [{epoch}] {flag} Results: Sensitivity {Sensitivity} Specificity {Specificity} Accuracy {accuracy} AUC {auc:.4f} Loss {loss.avg:.5f}'
                .format(epoch=epoch,flag=flag, Sensitivity=Sensitivity , Specificity=Specificity,accuracy = accuracy, auc=auc, loss=losses))
        print(output)

        if log is not None:
            log.write(output + '\n')
            log.flush()
        
        tf_writer.add_scalar('loss/'+ flag, losses.avg, epoch)
        tf_writer.add_scalars('results/' + flag, {'Sensitivity':Sensitivity,'Specificity':Specificity,'Accuracy':accuracy, 'AUC':auc}, epoch)
    return losses, auc

if __name__ == '__main__':
    args = parser.parse_args()

    block_list = []

    if len(block_list)!= 0:
        for i in block_list:
            args.block_index = i
            main(args)
    else:
        main(args)

    
