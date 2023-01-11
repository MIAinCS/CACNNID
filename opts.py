import argparse

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='LUNA16_1v4_block', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes ')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
# parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
# parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--start_ib_epoch', default=1, type=int, help='start epoch for IB Loss')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='10', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='./log')
parser.add_argument('--root_model', type=str, default='./checkpoint')
parser.add_argument('--subset_index',type=str, default='subset0')
parser.add_argument('--balanced_mixup',type=bool, default=True)
parser.add_argument('--mixup_a',type=float,default=0.1)
parser.add_argument('--mixup_b',type=float,default=1.0)
parser.add_argument('--T_max',type=int,default=4)
parser.add_argument('--total_epoch',type=int,default=1)
parser.add_argument('--start_loss',type=float,default=0.5)
parser.add_argument('--warmup',type=bool, default=False)
parser.add_argument('--scheduler',type=str,default='CosineAnnealingLR')
parser.add_argument('--T_mult',type=int,default=1)
parser.add_argument('--block_index',type=int,default=0)
parser.add_argument('--optimizer',type=str,default='SGD')
parser.add_argument('--save_epoch',type=int,default=0)