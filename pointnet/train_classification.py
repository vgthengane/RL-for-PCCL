"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

def setup_distributed(args):
    args.distributed = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        backend = "nccl" if (torch.cuda.is_available() and not args.use_cpu) else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if backend == "nccl":
            torch.cuda.set_device(args.local_rank)
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    return args


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


import datetime
import logging
import time
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--data_path', type=str, default='', required=True, help='data path')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40, device=None):
    classifier = model.eval()
    if device is None:
        device = torch.device("cpu") if args.use_cpu else torch.device("cuda")

    total_correct = torch.zeros(1, device=device, dtype=torch.float32)
    total_seen = torch.zeros(1, device=device, dtype=torch.float32)
    class_correct = torch.zeros(num_class, device=device, dtype=torch.float32)
    class_seen = torch.zeros(num_class, device=device, dtype=torch.float32)

    with torch.no_grad():
        for j, (points, target) in tqdm(
            enumerate(loader),
            total=len(loader),
            disable=not is_main_process()
        ):
            if not args.use_cpu:
                points = points.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            points = points.transpose(2, 1)
            pred, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long())
            total_correct += correct.sum()
            total_seen += target.numel()

            for cls in range(num_class):
                cls_mask = target == cls
                class_seen[cls] += cls_mask.sum()
                class_correct[cls] += correct[cls_mask].sum()

    if args.distributed:
        all_reduce_tensor(total_correct)
        all_reduce_tensor(total_seen)
        all_reduce_tensor(class_correct)
        all_reduce_tensor(class_seen)

    class_acc = (class_correct / class_seen.clamp(min=1)).mean().item()
    instance_acc = (total_correct / total_seen.clamp(min=1)).item()

    return instance_acc, class_acc


def main(args):
    '''HYPER PARAMETER'''
    args = setup_distributed(args)
    if args.use_cpu or (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        if args.distributed:
            device = torch.device("cuda", args.local_rank)
        else:
            try:
                gpu_id = int(str(args.gpu).split(",")[0])
                torch.cuda.set_device(gpu_id)
                device = torch.device("cuda", gpu_id)
            except (TypeError, ValueError):
                device = torch.device("cuda")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('_experiments/')
    exp_dir = exp_dir.joinpath('classification')
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    log_dir = exp_dir.joinpath('logs/')
    if is_main_process():
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
    if args.distributed:
        dist.barrier()

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        if is_main_process():
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            logger.addHandler(logging.NullHandler())

    def log_string(message):
        if is_main_process():
            logger.info(message)
            print(message)

    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = args.data_path

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=10,
        drop_last=True,
        pin_memory=not args.use_cpu
    )

    testDataLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=10,
        pin_memory=not args.use_cpu
    )

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    if is_main_process():
        shutil.copy('./models/%s.py' % args.model, str(exp_dir))
        shutil.copy('models/pointnet2_utils.py', str(exp_dir))
        shutil.copy('./train_classification.py', str(exp_dir))
    if args.distributed:
        dist.barrier()

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.to(device)
        criterion = criterion.to(device)
        if args.distributed:
            classifier = DDP(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.distributed:
        classifier = DDP(classifier)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth', map_location=device)
        start_epoch = checkpoint['epoch']
        if args.distributed:
            classifier.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    instance_acc = 0.0
    class_acc = 0.0

    '''TRANING'''
    log_string('Start training...')
    training_start_time = time.time()
    for epoch in range(start_epoch, args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        epoch_start_time = time.time()
        total_correct = torch.zeros(1, device=device, dtype=torch.float32)
        total_seen = torch.zeros(1, device=device, dtype=torch.float32)
        classifier = classifier.train()

        # scheduler.step()  # Removed as per instructions
        for batch_id, (points, target) in tqdm(
            enumerate(trainDataLoader, 0),
            total=len(trainDataLoader),
            smoothing=0.9,
            disable=not is_main_process()
        ):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long()).sum()
            total_correct += correct
            total_seen += target.numel()
            loss.backward()
            optimizer.step()
            global_step += 1

        if args.distributed:
            all_reduce_tensor(total_correct)
            all_reduce_tensor(total_seen)
        train_instance_acc = (total_correct / total_seen.clamp(min=1)).item()
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        if epoch % args.save_freq == 0:
            with torch.no_grad():
                instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class, device=device)

                if (instance_acc >= best_instance_acc):
                    best_instance_acc = instance_acc
                    best_epoch = epoch + 1

                if (class_acc >= best_class_acc):
                    best_class_acc = class_acc
                log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
                log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

                if (instance_acc >= best_instance_acc) and is_main_process():
                    log_string('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': best_epoch,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.module.state_dict() if args.distributed else classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
            
        savepath = str(checkpoints_dir) + '/last_model.pth'
        log_string('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'instance_acc': instance_acc,
            'class_acc': class_acc,
            'model_state_dict': classifier.module.state_dict() if args.distributed else classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

        epoch_time = time.time() - epoch_start_time
        log_string('Epoch Training Time: %.2f seconds' % epoch_time)

        scheduler.step()
        global_epoch += 1

    total_training_time = time.time() - training_start_time
    log_string('Total Training Time: %.2f seconds (%.2f hours)' % (total_training_time, total_training_time/3600))
    log_string('End of training...')
    if args.distributed:
        dist.barrier()


if __name__ == '__main__':
    args = parse_args()
    main(args)
