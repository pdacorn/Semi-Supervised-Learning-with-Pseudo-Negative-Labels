import os
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from third_party.mean_teacher import data
from third_party.mean_teacher import mt_func
from third_party.mean_teacher.utils import *
from third_party.mean_teacher.data import NO_LABEL
from tqdm import tqdm
from src import ramps, losses, cli, run_context, datasets
from src import art as architectures

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def create_data_loaders(train_transformation, eval_transformation, datadir, args):

    traindir = os.path.join(datadir, args.train_subdir)
    traindir = '../third_party/data-local/images/cifar/cifar10/by-image/train+val'
    print(traindir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    evaldir = '../third_party/data-local/images/cifar/cifar10/by-image/test'
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
    ds_size = len(dataset.imgs)

    if args.labels:
        with open('../third_party/data-local/labels/cifar10/1000_balanced_labels/10.txt') as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

    elif args.labeled_batch_size:

        # domain adaptation dataset
        if args.target_domain is not None:

            target_dataset_config = datasets.__dict__[args.target_domain]()

            if args.target_domain == 'mnist':
                valid_sources = ['usps']

                target_traindir = '{0}/train'.format(target_dataset_config['datadir'])
                evaldir = '{0}/test'.format(target_dataset_config['datadir'])
                eval_transformation = target_dataset_config['eval_transformation']
            else:
                None

            target_dataset = torchvision.datasets.ImageFolder(target_traindir,
                                                              target_dataset_config['train_transformation'])
            target_labeled_idxs, target_unlabeled_idxs = data.relabel_dataset(target_dataset, {})

            dataset = ConcatDataset([dataset, target_dataset])
            unlabeled_idxs += [ds_size + i for i in range(0, len(target_dataset.imgs))]

        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader


def create_model(name, num_classes, ema=False):

    model_factory = architectures.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # decline lr
    lr *= ramps.zero_cosine_rampdown(epoch, args.epochs)

    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr


def generate_pseudo_labels(labels_pseudo, level_choose_num=2):
    num = len(labels_pseudo)
    onehot_labels_c = torch.zeros((num, 10))
    included_class = [i for i in range(10)]
    for i in range(num):
        current_class_index_level = included_class.copy()
        current_class_index_level.remove(labels_pseudo[i])
        level_choose_class = np.random.choice(current_class_index_level, size=level_choose_num, replace=True)
        onehot_labels_c[i][level_choose_class] = 1
    return onehot_labels_c.long().cuda()


def nl(outputs, complementary_mask):
    w = torch.sum(complementary_mask, dim=1)
    outputs_pro = torch.softmax(outputs, dim=1)
    outputs_pro = torch.tensor(1) - outputs_pro
    outputs_pro = -torch.log(outputs_pro + 1e-6)
    loss = (torch.sum((outputs_pro * complementary_mask), dim=1)/w).sum() / len(outputs)
    return loss


def train_epoch(train_loader, l_model, r_model, l_optimizer, r_optimizer, epoch):
    global global_step

    # define criterions
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    residual_logit_criterion = losses.symmetric_mse_loss
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss

    l_model.train()
    r_model.train()

    for i, ((l_input, r_input), target) in tqdm(enumerate(train_loader)):

        # adjust learning rate
        adjust_learning_rate(l_optimizer, epoch, i, len(train_loader))
        adjust_learning_rate(r_optimizer, epoch, i, len(train_loader))
        # prepare data
        l_input_var = Variable(l_input)
        r_input_var = Variable(r_input)
        le_input_var = Variable(r_input, requires_grad=False, volatile=True)
        re_input_var = Variable(l_input, requires_grad=False, volatile=True)

        target_var = Variable(target.cuda())
        un_index = target_var.data.le(NO_LABEL)
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size >= 0 and unlabeled_minibatch_size >= 0

        # forward
        l_model_out = l_model(l_input_var)
        r_model_out = r_model(r_input_var)
        le_model_out = l_model(le_input_var)
        re_model_out = r_model(re_input_var)

        if len(l_model_out) == 2:
            assert len(r_model_out) == 2
            l_logit1, l_logit2 = l_model_out
            r_logit1, r_logit2 = r_model_out
            le_logit1, le_logit2 = le_model_out
            re_logit1, re_logit2 = re_model_out

        # logit distance loss from mean teacher
        if args.logit_distance_cost >= 0:
            l_class_logit, l_cons_logit = l_logit1, l_logit2
            r_class_logit, r_cons_logit = r_logit1, r_logit2
            le_class_logit, le_cons_logit = le_logit1, le_logit2
            re_class_logit, re_cons_logit = re_logit1, re_logit2

            l_res_loss = args.logit_distance_cost * residual_logit_criterion(l_class_logit,
                                                                             l_cons_logit) / minibatch_size
            r_res_loss = args.logit_distance_cost * residual_logit_criterion(r_class_logit,
                                                                             r_cons_logit) / minibatch_size

        else:
            l_class_logit, l_cons_logit = l_logit1, l_logit1
            r_class_logit, r_cons_logit = r_logit1, r_logit1
            le_class_logit, le_cons_logit = le_logit1, le_logit2
            re_class_logit, re_cons_logit = re_logit1, re_logit2

            l_res_loss = 0.0
            r_res_loss = 0.0

        # classification loss
        l_class_loss = class_criterion(l_class_logit, target_var) / minibatch_size
        r_class_loss = class_criterion(r_class_logit, target_var) / minibatch_size

        l_loss, r_loss = l_class_loss, r_class_loss
        l_loss += l_res_loss
        r_loss += r_res_loss

        # consistency loss
        consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

        le_class_logit = Variable(le_class_logit.detach().data, requires_grad=False)
        l_consistency_loss = consistency_weight * consistency_criterion(l_cons_logit, le_class_logit) / minibatch_size

        l_loss += l_consistency_loss

        re_class_logit = Variable(re_class_logit.detach().data, requires_grad=False)
        r_consistency_loss = consistency_weight * consistency_criterion(r_cons_logit, re_class_logit) / minibatch_size

        r_loss += r_consistency_loss

        # stabilization loss
        # value (cls_v) and index (cls_i) of the max probability in the prediction
        l_cls_v, l_cls_i = torch.max(F.softmax(l_class_logit, dim=1), dim=1)
        r_cls_v, r_cls_i = torch.max(F.softmax(r_class_logit, dim=1), dim=1)
        le_cls_v, le_cls_i = torch.max(F.softmax(le_class_logit, dim=1), dim=1)
        re_cls_v, re_cls_i = torch.max(F.softmax(re_class_logit, dim=1), dim=1)

        l_cls_i = l_cls_i.data.cpu().numpy()
        r_cls_i = r_cls_i.data.cpu().numpy()
        le_cls_i = le_cls_i.data.cpu().numpy()
        re_cls_i = re_cls_i.data.cpu().numpy()

        # stable prediction mask
        l_mask = (l_cls_v > args.stable_threshold).data.cpu().numpy()
        r_mask = (r_cls_v > args.stable_threshold).data.cpu().numpy()
        le_mask = (le_cls_v > args.stable_threshold).data.cpu().numpy()
        re_mask = (re_cls_v > args.stable_threshold).data.cpu().numpy()

        # detach logit -> for generating stablilization target
        in_r_cons_logit = Variable(r_cons_logit.detach().data, requires_grad=False)
        tar_l_class_logit = Variable(l_class_logit.clone().detach().data, requires_grad=False)

        in_l_cons_logit = Variable(l_cons_logit.detach().data, requires_grad=False)
        tar_r_class_logit = Variable(r_class_logit.clone().detach().data, requires_grad=False)

        dnll_index_r = []
        dnll_index_l = []
        # generate target for each sample
        for sdx in range(0, unlabeled_minibatch_size):
            l_stable = False
            if l_mask[sdx] == 0 and le_mask[sdx] == 0:
                # unstable: do not satisfy the 2nd condition
                tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
                dnll_index_r.append(sdx)
            elif l_cls_i[sdx] != le_cls_i[sdx]:
                # unstable: do not satisfy the 1st condition
                tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
                dnll_index_r.append(sdx)
            else:
                l_stable = True

            r_stable = False
            if r_mask[sdx] == 0 and re_mask[sdx] == 0:
                # unstable: do not satisfy the 2nd condition
                tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
                dnll_index_l.append(sdx)
            elif r_cls_i[sdx] != re_cls_i[sdx]:
                # unstable: do not satisfy the 1st condition
                tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
                dnll_index_l.append(sdx)
            else:
                r_stable = True

            # calculate stability if both models are stable for a sample
            if l_stable and r_stable:
                # compare by consistency
                l_sample_cons = consistency_criterion(l_cons_logit[sdx:sdx + 1, ...], le_class_logit[sdx:sdx + 1, ...])
                r_sample_cons = consistency_criterion(r_cons_logit[sdx:sdx + 1, ...], re_class_logit[sdx:sdx + 1, ...])
                if l_sample_cons.data.cpu().numpy() < r_sample_cons.data.cpu().numpy():
                    # loss: l -> r
                    tar_r_class_logit[sdx, ...] = in_l_cons_logit[sdx, ...]
                    dnll_index_l.append(sdx)
                elif l_sample_cons.data.cpu().numpy() > r_sample_cons.data.cpu().numpy():
                    # loss: r -> l
                    tar_l_class_logit[sdx, ...] = in_r_cons_logit[sdx, ...]
                    dnll_index_r.append(sdx)

        # calculate stablization weight
        stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(epoch, args.stabilization_rampup)
        if not args.exclude_unlabeled:
            stabilization_weight = (unlabeled_minibatch_size / minibatch_size) * stabilization_weight

        # stabilization loss for r model
        if args.exclude_unlabeled:
            r_stabilization_loss = stabilization_weight * stabilization_criterion(r_cons_logit,
                                                                                  tar_l_class_logit) / minibatch_size
        else:
            # for idx in range(unlabeled_minibatch_size, minibatch_size):
            #     tar_l_class_logit[idx, ...] = in_r_cons_logit[idx, ...]
            r_stabilization_loss = stabilization_weight * stabilization_criterion(r_cons_logit[:unlabeled_minibatch_size],
                                                                                  tar_l_class_logit[:unlabeled_minibatch_size]) / unlabeled_minibatch_size

        r_loss += r_stabilization_loss

        # stabilization loss for l model
        if args.exclude_unlabeled:
            l_stabilization_loss = stabilization_weight * stabilization_criterion(l_cons_logit,
                                                                                  tar_r_class_logit) / minibatch_size
        else:
            # for idx in range(unlabeled_minibatch_size, minibatch_size):
            #     tar_r_class_logit[idx, ...] = in_l_cons_logit[idx, ...]
            l_stabilization_loss = stabilization_weight * stabilization_criterion(l_cons_logit[:unlabeled_minibatch_size],
                                                                                  tar_r_class_logit[:unlabeled_minibatch_size]) / unlabeled_minibatch_size

        l_loss += l_stabilization_loss


        # outputs_u_w_label_c_1_1 = generate_pseudo_labels(le_cls_i[dnll_index_l])
        outputs_u_w_label_c_2_1 = generate_pseudo_labels(re_cls_i[dnll_index_l])

        outputs_u_w_label_c_1_2 = generate_pseudo_labels(le_cls_i[dnll_index_r])
        # outputs_u_w_label_c_2_2 = generate_pseudo_labels(re_cls_i[dnll_index_r])

        if epoch > 15:
            nl_weight = 2
        else:
            nl_weight = args.nl_scale * ramps.sigmoid_rampup(epoch, args.nl_rampup)
        loss_u_l = (nl(l_cons_logit[dnll_index_l], outputs_u_w_label_c_2_1)) * nl_weight
        loss_u_r = (nl(r_cons_logit[dnll_index_r], outputs_u_w_label_c_1_2)) * nl_weight

        l_loss += loss_u_l
        r_loss += loss_u_r

        # update model
        l_optimizer.zero_grad()
        l_loss.backward()
        l_optimizer.step()

        r_optimizer.zero_grad()
        r_loss.backward()
        r_optimizer.step()

        global_step += 1


def validate(net, loader):
    # Evaluate
    net.eval()
    test_correct = 0
    test_all = 0
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.cuda(), target.cuda()
            output = net(image)
            test_all += target.shape[0]
            test_correct += (target == output[0].argmax(1)).sum().item()

    test_acc = test_correct / test_all * 100
    print('%d images tested.' % int(test_all))
    print('Test accuracy: %.4f' % test_acc)
    return test_acc


def main(context):
    global best_prec1
    global global_step

    # create loggers
    checkpoint_path = context.transient_dir

    # create dataloaders
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    # create models
    l_model = create_model(name='l', num_classes=num_classes)
    r_model = create_model(name='r', num_classes=num_classes)
    LOG.info(parameters_string(l_model))
    LOG.info(parameters_string(r_model))

    # create optimizers
    l_optimizer = torch.optim.SGD(params=l_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
    r_optimizer = torch.optim.SGD(params=r_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)

    # restore saved checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '=> no checkpoint found at: {}'.format(args.resume)
        LOG.info('=> loading checkpoint: {}'.format(args.resume))

        checkpoint = torch.load(args.resume)

        # globel parameters
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']

        # models and optimizers
        l_model.load_state_dict(checkpoint['l_model'])
        r_model.load_state_dict(checkpoint['r_model'])
        l_optimizer.load_state_dict(checkpoint['l_optimizer'])
        r_optimizer.load_state_dict(checkpoint['r_optimizer'])

        LOG.info('=> loaded checkpoint {} (epoch {})'.format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # training
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        train_epoch(train_loader, l_model, r_model, l_optimizer, r_optimizer, epoch)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time() - start_time))
        is_best = False
        if args.validation_epochs and (epoch + 1) % args.validation_epochs == 0:
            start_time = time.time()

            LOG.info('Validating the left model: ')
            l_prec1 = validate(l_model, eval_loader)
            LOG.info('Validating the right model: ')
            r_prec1 = validate(r_model, eval_loader)

            LOG.info('--- validation in {} seconds ---'.format(time.time() - start_time))
            better_prec1 = l_prec1 if l_prec1 > r_prec1 else r_prec1
            best_prec1 = max(better_prec1, best_prec1)
            is_best = better_prec1 > best_prec1

        # save checkpoint
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            mt_func.save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_prec1': best_prec1,
                'arch': args.arch,
                'l_model': l_model.state_dict(),
                'r_model': r_model.state_dict(),
                'l_optimizer': l_optimizer.state_dict(),
                'r_optimizer': r_optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)


LOG.info('Best top1 prediction: {0}'.format(best_prec1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parser_commandline_args()
    main(run_context.RunContext(__file__, 0))
