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

from src import ramps, losses, cli, run_context, datasets
from src import art as architectures
import collections
from tqdm import tqdm

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


class EMA(object):
    def __init__(self, base_net, net, decay=0.999):
        self.shadows = collections.OrderedDict()
        self.decay = decay
        self.net = base_net

        self.new_params = self.net.state_dict().copy()

        for name, param in net.named_parameters():
            self.shadows[name] = param.data.clone().detach()
            self.new_params[name].copy_(self.shadows[name])
        self.net.load_state_dict(self.new_params)

    def update(self, net):
        for name, param in net.named_parameters():
            # self.shadows[name] = (1.0 - self.decay) * self.shadows[name] + self.decay * param.data.clone().detach()
            # For lockless ops, although mostly not needed
            # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            self.shadows[name] -= self.decay * (self.shadows[name] - param.data.clone().detach())
            self.new_params[name].copy_(self.shadows[name])
        self.net.load_state_dict(self.new_params)

    def fill_in_bn(self, state_dict):
        for key in state_dict.keys():
            if ('running_mean' in key or 'running_var' in key) and key not in self.shadows.keys():
                self.shadows[key] = state_dict[key].clone()

    def state_dict(self):
        return self.shadows

    def predict(self, X):
        return self.net(X)


def create_data_loaders(train_transformation, eval_transformation, datadir, args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
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
            LOG.info('\nYou set target domain: {0} on script.\n'
                     'This is a domain adaptation experiment.\n'.format(args.target_domain))
            target_dataset_config = datasets.__dict__[args.target_domain]()

            if args.target_domain == 'mnist':
                valid_sources = ['usps']
                if not args.dataset in valid_sources:
                    LOG.error('\nYou set \'mnist\' as the target domain. \n'
                              'However, you use the source domain: \'{0}\'.\n'
                              'The source domain should be \'{1}\''.format(args.dataset, valid_sources))

                target_traindir = '{0}/train'.format(target_dataset_config['datadir'])
                evaldir = '{0}/test'.format(target_dataset_config['datadir'])
                eval_transformation = target_dataset_config['eval_transformation']
            else:
                LOG.error('Unsupport target domain: {0}.\n'.format(args.target_domain))

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


def create_model(num_classes, ema=False, name=None):
    LOG.info('=> creating {pretrained} {name} model: {arch}'.format(
        pretrained='pre-trained' if args.pretrained else 'non-pre-trained',
        name=name,
        arch=args.arch))

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


def consistency_loss(logits_w1, logits_w2):
    logits_w2 = logits_w2.detach()
    assert logits_w1.size() == logits_w2.size()
    return F.mse_loss(torch.softmax(logits_w1,dim=-1), torch.softmax(logits_w2,dim=-1), reduction='mean')


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


def train_epoch(train_loader, l_model, t_model, l_optimizer, epoch):
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
    for i, ((l_input, r_input), target) in tqdm(enumerate(train_loader)):

        # adjust learning rate
        adjust_learning_rate(l_optimizer, epoch, i, len(train_loader))

        # prepare data
        l_input_var = Variable(l_input.cuda())
        le_input_var = Variable(r_input.cuda(), requires_grad=False, volatile=True)
        target_var = Variable(target.cuda())

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size >= 0 and unlabeled_minibatch_size >= 0

        un_index = target_var.data.le(NO_LABEL)
        le_input_var = le_input_var[un_index]
        # forward
        l_model_out = l_model(l_input_var)

        t_model_out = t_model.predict(le_input_var)
        d_model_out = l_model(le_input_var)

        if isinstance(l_model_out, Variable):
            assert args.logit_distance_cost < 0
            l_logit1 = l_model_out

        elif len(l_model_out) == 2:
            assert len(t_model_out) == 2
            l_logit1, l_logit2 = l_model_out
            t_logit1, t_logit2 = t_model_out
            d_logit1, d_logit2 = d_model_out

        # logit distance loss from mean teacher
        if args.logit_distance_cost >= 0:
            l_class_logit, l_cons_logit = l_logit1, l_logit2
            le_class_logit, le_cons_logit = t_logit1, t_logit2
            re_class_logit, re_cons_logit = d_logit1, d_logit2
            l_res_loss = args.logit_distance_cost * residual_logit_criterion(l_class_logit, l_cons_logit) / minibatch_size
        else:
            l_class_logit, l_cons_logit = l_logit1, l_logit1
            l_res_loss = 0.0

        # classification loss
        l_class_loss = class_criterion(l_class_logit, target_var) / minibatch_size
        l_loss = l_class_loss
        l_loss += l_res_loss

        # consistency loss
        consistency_weight = args.consistency_scale * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

        le_class_logit = Variable(le_class_logit.detach().data, requires_grad=False)
        l_consistency_loss = consistency_weight * consistency_criterion(l_cons_logit[un_index], le_class_logit) / unlabeled_minibatch_size
        l_loss += l_consistency_loss

        if epoch > 15:
            nl_weight = 2
        else:
            nl_weight = args.nl_scale * ramps.sigmoid_rampup(epoch, args.nl_rampup)

        re_class_logit = Variable(re_class_logit.detach().data, requires_grad=False)
        re_cls_i = torch.max(F.softmax(re_class_logit, dim=1), dim=1)[1].data.cpu().numpy()
        outputs_u_w_label_c = generate_pseudo_labels(re_cls_i)
        loss_u = (nl(l_cons_logit[un_index], outputs_u_w_label_c)) * nl_weight

        l_loss += loss_u

        # update model
        l_optimizer.zero_grad()
        l_loss.backward()
        l_optimizer.step()
        t_model.update(l_model)

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
    l_model = create_model(num_classes=num_classes, name='student')
    t_model = EMA(base_net=create_model(num_classes=num_classes, ema=True), net=l_model)
    r_model = None
    LOG.info(parameters_string(l_model))

    # create optimizers
    l_optimizer = torch.optim.SGD(params=l_model.parameters(),
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)
    r_optimizer = None

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

    # # training
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        train_epoch(train_loader, l_model, t_model, l_optimizer, epoch)
        LOG.info('--- training epoch in {} seconds ---'.format(time.time() - start_time))

        is_best = False
        if args.validation_epochs and (epoch + 1) % args.validation_epochs == 0:
            start_time = time.time()
            LOG.info('Validating the left model: ')
            l_prec1 = validate(l_model, eval_loader)
            LOG.info('--- validation in {} seconds ---'.format(time.time() - start_time))
            better_prec1 = l_prec1
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
                'r_model': r_model,
                'l_optimizer': l_optimizer.state_dict(),
                'r_optimizer': r_optimizer.state_dict() if r_optimizer is not None else None,
            }, is_best, checkpoint_path, epoch + 1)

    LOG.info('Best top1 prediction: {0}'.format(best_prec1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parser_commandline_args()
    main(run_context.RunContext(__file__, 0))