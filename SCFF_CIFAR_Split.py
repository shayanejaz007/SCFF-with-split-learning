"""
SCFF CIFAR-10 Split Learning Implementation
============================================
Split point: After Layer 0 (client). Server runs Layers 1-2.
Mathematically identical to baseline due to greedy layer-wise training (.detach()).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP duplicate runtime on Windows

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR, LinearLR

import torch.nn.functional as F
import torch.nn.init as init
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import transforms, ToPILImage
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split, Subset
import argparse
import time

import numpy as np
from numpy import fft
import math
import json

# ============================================================
# Reproducibility
# ============================================================
torch.manual_seed(1234)

# ============================================================
# Transforms (unchanged from baseline)
# ============================================================
s = 0.5
transform1 = transforms.Compose([
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform2 = transforms.Compose([
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ============================================================
# Dataset classes (unchanged from baseline)
# ============================================================
class DualAugmentCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, augment="No", *args, **kwargs):
        super(DualAugmentCIFAR10, self).__init__(root, *args, **kwargs)
        self.augment = augment

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img_pil = ToPILImage()(img)
        img_original = transform_train(img_pil)

        if self.augment == "single":
            img1 = transform1(img_pil)
            return img_original, img1, img_original, target
        elif self.augment == "dual":
            img1 = transform1(img_pil)
            img2 = transform2(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, target


class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    def __init__(self, aug=False, *args, **kwargs):
        super(DualAugmentCIFAR10_test, self).__init__(*args, **kwargs)
        self.aug = aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = ToPILImage()(img)
        if self.aug:
            img = transform_train(img)
        else:
            img = transform_test(img)
        return img, target


def get_train(batchsize, augment, Factor):
    torch.manual_seed(1234)
    trainset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment=augment)
    sup_trainset = DualAugmentCIFAR10_test(root='./data', aug=True, train=True, download=True)
    factor = Factor
    train_len = int(len(trainset) * factor)
    indices = torch.randperm(len(trainset)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]
    train_data = Subset(trainset, train_indices)
    sup_train_data = Subset(sup_trainset, train_indices)
    val_data = Subset(sup_trainset, val_indices)
    testset = DualAugmentCIFAR10_test(root='./data', aug=False, train=False, download=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=2)
    if factor == 1:
        valloader = testloader
    else:
        valloader = DataLoader(val_data, batch_size=1000, shuffle=True, num_workers=2)
    sup_trainloader = DataLoader(sup_train_data, batch_size=64, shuffle=True)
    return trainloader, valloader, testloader, sup_trainloader

# ============================================================
# Utility functions (unchanged from baseline)
# ============================================================
def get_pos_neg_batch_imgcats(batch_pos1, batch_pos2, p=1):
    batch_size = len(batch_pos1)
    batch_pos = torch.cat((batch_pos1, batch_pos2), dim=1)
    random_indices = (torch.randperm(batch_size - 1) + 1)[:min(p, batch_size - 1)]
    labeles = torch.arange(batch_size)
    batch_negs = []
    for i in random_indices:
        batch_neg = batch_pos2[(labeles + i) % batch_size]
        batch_neg = torch.cat((batch_pos1, batch_neg), dim=1)
        batch_negs.append(batch_neg)
    return batch_pos, torch.cat(batch_negs)


def stdnorm(x, dims=[1, 2, 3]):
    x = x - torch.mean(x, dim=(dims), keepdim=True)
    x = x / (1e-10 + torch.std(x, dim=(dims), keepdim=True))
    return x


class standardnorm(nn.Module):
    def __init__(self, dims=[1, 2, 3]):
        super(standardnorm, self).__init__()
        self.dims = dims

    def forward(self, x):
        x = x - torch.mean(x, dim=(self.dims), keepdim=True)
        x = x / (1e-10 + torch.std(x, dim=(self.dims), keepdim=True))
        return x


class L2norm(nn.Module):
    def __init__(self, dims=[1, 2, 3]):
        super(L2norm, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x / (x.norm(p=2, dim=(self.dims), keepdim=True) + 1e-10)


class triangle(nn.Module):
    def __init__(self):
        super(triangle, self).__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)


# ============================================================
# Conv2d layer (unchanged from baseline)
# ============================================================
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, pad=0,
                 batchnorm=False, normdims=[1, 2, 3], norm="stdnorm", bias=True,
                 dropout=0.0, padding_mode="reflect", concat=True, act="relu"):
        super(Conv2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.normdims = normdims
        self.concat = concat
        self.relu = torch.nn.ReLU()
        self.conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                     kernel_size=kernel_size, bias=bias)
        init.xavier_uniform_(self.conv_layer.weight)
        self.padding_mode = padding_mode
        self.F_padding = (pad, pad, pad, pad)
        if act == 'relu':
            self.act = torch.nn.ReLU()
        else:
            self.act = triangle()
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(self.input_channels, affine=False)
        else:
            self.bn1 = nn.Identity()
        if norm == "L2norm":
            self.norm = L2norm(dims=normdims)
        elif norm == "stdnorm":
            self.norm = standardnorm(dims=normdims)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.bn1(x)
        x = F.pad(x, self.F_padding, self.padding_mode)
        x = self.norm(x)
        if self.concat:
            lenchannel = x.size(1) // 2
            out = self.conv_layer(x[:, :lenchannel]) + self.conv_layer(x[:, lenchannel:])
        else:
            out = self.conv_layer(x)
        return out


# ============================================================
# Scheduler and evaluation helpers (unchanged from baseline)
# ============================================================
class CustomStepLR(StepLR):
    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5 for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class EvaluationConfig:
    def __init__(self, device, dims, dims_in, dims_out, stdnorm_out, out_dropout,
                 Layer_out, pre_std, all_neurons):
        self.device = device
        self.dims = dims
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.stdnorm_out = stdnorm_out
        self.out_dropout = out_dropout
        self.Layer_out = Layer_out
        self.all_neurons = all_neurons
        self.pre_std = pre_std


def calculate_output_length(dims, nets, extra_pool, Layer, all_neurons):
    lengths = 0
    if all_neurons:
        for i, length in enumerate(dims):
            if i in Layer:
                lengths += length
    else:
        for i, length in enumerate(dims):
            if i in Layer:
                len_after_pool = math.ceil(
                    (math.sqrt(length / nets[i].output_channels) - extra_pool[i].kernel_size)
                    / extra_pool[i].stride + 1)
                lengths += len_after_pool * len_after_pool * nets[i].output_channels
    return lengths


def build_classifier(lengths, config):
    classifier = nn.Sequential(
        nn.Dropout(config.out_dropout),
        nn.Linear(lengths, 10)
    ).to(config.device)
    if torch.cuda.device_count() > 2:
        classifier = nn.DataParallel(classifier)
    return classifier


def create_layer(layer_config, opt_config, load_params, device, act):
    layer_num = layer_config['num'] - 1
    net = Conv2d(layer_config["ch_in"], layer_config["channels"],
                 (layer_config["kernel_size"], layer_config["kernel_size"]),
                 pad=layer_config["pad"], norm="stdnorm",
                 padding_mode=layer_config["padding_mode"], act=act)
    if load_params:
        net.load_state_dict(torch.load('./results/params_CIFAR_l' + str(layer_num) + '.pth', map_location='cpu'))
        for param in net.parameters():
            param.requires_grad = False
    if layer_config["pooltype"] == 'Avg':
        pool = nn.AvgPool2d(kernel_size=layer_config["pool_size"],
                            stride=layer_config["stride_size"],
                            padding=layer_config["padding"], ceil_mode=True)
    else:
        pool = nn.MaxPool2d(kernel_size=layer_config["pool_size"],
                            stride=layer_config["stride_size"],
                            padding=layer_config["padding"], ceil_mode=True)
    extra_pool = nn.AvgPool2d(kernel_size=layer_config["extra_pool_size"],
                               stride=layer_config["extra_pool_size"],
                               padding=0, ceil_mode=True)
    net.to(device)
    optimizer = AdamW(net.parameters(), lr=opt_config["lr"], weight_decay=opt_config["weight_decay"])
    scheduler = ExponentialLR(optimizer, opt_config["gamma"])
    return net, pool, extra_pool, optimizer, scheduler


# ============================================================
# SplitClient: Holds Layer 0
# ============================================================
class SplitClient:
    """Client holds Layer 0 and trains it on raw images."""

    def __init__(self, net, pool, extra_pool, optimizer, scheduler, device):
        self.net = net
        self.pool = pool
        self.extra_pool = extra_pool
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_step(self, x, threshold1, threshold2, a, b, lamda, p=1):
        """
        Train Layer 0 on a batch.

        Args:
            x: raw images [B, C, H, W]
            threshold1: positive threshold for layer 0
            threshold2: negative threshold for layer 0
            a, b: loss scaling
            lamda: regularization coefficient for layer 0
            p: number of negative samples

        Returns:
            x_pos_pooled: detached pooled activated pos output [B, 96, H', W']
            x_neg_pooled: detached pooled activated neg output [B*p, 96, H', W']
            goodness_pos_mean: scalar for logging
            goodness_neg_mean: scalar for logging
        """
        self.net.train()

        # Layer 0 has concat=True, so stdnorm + pos/neg creation
        x = stdnorm(x, dims=[1, 2, 3])
        x_pos, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

        out_pos = self.net(x_pos)
        out_neg = self.net(x_neg)

        goodness_pos = self.net.relu(out_pos).pow(2).mean([1])
        goodness_neg = self.net.relu(out_neg).pow(2).mean([1])

        # SCFF softplus loss (identical to baseline)
        self.optimizer.zero_grad()
        loss = (torch.log(1 + torch.exp(a * (-goodness_pos + threshold1))).mean([1, 2]).mean()
                + torch.log(1 + torch.exp(b * (goodness_neg - threshold2))).mean([1, 2]).mean()
                + lamda * torch.norm(goodness_pos, p=2, dim=(1, 2)).mean())
        loss.backward()
        self.optimizer.step()

        # Detach and pool for sending to server
        x_pos_pooled = self.pool(self.net.act(out_pos)).detach()
        x_neg_pooled = self.pool(self.net.act(out_neg)).detach()

        gp = torch.mean(goodness_pos.mean([1, 2])).item()
        gn = torch.mean(goodness_neg.mean([1, 2])).item()

        return x_pos_pooled, x_neg_pooled, gp, gn

    def extract_features(self, x, dims_in, dims_out, stdnorm_out, Layer_out):
        """
        Inference: extract Layer 0 features for evaluation.

        Returns:
            x_after_pool: activation after pool (to send to server)
            outputs: list of flattened features if layer 0 is in Layer_out
        """
        self.net.eval()
        outputs = []

        with torch.no_grad():
            # Layer 0 has concat=True
            x = stdnorm(x, dims=dims_in)
            x = torch.cat((x, x), dim=1)
            x = self.pool(self.net.act(self.net(x)))

            out = self.extra_pool(x)
            if stdnorm_out:
                out = stdnorm(out, dims=dims_out)
            out = out.flatten(start_dim=1)
            if 0 in Layer_out:
                outputs.append(out)

        return x, outputs


# ============================================================
# SplitServer: Holds Layers 1-2
# ============================================================
class SplitServer:
    """Server holds Layers 1..NL-1 and trains them on activations from client."""

    def __init__(self, nets, pools, extra_pools, optimizers, schedulers, device):
        """
        Args:
            nets: list of Conv2d for layers 1..NL-1
            pools, extra_pools, optimizers, schedulers: corresponding lists
        """
        self.nets = nets
        self.pools = pools
        self.extra_pools = extra_pools
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device

    def train_step(self, x, x_neg, threshold1_list, threshold2_list, a, b, lamda_list,
                   alleps, epoch, p=1):
        """
        Train server layers on received activations.

        Args:
            x: pos activations from client [B, C, H, W]
            x_neg: neg activations from client [B*p, C, H, W]
            threshold1_list: thresholds for server layers (indexed 0=layer1, 1=layer2, ...)
            threshold2_list: same for neg
            alleps: list of max epochs per server layer
            epoch: current epoch
            p: number of negative samples

        Returns:
            goodness_pos_mean, goodness_neg_mean: from last layer, for logging
        """
        gp, gn = 0, 0

        for i, net in enumerate(self.nets):
            net.train()

            if net.concat:
                # concat=True layers: stdnorm then create new pos/neg pairs
                x = stdnorm(x, dims=[1, 2, 3])
                x, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

            out_pos = net(x)
            out_neg = net(x_neg)

            goodness_pos = net.relu(out_pos).pow(2).mean([1])
            goodness_neg = net.relu(out_neg).pow(2).mean([1])

            if epoch < alleps[i]:
                self.optimizers[i].zero_grad()
                loss = (torch.log(1 + torch.exp(a * (-goodness_pos + threshold1_list[i]))).mean([1, 2]).mean()
                        + torch.log(1 + torch.exp(b * (goodness_neg - threshold2_list[i]))).mean([1, 2]).mean()
                        + lamda_list[i] * torch.norm(goodness_pos, p=2, dim=(1, 2)).mean())
                loss.backward()
                self.optimizers[i].step()

            # Detach and pool before passing to next layer
            x = self.pools[i](net.act(out_pos)).detach()
            x_neg = self.pools[i](net.act(out_neg)).detach()

            gp = torch.mean(goodness_pos.mean([1, 2])).item()
            gn = torch.mean(goodness_neg.mean([1, 2])).item()

        return gp, gn

    def extract_features(self, x, dims_in, dims_out, stdnorm_out, Layer_out):
        """
        Inference: extract features from server layers for evaluation.

        Args:
            x: activation from client's Layer 0 output
            Layer_out: which global layer indices to include (server layers are indices 1, 2, ...)

        Returns:
            outputs: list of flattened features for layers in Layer_out
        """
        outputs = []
        for i, net in enumerate(self.nets):
            net.eval()
            with torch.no_grad():
                global_layer_idx = i + 1  # server layer 0 = global layer 1

                if net.concat:
                    x = stdnorm(x, dims=dims_in)
                    x = torch.cat((x, x), dim=1)

                x = self.pools[i](net.act(net(x)))

                out = self.extra_pools[i](x)
                if stdnorm_out:
                    out = stdnorm(out, dims=dims_out)
                out = out.flatten(start_dim=1)
                if global_layer_idx in Layer_out:
                    outputs.append(out)

        return outputs


# ============================================================
# SplitTrainer: Orchestrates training and evaluation
# ============================================================
class SplitTrainer:
    """Orchestrates split learning between client and server."""

    def __init__(self, client, server, loaders, eval_config, threshold1, threshold2,
                 lamda, period, alleps, p=1, a=1, b=1):
        self.client = client
        self.server = server
        self.loaders = loaders
        self.eval_config = eval_config

        # threshold1/2 and lamda: index 0 = client (layer 0), indices 1+ = server layers
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.lamda = lamda
        self.period = period
        self.alleps = alleps
        self.p = p
        self.a = a
        self.b = b

        self.all_pos = [[] for _ in range(len(threshold1))]
        self.all_neg = [[] for _ in range(len(threshold1))]
        self.Dims = []

    def train(self, epochs):
        """Run the full SCFF training loop."""
        trainloader = self.loaders[0]
        device = self.eval_config.device
        nbbatches = 0
        firstpass = True

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self.client.net.train()
            for net in self.server.nets:
                net.train()

            goodness_pos_accum = 0
            goodness_neg_accum = 0

            for numbatch, (x, _) in enumerate(trainloader):
                nbbatches += 1
                x = x.to(device)

                # --- Client: Layer 0 ---
                x_pos_pooled, x_neg_pooled, gp_client, gn_client = self.client.train_step(
                    x, self.threshold1[0], self.threshold2[0],
                    self.a, self.b, self.lamda[0], p=self.p
                )

                if firstpass:
                    _, ch, h, w = x_pos_pooled.shape
                    self.Dims.append(ch * h * w)
                    print(f"Layer 0 : x.shape: {x_pos_pooled.shape} y.shape (after MaxP): {x_pos_pooled.shape}", end=" ")

                # Client scheduler
                if epoch < self.alleps[0] and (nbbatches + 1) % self.period[0] == 0:
                    self.client.scheduler.step()
                    print(f'nbbatches {nbbatches + 1} learning rate: {self.client.scheduler.get_last_lr()[0]}')

                # --- Server: Layers 1+ ---
                # Server thresholds/lamda are indices 1+ in the global lists
                server_th1 = self.threshold1[1:]
                server_th2 = self.threshold2[1:]
                server_lamda = self.lamda[1:]
                server_alleps = self.alleps[1:]

                gp_server, gn_server = self.server.train_step(
                    x_pos_pooled, x_neg_pooled,
                    server_th1, server_th2,
                    self.a, self.b, server_lamda,
                    server_alleps, epoch, p=self.p
                )

                if firstpass:
                    # Compute dims for server layers by running a dummy forward
                    dummy_x = x_pos_pooled
                    for si, snet in enumerate(self.server.nets):
                        with torch.no_grad():
                            if snet.concat:
                                dummy_x = stdnorm(dummy_x, dims=[1, 2, 3])
                                dummy_x = torch.cat((dummy_x, dummy_x), dim=1)
                            dummy_x = self.server.pools[si](snet.act(snet(dummy_x)))
                            _, ch, h, w = dummy_x.shape
                            self.Dims.append(ch * h * w)
                            print(f"Layer {si + 1} : x.shape: {dummy_x.shape} y.shape (after MaxP): {dummy_x.shape}", end=" ")
                    print()
                    firstpass = False

                # Server schedulers
                for si in range(len(self.server.nets)):
                    if epoch < self.alleps[si + 1] and (nbbatches + 1) % self.period[si + 1] == 0:
                        self.server.schedulers[si].step()
                        print(f'nbbatches {nbbatches + 1} learning rate: {self.server.schedulers[si].get_last_lr()[0]}')

                goodness_pos_accum += gp_server
                goodness_neg_accum += gn_server

                if numbatch == len(trainloader) - 1:
                    print(goodness_pos_accum / len(trainloader), goodness_neg_accum / len(trainloader))
                    # Log for last layer (global index NL-1)
                    nl = len(self.threshold1)
                    self.all_pos[nl - 1].append(goodness_pos_accum)
                    self.all_neg[nl - 1].append(goodness_neg_accum)
                    goodness_pos_accum, goodness_neg_accum = 0, 0

        print("Training done..")
        return self.Dims

    def evaluate(self):
        """
        Evaluate by training a linear readout on features from all layers.

        Returns:
            acc_train, acc_val: final training and validation accuracy
        """
        config = self.eval_config
        _, valloader, testloader, suptrloader = self.loaders

        current_rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        # Build full nets/pools/extra_pools lists for calculate_output_length
        all_nets = [self.client.net] + self.server.nets
        all_extra_pools = [self.client.extra_pool] + self.server.extra_pools

        lengths = calculate_output_length(self.Dims, all_nets, all_extra_pools,
                                          config.Layer_out, config.all_neurons)
        print(f"Classifier input length: {lengths}")
        classifier = build_classifier(lengths, config)

        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        lr_scheduler = CustomStepLR(optimizer, nb_epochs=50)
        criterion = nn.CrossEntropyLoss()

        # Use test set for validation (search=False in baseline default)
        eval_valloader = testloader

        # Set all nets to eval
        self.client.net.eval()
        for net in self.server.nets:
            net.eval()

        train_accs = []
        test_accs = []

        for epoch_eval in range(50):
            # --- Train readout ---
            classifier.train()
            correct = 0
            total = 0
            for x, labels in suptrloader:
                x = x.to(config.device)
                labels = labels.to(config.device)

                with torch.no_grad():
                    x_after_client, client_outputs = self.client.extract_features(
                        x, config.dims_in, config.dims_out, config.stdnorm_out, config.Layer_out)
                    server_outputs = self.server.extract_features(
                        x_after_client, config.dims_in, config.dims_out, config.stdnorm_out, config.Layer_out)

                outputs = client_outputs + server_outputs
                outputs = torch.cat(outputs, dim=1)

                optimizer.zero_grad()
                preds = classifier(outputs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            lr_scheduler.step()

            if epoch_eval % 20 == 0 or epoch_eval == 49:
                print(f'Accuracy of the network on the 50000 train images: {100 * train_acc} %')

                # Compute clean train accuracy
                train_acc_clean = self._test_readout(classifier, suptrloader, criterion, config, epoch_eval, 'Train')
                test_acc = self._test_readout(classifier, eval_valloader, criterion, config, epoch_eval, 'Val')

            # Record every epoch for plotting
            train_accs.append(train_acc)
            # For test, only update on eval epochs, else repeat last
            if epoch_eval % 20 == 0 or epoch_eval == 49:
                last_test_acc = test_acc
            test_accs.append(last_test_acc if 'last_test_acc' in dir() else 0)

        # Final evaluation
        final_train = self._test_readout(classifier, suptrloader, criterion, config, 49, 'Train')
        final_test = self._test_readout(classifier, eval_valloader, criterion, config, 49, 'Val')

        torch.set_rng_state(current_rng_state)

        # Plot
        self._plot_accuracy(train_accs, test_accs)

        return final_train, final_test

    def _test_readout(self, classifier, loader, criterion, config, epoch, mode):
        """Test readout classifier on a loader."""
        classifier.eval()
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for x, labels in loader:
                x = x.to(config.device)
                labels = labels.to(config.device)

                x_after_client, client_outputs = self.client.extract_features(
                    x, config.dims_in, config.dims_out, config.stdnorm_out, config.Layer_out)
                server_outputs = self.server.extract_features(
                    x_after_client, config.dims_in, config.dims_out, config.stdnorm_out, config.Layer_out)

                outputs = client_outputs + server_outputs
                outputs = torch.cat(outputs, dim=1)
                preds = classifier(outputs)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(preds, labels)
                running_loss += loss.item()

        acc = correct / total
        if mode == 'Val':
            print(f'Accuracy of the network on the 10000 {mode} images: {100 * acc} %')
            print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

        classifier.train()
        return acc

    def _plot_accuracy(self, train_accs, test_accs):
        """Save epoch-vs-accuracy plot."""
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(train_accs) + 1))
        plt.plot(epochs, [a * 100 for a in train_accs], label='Train Accuracy', marker='.')
        plt.plot(epochs, [a * 100 for a in test_accs], label='Test Accuracy', marker='.')
        plt.xlabel('Readout Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('SCFF Split Learning - Linear Readout Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('/home/claw/.openclaw/workspace/scff_split_accuracy.png', dpi=150)
        plt.close()
        print("Plot saved to scff_split_accuracy.png")


# ============================================================
# Pre-training validation
# ============================================================
def run_validation_checks(client, server, device, trainloader, config_json):
    """Run pre-training validation checks."""
    print("=" * 60)
    print("Running pre-training validation checks...")
    print("=" * 60)

    # 1. Config integrity
    required_keys = ['CIFAR']
    for key in required_keys:
        assert key in config_json, f"Missing config key: {key}"
    assert 'layer_configs' in config_json['CIFAR'], "Missing 'layer_configs' in config"
    assert 'opt_configs' in config_json['CIFAR'], "Missing 'opt_configs' in config"
    print("[OK] Config integrity check passed")

    # 2. Input tensor dimensions
    x_sample, _ = next(iter(trainloader))
    x_sample = x_sample.to(device)
    assert x_sample.shape[1:] == (3, 32, 32), \
        f"Expected input shape (B, 3, 32, 32), got {x_sample.shape}"
    print(f"[OK] Input tensor shape: {x_sample.shape}")

    # 3. Channel consistency at split boundary
    client_out_channels = client.net.output_channels
    server_in_channels = server.nets[0].input_channels
    assert client_out_channels == server_in_channels, \
        f"Channel mismatch at split: client outputs {client_out_channels}, server expects {server_in_channels}"
    print(f"[OK] Channel consistency: client out={client_out_channels}, server in={server_in_channels}")

    # 4. Split activation shapes
    with torch.no_grad():
        client.net.eval()
        x_test = stdnorm(x_sample[:4], dims=[1, 2, 3])
        x_pos_test, x_neg_test = get_pos_neg_batch_imgcats(x_test, x_test, p=1)
        out_test = client.net(x_pos_test)
        x_pooled_test = client.pool(client.net.act(out_test))
        print(f"[OK] Client output shape after pool: {x_pooled_test.shape}")
        assert x_pooled_test.shape[1] == client_out_channels, \
            f"Pooled output channels {x_pooled_test.shape[1]} != expected {client_out_channels}"

    # 5. Client/server forward compatibility (dummy batch)
    with torch.no_grad():
        # Run through server layers
        dummy = x_pooled_test
        for i, net in enumerate(server.nets):
            net.eval()
            if net.concat:
                dummy = stdnorm(dummy, dims=[1, 2, 3])
                dummy = torch.cat((dummy, dummy), dim=1)
            dummy = server.pools[i](net.act(net(dummy)))
        print(f"[OK] Server final output shape: {dummy.shape}")

    # 6. Label alignment
    for x_batch, labels_batch in trainloader:
        assert x_batch.shape[0] == labels_batch.shape[0], \
            f"Batch size mismatch: images {x_batch.shape[0]}, labels {labels_batch.shape[0]}"
        break
    print("[OK] Label alignment check passed")

    # 7. NL check for split
    NL = 1 + len(server.nets)
    if NL < 2:
        raise ValueError(f"Need at least 2 layers for split learning, got NL={NL}")
    print(f"[OK] NL={NL} is valid for split learning")

    print("=" * 60)
    print("All validation checks passed!")
    print("=" * 60)


# ============================================================
# Main entry point
# ============================================================
def get_arguments():
    parser = argparse.ArgumentParser(description="SCFF Split Learning Training Script", add_help=False)
    parser.add_argument("--lr", nargs='+', type=float, default=[0.02, 0.001, 0.0004])
    parser.add_argument("--gamma", nargs='+', type=float, default=[0.99, 0.9, 0.99])
    parser.add_argument("--period", nargs='+', type=int, default=[500, 500, 500])
    parser.add_argument("--weight_decay", nargs='+', type=float, default=[0.0001, 0.0003, 0.0001])
    parser.add_argument("--lamda", nargs='+', type=float, default=[0.0008, 0.0004, 0.0016])
    parser.add_argument("--th1", nargs='+', type=int, default=[1, 4, 5])
    parser.add_argument("--th2", nargs='+', type=int, default=[2, 5, 7])
    parser.add_argument("--NL", type=int, default=3)
    parser.add_argument("--concats", type=tuple, default=(1, 0, 1))
    parser.add_argument("--act", nargs='+', type=str, default=["triangle", "triangle", "relu"])
    parser.add_argument("--alleps", nargs='+', type=int, default=[6, 6, 13])
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--seed_num", type=int, default=1234)
    parser.add_argument("--save_model", action="store_true")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SCFF Split Learning', parents=[get_arguments()])
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    NL = args.NL
    assert NL >= 2, f"Split learning requires NL >= 2, got {NL}"

    device = 'cuda:' + str(args.device_num) if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    torch.manual_seed(args.seed_num)

    # Load config
    with open('config.json', 'r') as f:
        config_json = json.load(f)

    layer_configs = config_json['CIFAR']['layer_configs'][:NL]
    opt_configs = config_json['CIFAR']['opt_configs'][:NL]
    concats = args.concats
    act_list = args.act

    # Load data
    loaders = get_train(batchsize=100, augment="no", Factor=1)

    # ---- Build Client (Layer 0) ----
    lc0 = layer_configs[0]
    oc0 = opt_configs[0]
    net0, pool0, extra_pool0, _, _ = create_layer(lc0, oc0, load_params=False, device=device, act=act_list[0])
    net0.concat = bool(concats[0])
    opt0 = AdamW(net0.parameters(), lr=args.lr[0], weight_decay=args.weight_decay[0])
    sched0 = ExponentialLR(opt0, args.gamma[0])

    client = SplitClient(net0, pool0, extra_pool0, opt0, sched0, device)

    # ---- Build Server (Layers 1..NL-1) ----
    server_nets = []
    server_pools = []
    server_extra_pools = []
    server_optimizers = []
    server_schedulers = []

    for i in range(1, NL):
        lc = layer_configs[i]
        oc = opt_configs[i]
        net_i, pool_i, extra_pool_i, _, _ = create_layer(lc, oc, load_params=False, device=device, act=act_list[i])
        net_i.concat = bool(concats[i])
        opt_i = AdamW(net_i.parameters(), lr=args.lr[i], weight_decay=args.weight_decay[i])
        sched_i = ExponentialLR(opt_i, args.gamma[i])

        server_nets.append(net_i)
        server_pools.append(pool_i)
        server_extra_pools.append(extra_pool_i)
        server_optimizers.append(opt_i)
        server_schedulers.append(sched_i)

    server = SplitServer(server_nets, server_pools, server_extra_pools,
                         server_optimizers, server_schedulers, device)

    # ---- Evaluation config ----
    eval_config = EvaluationConfig(
        device=device,
        dims=(1, 2, 3),
        dims_in=(1, 2, 3),
        dims_out=(1, 2, 3),
        stdnorm_out=True,
        out_dropout=0.2,
        Layer_out=[2, 1, 0],
        pre_std=True,
        all_neurons=False
    )

    # ---- Pre-training validation ----
    run_validation_checks(client, server, device, loaders[0], config_json)

    # ---- Build trainer ----
    trainer = SplitTrainer(
        client=client,
        server=server,
        loaders=loaders,
        eval_config=eval_config,
        threshold1=args.th1,
        threshold2=args.th2,
        lamda=args.lamda,
        period=args.period,
        alleps=args.alleps,
        p=1,
        a=1,
        b=1
    )

    # ---- Train ----
    total_epochs = max(args.alleps)
    Dims = trainer.train(total_epochs)
    print(f"Layer output dims: {Dims}")

    # ---- Evaluate ----
    acc_train, acc_test = trainer.evaluate()
    print(f"\nFinal Results:")
    print(f"  Train accuracy: {100 * acc_train:.2f}%")
    print(f"  Test accuracy:  {100 * acc_test:.2f}%")

    # ---- Save model if requested ----
    if args.save_model:
        import os
        os.makedirs('./results', exist_ok=True)
        torch.save(client.net.state_dict(), './results/params_CIFAR_split_client_l0.pth')
        for i, net in enumerate(server.nets):
            torch.save(net.state_dict(), f'./results/params_CIFAR_split_server_l{i + 1}.pth')
        print("Models saved.")
