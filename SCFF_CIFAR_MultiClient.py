"""
SCFF CIFAR-10 Multi-Client Split Learning
==========================================
1 Server (Layers 1-2) + N Clients (each holds Layer 0 copy + data shard).

Sequential/Relay approach:
  - CIFAR-10 is split equally among N clients
  - Each round, clients take turns:
      Client_i trains Layer 0 on its shard → sends activations to server
  - Between clients, Layer 0 weights are relayed (passed to next client)
  - Server layers are shared and train on activations from ALL clients
  - After cycling through all N clients = 1 round ≈ 1 epoch over full dataset
  - Multiple rounds → accuracy within ±1-2% of baseline

Usage:
  python SCFF_CIFAR_MultiClient.py --num_clients 10
  python SCFF_CIFAR_MultiClient.py --num_clients 5
  python SCFF_CIFAR_MultiClient.py --num_clients 2
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR

import torch.nn.functional as F
import torch.nn.init as init
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision
from torchvision.transforms import transforms, ToPILImage
from torch.utils.data import DataLoader, Subset
import argparse
import copy
import math
import json
import numpy as np

# ============================================================
# Reproducibility
# ============================================================
torch.manual_seed(1234)

# ============================================================
# Transforms
# ============================================================
s = 0.5
transform1 = transforms.Compose([
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
# Dataset classes
# ============================================================
class DualAugmentCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, augment="No", *args, **kwargs):
        super().__init__(root, *args, **kwargs)
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
            img2 = transforms.Compose([
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])(img_pil)
            return img_original, img1, img2, target
        else:
            return img_original, target


class DualAugmentCIFAR10_test(torchvision.datasets.CIFAR10):
    def __init__(self, aug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug = aug

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = ToPILImage()(img)
        img = transform_train(img) if self.aug else transform_test(img)
        return img, target


# ============================================================
# Data splitting for multi-client
# ============================================================
def get_client_loaders(num_clients, batchsize, iid=True):
    """
    Split CIFAR-10 training set into `num_clients` equal shards.

    Args:
        num_clients: number of clients
        batchsize: batch size per client
        iid: if True, random IID split; if False, sort by class (non-IID)

    Returns:
        client_trainloaders: list of DataLoaders, one per client
        sup_trainloader: full supervised train loader (for evaluation)
        testloader: test loader
    """
    torch.manual_seed(1234)

    trainset = DualAugmentCIFAR10(root='./data', train=True, download=True, augment="no")
    sup_trainset = DualAugmentCIFAR10_test(root='./data', aug=True, train=True, download=True)
    testset = DualAugmentCIFAR10_test(root='./data', aug=False, train=False, download=True)

    n_total = len(trainset)  # 50,000
    shard_size = n_total // num_clients

    if iid:
        # Random IID split
        indices = torch.randperm(n_total).tolist()
    else:
        # Non-IID: sort by label (harder scenario)
        targets = torch.tensor(trainset.targets)
        indices = targets.argsort().tolist()

    client_trainloaders = []
    for c in range(num_clients):
        start = c * shard_size
        end = start + shard_size if c < num_clients - 1 else n_total
        client_indices = indices[start:end]
        client_subset = Subset(trainset, client_indices)
        loader = DataLoader(client_subset, batch_size=batchsize, shuffle=True, num_workers=2)
        client_trainloaders.append(loader)

    print(f"Data split: {n_total} samples → {num_clients} clients × {shard_size} samples each")

    sup_trainloader = DataLoader(
        Subset(sup_trainset, list(range(n_total))),
        batch_size=64, shuffle=True
    )
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

    return client_trainloaders, sup_trainloader, testloader


# ============================================================
# Utility functions (same as baseline)
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
    x = x - torch.mean(x, dim=dims, keepdim=True)
    x = x / (1e-10 + torch.std(x, dim=dims, keepdim=True))
    return x


class standardnorm(nn.Module):
    def __init__(self, dims=[1, 2, 3]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x - torch.mean(x, dim=self.dims, keepdim=True)
        x = x / (1e-10 + torch.std(x, dim=self.dims, keepdim=True))
        return x


class L2norm(nn.Module):
    def __init__(self, dims=[1, 2, 3]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x / (x.norm(p=2, dim=self.dims, keepdim=True) + 1e-10)


class triangle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x - torch.mean(x, axis=1, keepdims=True)
        return F.relu(x)


# ============================================================
# Conv2d layer (same as baseline)
# ============================================================
class Conv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, pad=0,
                 batchnorm=False, normdims=[1, 2, 3], norm="stdnorm", bias=True,
                 dropout=0.0, padding_mode="reflect", concat=True, act="relu"):
        super().__init__()
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
        self.act = torch.nn.ReLU() if act == 'relu' else triangle()
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
# Helpers
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


def create_layer(layer_config, opt_config, device, act):
    net = Conv2d(layer_config["ch_in"], layer_config["channels"],
                 (layer_config["kernel_size"], layer_config["kernel_size"]),
                 pad=layer_config["pad"], norm="stdnorm",
                 padding_mode=layer_config["padding_mode"], act=act)
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
    return net, pool, extra_pool


# ============================================================
# SplitClient: Holds Layer 0 weights (relayed between clients)
# ============================================================
class SplitClient:
    """
    Represents the client-side model (Layer 0).
    In multi-client mode, there's ONE set of Layer 0 weights
    that gets relayed from client to client.
    """

    def __init__(self, net, pool, extra_pool, device, lr, weight_decay, gamma):
        self.net = net
        self.pool = pool
        self.extra_pool = extra_pool
        self.device = device
        # Store optimizer hyperparams for re-creation after weight relay
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        # Create optimizer/scheduler
        self.optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma)

    def get_weights(self):
        """Get Layer 0 weights + optimizer state for relay."""
        return {
            'model': copy.deepcopy(self.net.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
        }

    def set_weights(self, state):
        """Receive relayed weights from previous client."""
        self.net.load_state_dict(state['model'])
        # Recreate optimizer to bind to current net params, then load state
        self.optimizer = AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer.load_state_dict(state['optimizer'])
        # Update scheduler to track new optimizer
        self.scheduler = ExponentialLR(self.optimizer, self.gamma)

    def train_step(self, x, threshold1, threshold2, a, b, lamda, p=1):
        """Train Layer 0 on a batch. Returns pooled activations for server."""
        self.net.train()
        x = stdnorm(x, dims=[1, 2, 3])
        x_pos, x_neg = get_pos_neg_batch_imgcats(x, x, p=p)

        out_pos = self.net(x_pos)
        out_neg = self.net(x_neg)

        goodness_pos = self.net.relu(out_pos).pow(2).mean([1])
        goodness_neg = self.net.relu(out_neg).pow(2).mean([1])

        self.optimizer.zero_grad()
        loss = (torch.log(1 + torch.exp(a * (-goodness_pos + threshold1))).mean([1, 2]).mean()
                + torch.log(1 + torch.exp(b * (goodness_neg - threshold2))).mean([1, 2]).mean()
                + lamda * torch.norm(goodness_pos, p=2, dim=(1, 2)).mean())
        loss.backward()
        self.optimizer.step()

        x_pos_pooled = self.pool(self.net.act(out_pos)).detach()
        x_neg_pooled = self.pool(self.net.act(out_neg)).detach()

        gp = torch.mean(goodness_pos.mean([1, 2])).item()
        gn = torch.mean(goodness_neg.mean([1, 2])).item()

        return x_pos_pooled, x_neg_pooled, gp, gn

    def extract_features(self, x, dims_in, dims_out, stdnorm_out, Layer_out):
        """Inference: extract Layer 0 features."""
        self.net.eval()
        outputs = []
        with torch.no_grad():
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
# SplitServer: Holds Layers 1-2 (shared across all clients)
# ============================================================
class SplitServer:
    """Server holds Layers 1..NL-1, shared by all clients."""

    def __init__(self, nets, pools, extra_pools, optimizers, schedulers, device):
        self.nets = nets
        self.pools = pools
        self.extra_pools = extra_pools
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device

    def train_step(self, x, x_neg, threshold1_list, threshold2_list, a, b, lamda_list,
                   alleps, epoch, p=1):
        """Train server layers on received activations."""
        gp, gn = 0, 0
        for i, net in enumerate(self.nets):
            net.train()
            if net.concat:
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

            x = self.pools[i](net.act(out_pos)).detach()
            x_neg = self.pools[i](net.act(out_neg)).detach()

            gp = torch.mean(goodness_pos.mean([1, 2])).item()
            gn = torch.mean(goodness_neg.mean([1, 2])).item()

        return gp, gn

    def extract_features(self, x, dims_in, dims_out, stdnorm_out, Layer_out):
        """Inference: extract features from server layers."""
        outputs = []
        for i, net in enumerate(self.nets):
            net.eval()
            with torch.no_grad():
                global_layer_idx = i + 1
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
# MultiClientTrainer: Orchestrates N clients + 1 server
# ============================================================
class MultiClientTrainer:
    """
    Sequential/Relay Split Learning with N clients.

    Each "round":
      for client_id in 0..N-1:
        1. Client receives relayed Layer 0 weights
        2. Client trains Layer 0 on its local data shard
        3. Client sends activations to server
        4. Server trains Layers 1-2 on those activations
        5. Client passes Layer 0 weights to next client

    After N clients = 1 round ≈ 1 full epoch over 50k samples.
    """

    def __init__(self, client, server, client_loaders, eval_loaders, eval_config,
                 threshold1, threshold2, lamda, period, alleps,
                 p=1, a=1, b=1, num_clients=10):
        self.client = client
        self.server = server
        self.client_loaders = client_loaders  # list of N DataLoaders
        self.eval_loaders = eval_loaders      # (sup_trainloader, testloader)
        self.eval_config = eval_config
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.lamda = lamda
        self.period = period
        self.alleps = alleps
        self.p = p
        self.a = a
        self.b = b
        self.num_clients = num_clients
        self.Dims = []

    def train(self, total_rounds):
        """
        Run multi-client training.

        Args:
            total_rounds: number of full rounds (each round = all N clients train once)
                          Equivalent to epochs in the single-client case.
        """
        device = self.eval_config.device
        firstpass = True
        nbbatches = 0

        for round_num in range(total_rounds):
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{total_rounds} "
                  f"(equivalent epoch over full dataset)")
            print(f"{'='*60}")

            round_gp_accum = 0
            round_gn_accum = 0
            round_batch_count = 0

            # --- Sequential relay through all clients ---
            for client_id in range(self.num_clients):
                client_loader = self.client_loaders[client_id]

                self.client.net.train()
                for net in self.server.nets:
                    net.train()

                client_gp = 0
                client_gn = 0

                for numbatch, (x, _) in enumerate(client_loader):
                    nbbatches += 1
                    x = x.to(device)

                    # --- Client: Layer 0 ---
                    x_pos_pooled, x_neg_pooled, gp_c, gn_c = self.client.train_step(
                        x, self.threshold1[0], self.threshold2[0],
                        self.a, self.b, self.lamda[0], p=self.p
                    )

                    if firstpass:
                        _, ch, h, w = x_pos_pooled.shape
                        self.Dims.append(ch * h * w)
                        print(f"Layer 0 : shape {x_pos_pooled.shape}")

                    # Client LR scheduler
                    if round_num < self.alleps[0] and (nbbatches + 1) % self.period[0] == 0:
                        self.client.scheduler.step()

                    # --- Server: Layers 1+ ---
                    server_th1 = self.threshold1[1:]
                    server_th2 = self.threshold2[1:]
                    server_lamda = self.lamda[1:]
                    server_alleps = self.alleps[1:]

                    gp_s, gn_s = self.server.train_step(
                        x_pos_pooled, x_neg_pooled,
                        server_th1, server_th2,
                        self.a, self.b, server_lamda,
                        server_alleps, round_num, p=self.p
                    )

                    if firstpass:
                        dummy_x = x_pos_pooled
                        for si, snet in enumerate(self.server.nets):
                            with torch.no_grad():
                                if snet.concat:
                                    dummy_x = stdnorm(dummy_x, dims=[1, 2, 3])
                                    dummy_x = torch.cat((dummy_x, dummy_x), dim=1)
                                dummy_x = self.server.pools[si](snet.act(snet(dummy_x)))
                                _, ch, h, w = dummy_x.shape
                                self.Dims.append(ch * h * w)
                                print(f"Layer {si + 1} : shape {dummy_x.shape}")
                        firstpass = False

                    # Server LR schedulers
                    for si in range(len(self.server.nets)):
                        if round_num < self.alleps[si + 1] and (nbbatches + 1) % self.period[si + 1] == 0:
                            self.server.schedulers[si].step()

                    client_gp += gp_s
                    client_gn += gn_s
                    round_batch_count += 1

                n_batches_client = len(client_loader)
                print(f"  Client {client_id}: {n_batches_client} batches, "
                      f"pos_goodness={client_gp/max(n_batches_client,1):.4f}, "
                      f"neg_goodness={client_gn/max(n_batches_client,1):.4f}")

                round_gp_accum += client_gp
                round_gn_accum += client_gn

            # End of round summary
            print(f"Round {round_num} done: "
                  f"avg_pos={round_gp_accum/max(round_batch_count,1):.4f}, "
                  f"avg_neg={round_gn_accum/max(round_batch_count,1):.4f}")

        print("\nTraining complete.")
        return self.Dims

    def evaluate(self):
        """Evaluate by training a linear readout on features from all layers."""
        config = self.eval_config
        sup_trainloader, testloader = self.eval_loaders

        current_rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        all_nets = [self.client.net] + self.server.nets
        all_extra_pools = [self.client.extra_pool] + self.server.extra_pools

        lengths = calculate_output_length(self.Dims, all_nets, all_extra_pools,
                                          config.Layer_out, config.all_neurons)
        print(f"Classifier input length: {lengths}")
        classifier = build_classifier(lengths, config)

        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        lr_scheduler = CustomStepLR(optimizer, nb_epochs=50)
        criterion = nn.CrossEntropyLoss()

        self.client.net.eval()
        for net in self.server.nets:
            net.eval()

        train_accs = []
        test_accs = []
        last_test_acc = 0

        for epoch_eval in range(50):
            classifier.train()
            correct = 0
            total = 0
            for x, labels in sup_trainloader:
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
                print(f'Readout epoch {epoch_eval}: train acc = {100 * train_acc:.2f}%')
                test_acc = self._test_readout(classifier, testloader, criterion, config, epoch_eval)
                last_test_acc = test_acc

            train_accs.append(train_acc)
            test_accs.append(last_test_acc)

        final_train = train_acc
        final_test = self._test_readout(classifier, testloader, criterion, config, 49)

        torch.set_rng_state(current_rng_state)
        self._plot_accuracy(train_accs, test_accs)

        return final_train, final_test

    def _test_readout(self, classifier, loader, criterion, config, epoch):
        classifier.eval()
        correct = 0
        total = 0
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

        acc = correct / total
        print(f'  Test accuracy: {100 * acc:.2f}%')
        classifier.train()
        return acc

    def _plot_accuracy(self, train_accs, test_accs):
        plt.figure(figsize=(10, 6))
        epochs = list(range(1, len(train_accs) + 1))
        plt.plot(epochs, [a * 100 for a in train_accs], label='Train Accuracy', marker='.')
        plt.plot(epochs, [a * 100 for a in test_accs], label='Test Accuracy', marker='.')
        plt.xlabel('Readout Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'SCFF Multi-Client Split Learning ({self.num_clients} clients) - Readout Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        outpath = f'/home/claw/.openclaw/workspace/scff_split/multiclient_{self.num_clients}_accuracy.png'
        plt.savefig(outpath, dpi=150)
        plt.close()
        print(f"Plot saved to {outpath}")


# ============================================================
# Main
# ============================================================
def get_arguments():
    parser = argparse.ArgumentParser(description="SCFF Multi-Client Split Learning", add_help=False)
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients (data split equally)")
    parser.add_argument("--iid", action="store_true", default=True, help="IID data split (default: True)")
    parser.add_argument("--non_iid", action="store_true", help="Non-IID data split (sorted by class)")
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
    parser = argparse.ArgumentParser('SCFF Multi-Client', parents=[get_arguments()])
    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")

    NL = args.NL
    assert NL >= 2

    device = 'cuda:' + str(args.device_num) if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    torch.manual_seed(args.seed_num)

    with open('config.json', 'r') as f:
        config_json = json.load(f)

    layer_configs = config_json['CIFAR']['layer_configs'][:NL]
    opt_configs = config_json['CIFAR']['opt_configs'][:NL]
    concats = args.concats
    act_list = args.act

    iid = not args.non_iid
    num_clients = args.num_clients

    # --- Load data (split among clients) ---
    client_loaders, sup_trainloader, testloader = get_client_loaders(
        num_clients=num_clients, batchsize=100, iid=iid
    )

    # --- Build Client (Layer 0) - single set of weights relayed ---
    lc0 = layer_configs[0]
    net0, pool0, extra_pool0 = create_layer(lc0, opt_configs[0], device, act_list[0])
    net0.concat = bool(concats[0])

    client = SplitClient(
        net0, pool0, extra_pool0, device,
        lr=args.lr[0], weight_decay=args.weight_decay[0], gamma=args.gamma[0]
    )

    # --- Build Server (Layers 1..NL-1) ---
    server_nets = []
    server_pools = []
    server_extra_pools = []
    server_optimizers = []
    server_schedulers = []

    for i in range(1, NL):
        lc = layer_configs[i]
        net_i, pool_i, extra_pool_i = create_layer(lc, opt_configs[i], device, act_list[i])
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

    # --- Evaluation config ---
    eval_config = EvaluationConfig(
        device=device, dims=(1, 2, 3), dims_in=(1, 2, 3), dims_out=(1, 2, 3),
        stdnorm_out=True, out_dropout=0.2, Layer_out=[2, 1, 0],
        pre_std=True, all_neurons=False
    )

    # --- Build trainer ---
    trainer = MultiClientTrainer(
        client=client,
        server=server,
        client_loaders=client_loaders,
        eval_loaders=(sup_trainloader, testloader),
        eval_config=eval_config,
        threshold1=args.th1,
        threshold2=args.th2,
        lamda=args.lamda,
        period=args.period,
        alleps=args.alleps,
        p=1, a=1, b=1,
        num_clients=num_clients
    )

    # --- Train ---
    total_rounds = max(args.alleps)  # each round = full pass through all clients
    Dims = trainer.train(total_rounds)
    print(f"Layer output dims: {Dims}")

    # --- Evaluate ---
    acc_train, acc_test = trainer.evaluate()
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({num_clients} clients, {'IID' if iid else 'Non-IID'} split)")
    print(f"{'='*60}")
    print(f"  Train accuracy: {100 * acc_train:.2f}%")
    print(f"  Test accuracy:  {100 * acc_test:.2f}%")

    # --- Save ---
    if args.save_model:
        os.makedirs('./results', exist_ok=True)
        torch.save(client.net.state_dict(), './results/params_CIFAR_multiclient_l0.pth')
        for i, net in enumerate(server.nets):
            torch.save(net.state_dict(), f'./results/params_CIFAR_multiclient_l{i + 1}.pth')
        print("Models saved.")
