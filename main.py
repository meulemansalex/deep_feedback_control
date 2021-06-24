#!/usr/bin/env python3
# Copyright 2021 Alexander Meulemans, Matilde Tristany Farinha, Javier Garcia Gordonez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.utils import FastMNIST, FastFashionMNIST
from utils.train import train, train_bp
from utils.args import parse_cmd_arguments
from utils import builders, utils
from tensorboardX import SummaryWriter
import os.path
import pickle


def run():

    """
    - Parsing command-line arguments
    - Creating synthetic regression data
    - Initiating training process
    - Testing final network
    """
    args = parse_cmd_arguments()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Using cuda: ' + str(use_cuda))

    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    if args.dataset in ['mnist', 'mnist_autoencoder']:
        if torchvision.__version__ != '0.9.1':
            datasets.MNIST.resources = [
                (
                'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
                'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
                (
                'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
                'd53e105ee54ea40749a09fcbcd1e9432'),
                (
                'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
                '9fb629c4189551a2d022fa330f9573f3'),
                (
                'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
                'ec29112dd5afa0611ce80d1b7f02629c')
            ]

        if args.dataset == 'mnist':
            print('### Training on MNIST ###')
        elif args.dataset == 'mnist_autoencoder':
            print('### Training on MNIST Autoencoder ###')
        if args.multiple_hpsearch:
            data_dir = '../../../../../data'
        elif args.hpsearch:
            data_dir = '../../../../../data'
        else:
            data_dir = './data'

        if args.no_preprocessing_mnist:
            print('This option is deprecated.')

        train_dataset = FastMNIST(data_dir, device, args.double_precision, train=True, download=True)

        if args.no_val_set:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0)
            val_loader = None
        else:
            if torch.__version__ == '1.4.0':  # quickfix to avoid error
                trainset, valset = torch.utils.data.random_split(train_dataset,
                                                                 [55000, 5000])
            else:
                if (torch.__version__ == '1.8.1' or torch.__version__ == '1.8.1+cu102')\
                        and device.type == 'cuda':
                    g_cuda = torch.Generator(device='cuda:0')
                else:
                    g_cuda = torch.Generator(device=device)

                trainset, valset = torch.utils.data.random_split(train_dataset,
                                                                 [55000, 5000],
                                                                 g_cuda)
            train_loader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True, num_workers=0)
            val_loader = torch.utils.data.DataLoader(valset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False, num_workers=0)
        test_dataset = FastMNIST(data_dir, device, args.double_precision, train=False, download=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0)

    elif args.dataset == 'fashion_mnist':
        print('### Training on Fashion-MNIST ###')
        if args.multiple_hpsearch:
            data_dir = '../../../../../data'
        elif args.hpsearch:
            data_dir = '../../../../../data'
        else:
            data_dir = './data'

        train_dataset = FastFashionMNIST(data_dir, device, args.double_precision,
                                  train=True, download=True)

        if args.no_val_set:
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0)
            val_loader = None
        else:
            if torch.__version__ == '1.4.0':  # quickfix to avoid error
                trainset, valset = torch.utils.data.random_split(train_dataset,
                                                                 [55000, 5000])
            else:
                if (torch.__version__ == '1.8.1' or torch.__version__ == '1.8.1+cu102') \
                        and device.type == 'cuda':
                    g_cuda = torch.Generator(device='cuda:0')
                else:
                    g_cuda = torch.Generator(device=device)

                trainset, valset = torch.utils.data.random_split(train_dataset,
                                                                 [55000, 5000],
                                                                 g_cuda)
            train_loader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True, num_workers=0)
            val_loader = torch.utils.data.DataLoader(valset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False, num_workers=0)
        test_dataset = FastFashionMNIST(data_dir, device, args.double_precision,
                                        train=False, download=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0)

    elif args.dataset == 'cifar10':
        print('### Training on CIFAR10 ###')
        if args.multiple_hpsearch:
            data_dir = '../../../../../data'
        elif args.hpsearch:
            data_dir = '../../../../../data'
        else:
            data_dir = './data'

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset_total = torchvision.datasets.CIFAR10(root=data_dir,
                                                      train=True,
                                                    download=True,
                                                    transform=transform)
        if args.no_val_set:
            train_loader = torch.utils.data.DataLoader(trainset_total,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0)
            val_loader = None
        else:
            if (torch.__version__ == '1.8.1' or torch.__version__ == '1.8.1+cu102') \
                    and device.type == 'cuda':
                g_cuda = torch.Generator(device='cuda:0')
            else:
                g_cuda = torch.Generator(device=device)
            trainset, valset = torch.utils.data.random_split(trainset_total,
                                                             [45000, 5000],
                                                             g_cuda)
            train_loader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=args.batch_size,
                                                       shuffle=True, num_workers=0)
            val_loader = torch.utils.data.DataLoader(valset,
                                                     batch_size=args.batch_size,
                                                     shuffle=False, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=True,
                                               transform=transform)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0)

    elif args.dataset == 'student_teacher':
        print('### Training a student_teacher regression model ###')
        torch.set_default_tensor_type('torch.FloatTensor')
        if args.double_precision:
            torch.set_default_dtype(torch.float64)

        if not args.load_ST_dataset:
            train_x, test_x, val_x, train_y, test_y, val_y = \
                builders.generate_data_from_teacher(
                    n_in=args.size_input, n_out=args.size_output,
                    n_hidden=[1000, 1000, 1000, 1000], device=device,
                    num_train=args.num_train, num_test=args.num_test,
                    num_val=args.num_val,
                    args=args, activation='relu')
        else:
            train_x = np.load('./data/train_x.npy')
            test_x = np.load('./data/test_x.npy')
            val_x = np.load('./data/val_x.npy')
            train_y = np.load('./data/train_y.npy')
            test_y = np.load('./data/test_y.npy')
            val_y = np.load('./data/val_y.npy')

        if use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if args.double_precision:
            torch.set_default_dtype(torch.float64)

        train_loader = DataLoader(utils.RegressionDataset(train_x, train_y,
                                                          args.double_precision),
                                  batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(utils.RegressionDataset(test_x, test_y,
                                                         args.double_precision),
                                 batch_size=args.batch_size, shuffle=False)

        if args.no_val_set:
            val_loader = None
        else:
            val_loader = DataLoader(utils.RegressionDataset(val_x, val_y,
                                                            args.double_precision),
                                    batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))

    if args.log_interval is None:
        args.log_interval = max(1, int(len(train_loader)/100))

    if args.save_logs:
        writer = SummaryWriter(logdir=args.out_dir)
    else:
        writer = None

    summary = utils.setup_summary_dict(args)

    net = builders.build_network(args)
    net.to(device)

    if (args.network_type == "DFC" and not args.ndi) or (args.network_type == "TPDI" and not net._fast_di and not net._ndi):
        args.save_convergence = True
        print('Saving how many samples converge/diverge/neither.')
    else:
        args.save_convergence = False
        print('Not saving convergence results.')

    if not args.network_type in ('BP', 'BPConv'):
        summary = train(args=args,
                        device=device,
                        train_loader=train_loader,
                        net=net,
                        writer=writer,
                        test_loader=test_loader,
                        summary=summary,
                        val_loader=val_loader)
    else:
        summary = train_bp(args=args, device=device, train_loader=train_loader, net=net, writer=writer,
                           test_loader=test_loader, summary=summary, val_loader=val_loader)

    if (args.save_df and args.network_type != 'BP'):
        summary['bp_angles'] = net.bp_angles
        summary['gnt_angles'] = net.gnt_angles
        summary['gn_angles'] = net.gn_angles
        summary['gn_angles_network'] = net.gn_angles_network
        summary['gnt_angles_network'] = net.gnt_angles_network
        summary['nullspace_relative_norm_angles'] = net.nullspace_relative_norm
        if args.network_type == 'DFC':
            summary['ndi_angles'] = net.ndi_angles
            summary['ndi_angles_network'] = net.ndi_angles_network
            summary['condition_gn_angles'] = net.condition_gn
            summary['condition_gn_angles_init'] = net.condition_gn_init
            summary['jac_transpose_angles'] = net.jac_transpose_angles
            summary['jac_transpose_angles_init'] = net.jac_transpose_angles_init
            summary['jac_pinv_angles'] = net.jac_pinv_angles
            summary['jac_pinv_angles_init'] = net.jac_pinv_angles_init

    if summary['finished'] == 0:
        summary['finished'] = 1
        utils.save_summary_dict(args, summary)
    if writer is not None:
        writer.close()

    filename = os.path.join(args.out_dir, 'results.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(summary, f)

    return summary

if __name__ == '__main__':
    run()


