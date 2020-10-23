from __future__ import absolute_import, print_function

import os
import sys
import torch
from torch.utils.data import DataLoader

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC,TrackerSiamVGG
from got.experiments import *

import argparse
parser = argparse.ArgumentParser(description='tracker')
parser.add_argument('--tracker', default='siamfc', type=str, required=True,help='tracker name (default: siamfc)')
args = parser.parse_args()

if __name__ == '__main__':
    # setup dataset
    name = 'GOT-10k' # use only GOT10k database to train
    assert name in ['VID', 'GOT-10k']
    if name == 'GOT-10k':
        root_dir = '../../data/GOT' #need change
        seq_dataset = GOT10k(root_dir, subset='train')
    elif name == 'VID':
        root_dir = 'data/ILSVRC' #need change
        seq_dataset = ImageNetVID(root_dir, subset=('train', 'val'))
    pair_dataset = Pairwise(seq_dataset)

    # setup data loader
    cuda = torch.cuda.is_available()
    loader = DataLoader(
        pair_dataset, batch_size=8, shuffle=True,
        pin_memory=cuda, drop_last=True, num_workers=8) # when run in win, num_workers=0

    if args["tracker"] == "siamfc":
        # setup tracker
        tracker = TrackerSiamFC()
        # path for saving checkpoints
        net_dir = 'pretrained/siamfc_new'
    elif args["tracker"] == "siamvgg":
        # setup tracker
        tracker = TrackerSiamVGG()
        # path for saving checkpoints
        net_dir = 'pretrained/siamvgg_new'

    if not os.path.exists(net_dir):
        os.makedirs(net_dir)

    # training loop
    epoch_num = 50
    for epoch in range(epoch_num):
        for step, batch in enumerate(loader):
            loss = tracker.step(
                batch, backward=True, update_lr=(step == 0))
            if step % 20 == 0:
                print('Epoch [{}][{}/{}]: Loss: {:.3f}'.format(
                    epoch + 1, step + 1, len(loader), loss))
                sys.stdout.flush()

        # save checkpoint
        net_path = os.path.join(net_dir, 'model_e%d.pth' % (epoch + 1))
        torch.save(tracker.net.state_dict(), net_path)
        
        # test on OTB2015 dataset
        if args.tracker == "siamfc":
            tracker_test = TrackerSiamFC(net_path=net_path)
        elif args.tracker == "siamvgg":
            tracker_test = TrackerSiamVGG(net_path=net_path)

        experiments = ExperimentOTB('../../data/OTB100', version=2015,
                                    result_dir='{}_dataset/results_{}'.format(name, epoch + 1),
                                    report_dir='{}_dataset/reports_{}'.format(name, epoch + 1))  #change OTB2015 database dir

        # run tracking experiments and report performance
        experiments.run(tracker_test, visualize=False)
        experiments.report([tracker_test.name])

