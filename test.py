from __future__ import absolute_import

from got.experiments import *
'''
OR
delete "got" file
$ pip install got10k
from got10k.experiments import *
-----------
got modified :
1、delete some code to help continuously run train.py
2、modified show_frame() to show_image(), to enable visualize
'''
from siamfc import TrackerSiamFC,TrackerSiamVGG

import argparse
parser = argparse.ArgumentParser(description='tracker')
parser.add_argument('--tracker', default='siamfc', type=str, help='tracker name (default: siamfc)')
args = parser.parse_args()

# to enable multiprocess, delete this , speed drop to 1/4
import multiprocessing
multiprocessing.set_start_method('spawn',True)

if __name__ == '__main__':
    # setup tracker
    if args.tracker=="siamfc":
        net_path = 'model_siamfc.pth'  #model dir
        tracker = TrackerSiamFC(net_path=net_path)
    elif args.tracker=="siamvgg":
        net_path = 'model_siamvgg.pth' #siamvgg model dir
        tracker = TrackerSiamVGG(net_path=net_path)

    # setup experiments
    # got-10k toolkit
    # modified according to database you use
    experiments = [
        #ExperimentGOT10k('data/GOT-10k', subset='test'),
        #ExperimentOTB('data/OTB', version=2013),
        ExperimentOTB('F:/database/OTB2015/OTB100', version=2015),
        #ExperimentVOT('../../data3/VOT/vot2016', version=2016),
        #ExperimentVOT('../../data/VOT2018', version=2018),
        #ExperimentDTB70('data/DTB70'),
        #ExperimentTColor128('data/Temple-color-128'),
        #ExperimentUAV123('data/UAV123', version='UAV123'),
        #ExperimentUAV123('data/UAV123', version='UAV20L'),
        #ExperimentNfS('data/nfs', fps=30),
        #ExperimentNfS('data/nfs', fps=240)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=True) #if run in server , change to False
        e.report([tracker.name])
