# -*- coding: utf-8 -*-
import sys
sys.path.append('./Evaluation')
from eval_detection_gentime import ANETdetection
import matplotlib.pyplot as plt
import numpy as np

def run_evaluation_detection(ground_truth_filename, prediction_filename, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation', verbose=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=False)
    anet_detection.evaluate()
    
    ap = anet_detection.ap
    mAP = anet_detection.mAP
    tdiff = anet_detection.tdiff
    
    return (mAP, ap, tdiff)

def evaluation_detection(opt, verbose=True):
    
    mAP, AP, tdiff = run_evaluation_detection(
        opt["video_anno"],
        opt["result_file"],
        tiou_thresholds=np.linspace(0.3, 0.70, 5),
        subset=opt['inference_subset'], verbose=verbose)
    
    if verbose:    
        print('mAP')
        print(mAP)
        #print('AP')
        #print(AP)
        print('AEDT')
        print(tdiff)
    
    return mAP

