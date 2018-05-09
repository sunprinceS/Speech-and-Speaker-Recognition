import argparse
import os
import numpy as np
from keras.models import load_model
from keras.utils import plot_model
from metric import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='predict.py',
            description='Phoneme Recognition Evaluating')
    parser.add_argument('--feat_type',type=str,default='dlmfcc',choices=['dlmfcc','lmfcc','mspec','dmspec'])
    parser.add_argument('--num_layer',type=int,default=4)
    parser.add_argument('--num_hid',type=int,default=256)
    parser.add_argument('--act_func',type=str,default='relu',choices=['relu','sigmoid'])
    parser.add_argument('--optimizer',type=str,default='adam',choices=['adam','sgd'])
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=20)


    args = parser.parse_args()
    log_path = 'log/{feat_type}_{num_layer}layer_{num_hid}hidden_{act_func}_{optimizer}_epoch{epoch}'.format(**vars(args))
    print("[LOG] Load model from {}".format(log_path))
    
    # phoneme_clr = load_model(os.path.join(log_path,'model.h5'))
    # plot_model(phoneme_clr,to_file=os.path.join(log_path,'model.png'))
    # testing_data = np.load('data/{}_test_x.npz'.format(args.feat_type))
    # pred = phoneme_clr.predict_classes(testing_data['test_x'],batch_size=args.batch_size)
    
    pred = np.load('pred_y.npz')['pred']
    ans = np.load('data/test_y.npz')['y']
    print("[LOG] answer loaded")
    
    print("\nF2F Accuracy on state level: {:3f}".format(frameByFrame('state',pred,ans)))
    print("\nF2F Accuracy on phoneme level: {:3f}".format(frameByFrame('phoneme',pred,ans)))
    print("\nPER on state level: {:3f}".format(phone_error_rate('state',pred,ans)))
    print("\nPER on phoneme level: {:3f}".format(phone_error_rate('phoneme',pred,ans)))
