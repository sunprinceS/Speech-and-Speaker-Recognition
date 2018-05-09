import argparse
import os
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.utils import np_utils
from model import build_model
from states import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='train.py',
            description='Phoneme Recognition Training')
    parser.add_argument('--feat_type',type=str,default='dlmfcc',choices=['dlmfcc','lmfcc','mspec','dmspec'])
    parser.add_argument('--num_layer',type=int,default=4)
    parser.add_argument('--num_hid',type=int,default=256)
    parser.add_argument('--act_func',type=str,default='relu',choices=['relu','sigmoid'])
    parser.add_argument('--optimizer',type=str,default='adam',choices=['adam','sgd'])
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=20)


    args = parser.parse_args()
    log_path = 'log/{feat_type}_{num_layer}layer_{num_hid}hidden_{act_func}_{optimizer}_epoch{epoch}'.format(**vars(args))
    print("[LOG] Set path to {}".format(log_path))

    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    nb_class = len(stateList)
    phoneme_clr = build_model(nb_class = nb_class,**vars(args))
    tbCallBack = TensorBoard(log_dir = log_path,histogram_freq = 0,write_graph=True,write_images=True)
    modelCkptCallBack = ModelCheckpoint(os.path.join(log_path,'model.h5'),monitor='val_loss',save_best_only=True)

    training_data = np.load('data/{}_train_x.npz'.format(args.feat_type))

    tr_x = (training_data['train_x']).astype('float32')
    val_x = (training_data['val_x']).astype('float32')

    tr_y = np_utils.to_categorical(np.load('data/train_y.npz')['y'],nb_class)
    val_y = np_utils.to_categorical(np.load('data/val_y.npz')['y'],nb_class)

    phoneme_clr.fit(x=tr_x,y=tr_y,batch_size = args.batch_size,epochs = args.epoch, callbacks = [tbCallBack,modelCkptCallBack], validation_data=(val_x,val_y))
