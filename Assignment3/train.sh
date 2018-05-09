#!/usr/bin/env sh
pip install keras
python train.py --feat_type dlmfcc --num_layer 4 --num_hid 256 --act_func relu --optimizer adam
python train.py --feat_type lmfcc --num_layer 4 --num_hid 256 --act_func relu --optimizer adam
python train.py --feat_type dmspec --num_layer 4 --num_hid 256 --act_func relu --optimizer adam
python train.py --feat_type mspec --num_layer 4 --num_hid 256 --act_func relu --optimizer adam
python train.py --feat_type dlmfcc --num_layer 1 --num_hid 1024 --act_func relu --optimizer adam
python train.py --feat_type dlmfcc --num_layer 4 --num_hid 256 --act_func relu --optimizer sgd
python train.py --feat_type dlmfcc --num_layer 4 --num_hid 256 --act_func sigmoid --optimizer adam
