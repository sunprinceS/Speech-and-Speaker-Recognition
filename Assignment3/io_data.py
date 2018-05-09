import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

def prepare_train_data(feat_type = 'lmfcc',dfeat_winlen = None):
    print("[INFO] Start preparing training data")
    data = np.load('data/train_data.npz')['data']
    print("[INFO] Training data loaded")
    val_data = np.array(list(data[:385]) + list(data[4235:4620]))
    tr_data = np.array(list(data[385:4235]) + list(data[4620:]))

    # val_data = data[:3]
    # tr_data = data[385:388]

    val_feat = val_data[0][feat_type]
    tr_feat = tr_data[0][feat_type]
    val_label = val_data[0]['targets']
    tr_label = tr_data[0]['targets']

    for i,d in enumerate(val_data[1:]):
        val_feat = np.vstack((val_feat,d[feat_type]))
        val_label.extend(d['targets'])
    for i,d in enumerate(tr_data[1:]):
        tr_feat = np.vstack((tr_feat,d[feat_type]))
        tr_label.extend(d['targets'])
    
    ## Dynamic Feature
    num_dim = val_feat.shape[1]
    num_tr_data = tr_feat.shape[0]
    num_val_data = val_feat.shape[0]
    if dfeat_winlen is not None:
        new_val_feat = np.zeros((num_val_data,num_dim*(dfeat_winlen*2+1)))
        new_tr_feat = np.zeros((num_tr_data,num_dim*(dfeat_winlen*2+1)))

        for i in range(dfeat_winlen):
            new_val_feat[i][(dfeat_winlen-i)*num_dim:] = np.hstack(val_feat[0:i+4,:])
            new_val_feat[i][:(dfeat_winlen-i)*num_dim] = np.hstack(val_feat[3-i:0:-1,:])
            new_tr_feat[i][(dfeat_winlen-i)*num_dim:] = np.hstack(tr_feat[0:i+4,:])
            new_tr_feat[i][:(dfeat_winlen-i)*num_dim] = np.hstack(tr_feat[3-i:0:-1,:])
            

        for i in range(dfeat_winlen,num_val_data - dfeat_winlen):
            new_val_feat[i] = np.hstack(val_feat[i-3:i+4,:])

        for i in range(dfeat_winlen,num_tr_data - dfeat_winlen):
            new_tr_feat[i] = np.hstack(tr_feat[i-3:i+4,:])
        
        for sh,i in enumerate(range(num_val_data - dfeat_winlen,num_val_data)):
            new_val_feat[i][:(dfeat_winlen*2-sh)*num_dim] = np.hstack(val_feat[i-3:,:])
            new_val_feat[i][(dfeat_winlen*2-sh)*num_dim:] = np.hstack(val_feat[-2:-3-sh:-1,:])
        for sh,i in enumerate(range(num_tr_data - dfeat_winlen,num_tr_data)):
            new_tr_feat[i][:(dfeat_winlen*2-sh)*num_dim] = np.hstack(tr_feat[i-3:,:])
            new_tr_feat[i][(dfeat_winlen*2-sh)*num_dim:] = np.hstack(tr_feat[-2:-3-sh:-1,:])
        tr_feat = new_tr_feat
        val_feat = new_val_feat

    ## Normalization
    scaler = StandardScaler()
    tr_feat = scaler.fit_transform(tr_feat)
    val_feat = scaler.transform(val_feat)

    return {'tr_feat':tr_feat,'val_feat':val_feat,'tr_label':tr_label,'val_label':val_label, 'cfg':{'scaler':scaler,'dfeat_winlen':dfeat_winlen,'feat_type':feat_type}}

def prepare_test_data(cfg):
    print("[INFO] Start preparing testing data")
    data = np.load('data/test_data.npz')['data']
    print("[INFO] Testing data loaded")
    feat = data[0][cfg['feat_type']]
    label = data[0]['targets']
    # for i,d in enumerate(data[1:10]):
    for i,d in enumerate(data[1:]):
        feat = np.vstack((feat,d[cfg['feat_type']]))
        label.extend(d['targets'])
    num_dim = feat.shape[1]
    num_data = feat.shape[0]
    
    if cfg['dfeat_winlen'] is not None:
        dfeat_winlen = cfg['dfeat_winlen']
        new_feat = np.zeros((num_data,num_dim*(dfeat_winlen*2+1)))

        for i in range(dfeat_winlen):
            new_feat[i][(dfeat_winlen-i)*num_dim:] = np.hstack(feat[0:i+4,:])
        for i in range(dfeat_winlen,num_data-dfeat_winlen):
            new_feat[i] = np.hstack(feat[i-3:i+4,:])
        for sh,i in enumerate(range(num_data-dfeat_winlen,num_data)):
            new_feat[i][:(dfeat_winlen*2-sh)*num_dim] = np.hstack(feat[i-3:,:])
            new_feat[i][(dfeat_winlen*2-sh)*num_dim:] = np.hstack(feat[-2:-3-sh:-1,:])
        feat = new_feat

    feat = cfg['scaler'].transform(feat)

    return {'te_feat':feat,'te_label':label,'cfg':cfg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preparing Data for Phoneme Recognition Model')
    parser.add_argument('feat_type',type=str,help='feature type',choices=['lmfcc','mspec'])
    parser.add_argument('--dfeat_winlen',type=int,help='dynamic feature padding length')
    parser.add_argument('--label',action='store_true')
    args = parser.parse_args()
    data = prepare_train_data(args.feat_type,args.dfeat_winlen)
    prefix = args.feat_type
    if args.dfeat_winlen is not None:
        prefix = 'd'+prefix

    te_data = prepare_test_data(data['cfg'])
    print(prefix)

    print("[INFO] Saving features...")
    np.savez('data/{}_train_x.npz'.format(prefix),train_x = data['tr_feat'],val_x = data['val_feat'],cfg = data['cfg'])
    np.savez('data/{}_test_x.npz'.format(prefix),test_x = te_data['te_feat'])
    print("[INFO] features saved to data/")

    if args.label:
        print("[INFO] Saving labels...")
        np.savez('data/train_y.npz',train_y = data['tr_label'])
        np.savez('data/val_y.npz',val_y = data['val_label'])
        np.savez('data/test_y.npz',test_y = te_data['te_label'])
    print("[INFO] labels saved to data/")
