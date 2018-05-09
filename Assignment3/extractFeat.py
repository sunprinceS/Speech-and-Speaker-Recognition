import os
from prondict import *
from lab3_proto import *
import argparse

parser = argparse.ArgumentParser(description='Extract Feature from Audio Sample')
parser.add_argument('datapath',type=str, help='wav files location')
parser.add_argument('outfile' ,type=str, help='output file name')

args = parser.parse_args()
data = []
cnt = 0

for root, dirs, files in os.walk(args.datapath):
    for file in files:
        if file.endswith('.wav'):
            cnt += 1
            filename = os.path.join(root,file)
            wordTrans = list(path2info(filename)[2])
            phoneTrans = words2phones(wordTrans,prondict)
            samples,sr_rate = loadAudio(filename)
            mspec = melspec(samples)
            lmfcc = mfcc(samples)
            targets = forcedAlignment(lmfcc,phoneHMMs,phoneTrans,''.join(path2info(filename)[2:]))
            data.append({'filename':filename,'lmfcc':lmfcc,'mspec':mspec,'targets':targets})
            if cnt % 100 == 0:
                print("[INFO]: Preprocess {} files to {}".format(cnt,args.outfile))
np.savez('data/{}.npz'.format(args.outfile),data = data)

