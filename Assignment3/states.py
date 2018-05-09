import numpy as np

phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}


with open('stateList') as f:
    stateList  = f.read().splitlines()
with open('phonemeList') as f:
    phonemeList = f.read().splitlines()

if __name__ == "__main__":
    stateList = [ph + '_' + str(idx) for ph in phones for idx in range(nstates[ph])]
    with open('stateList','w') as f:
        f.write('\n'.join(stateList))
    with open('phonemeList','w') as f:
        f.write('\n'.join(phones))
