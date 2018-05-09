import numpy as np
from states import *
import editdistance

def frameByFrame(level,pred,ans):
    """
    pred and ans are np array with integer type
    """
    num_frame = ans.shape[0]
    if level == 'state':
        return np.sum(pred == ans)/num_frame
    else: # phoneme
        pred = np.array([stateList[idx][:-2] for idx in pred])
        ans = np.array([stateList[idx][:-2] for idx in ans])
        return np.sum(pred == ans)/num_frame

def edit_distance(s1,s2):
    m = len(s1)
    n = len(s2)

    print(m,n)
    DP = np.zeros((m+1,n+1),dtype=np.uint8)
    for i in range(m+1):
        DP[i][0] = i
    for j in range(n+1):
        DP[0][j] = j
    
    for i in range(1,m+1):
        for j in range(1,n+1):
            DP[i][j] = min(DP[i-1][j] + 1, DP[i][j-1]+1,DP[i-1][j-1] + int(s1[i-1] != s2[j-1]))
    return DP[m][n]

def phone_error_rate(level,pred,ans):
    te = np.load('data/test_data.npz')['data']
    utt_len_ls = [len(utt['targets']) for utt in te]

    # pred = pred[:162]
    # ans = ans[:162]
    # utt_len_ls = [[len(utt['targets']) for utt in te][0]]
    
    cur_ptr = 0
    s = 0
    print("Total {} utterances".format(len(utt_len_ls)))
    for cnt,utt_len in enumerate(utt_len_ls):
        if level == 'state':
            tmp_pred = np.array([stateList[idx] for idx in pred[cur_ptr:cur_ptr + utt_len]])
            tmp_ans = np.array([stateList[idx] for idx in ans[cur_ptr:cur_ptr + utt_len]])
        else:
            tmp_pred = np.array([stateList[idx][:-2] for idx in pred[cur_ptr:cur_ptr + utt_len]])
            tmp_ans = np.array([stateList[idx][:-2] for idx in ans[cur_ptr:cur_ptr + utt_len]])

        cur_ptr += utt_len
        
        new_pred = [tmp_pred[0]]
        for st in tmp_pred[1:]:
            if new_pred[-1] != st:
                new_pred.append(st)

        new_ans = [tmp_ans[0]]
        for st in tmp_ans[1:]:
            if new_ans[-1] != st:
                new_ans.append(st)

        s += editdistance.eval(new_pred,new_ans)/len(new_ans)
        # s += edit_distance(new_pred,new_ans)/len(new_ans)

    return (100 * s) / len(utt_len_ls)

if __name__ == "__main__":
    s1 = "AGCCT"
    s2 = "ATCT"
    # print(editdistance(s1,s2))
    print(edit_distance(s1,s2))
