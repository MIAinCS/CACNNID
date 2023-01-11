import numpy as np
import time
from cleanlab.filter import find_label_issues

def clean(filter_mode):
    test_result = [each.strip().split('\t') for each in open('all_pred.txt')]
    labels = []
    pred_probs = []
    paths = []
    for rows in test_result:
        labels.append(int(rows[2]))
        pred_probs.append([float(rows[0]),float(rows[1])])
        paths.append(rows[-1])
    pred_probs = np.array(pred_probs)

    start = time.time()

    ranked_label_issues = find_label_issues(
        labels,
        pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by = filter_mode,
        frac_noise = 0.8
    )

    end = time.time()

    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")
    print(f"time = {end-start}")


    with open('./wash_result/error_file_{}.txt'.format(filter_mode),'w') as fw:
        for index in ranked_label_issues:
            fw.write(paths[index])
            fw.write('\n')

if __name__ == '__main__':
    filter_list = ['prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given']
    
    for filter_mode in filter_list:
        print(f'start clean! mode = {filter_mode}')
        clean(filter_mode)