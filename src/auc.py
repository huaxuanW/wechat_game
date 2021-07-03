import pandas as pd

import numpy as np

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from lightgbm.sklearn import LGBMClassifier

from collections import defaultdict

import gc

import time


def uAUC(labels, preds, user_id_list):

    """Calculate user AUC"""

    user_pred = defaultdict(lambda: [])

    user_truth = defaultdict(lambda: [])

    for idx, truth in enumerate(labels):

        user_id = user_id_list[idx]

        pred = preds[idx]

        truth = labels[idx]

        user_pred[user_id].append(pred)

        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):

        truths = user_truth[user_id]

        flag = False

        # 若全是正样本或全是负样本，则flag为False

        for i in range(len(truths) - 1):

            if truths[i] != truths[i + 1]:

                flag = True

                break

        user_flag[user_id] = flag

    total_auc = 0.0

    size = 0.0

    for user_id in user_flag:

        if user_flag[user_id]:

            auc = roc_auc_score(
                np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id])
            )

            total_auc += auc

            size += 1.0

    user_auc = float(total_auc) / size

    return user_auc