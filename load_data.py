import numpy as np
import pandas as pd


def load_train_test(f='../data/OutpatientONC_v1_1_enc_data_features.csv',
    rnd_seed=123456, prop_train=0.7, label_name='label'):
    enc_data = pd.read_csv(f)

    enc_data['DEATH_DATE'] = pd.to_datetime(enc_data['DEATH_DATE'])
    enc_data['APPT_TIME'] = pd.to_datetime(enc_data['APPT_TIME'])

    res2 = enc_data

    np.random.seed(rnd_seed)

    ## Randomly sample one encounter per EMPI
    idx = list(res2.index)
    np.random.shuffle(idx)
    res2 = res2.iloc[idx,:].drop_duplicates('EMPI')
    ##

    res2.index = range(res2.shape[0])
    EMPIs = res2['EMPI'].unique()

    n_train = int(len(EMPIs) * prop_train)

    print(n_train, len(EMPIs) - n_train)

    np.random.shuffle(EMPIs)
    train_empis = EMPIs[:n_train]
    test_empis = EMPIs[n_train:]
    train_idx = res2['EMPI'].isin(train_empis)
    test_idx = res2['EMPI'].isin(test_empis)

    train = res2[train_idx]
    test = res2[test_idx]
    return train, test