import pandas as pd
import matplotlib.pylab as plt
from scipy import stats
import numpy as np
from sklearn import metrics


def beta_errors(num, denom):
    return stats.beta.interval(.95, num+1, denom-num+1)


def calibration_curve_error_bars(a, p, n_bins=10):
    pmin, pmax = p.min(), p.max()
    binstarts = np.linspace(pmin, pmax, n_bins+1)
    bincentres = binstarts[:-1] + (binstarts[1] - binstarts[0])/2.0
    numerators = np.zeros(n_bins)
    denomonators = np.zeros(n_bins)
    for b in range(n_bins):
        idx_bin = (p >= binstarts[b]) & (p < binstarts[b+1])
        denomonators[b] = idx_bin.sum()
        numerators[b] = a[idx_bin].sum()

    errors = beta_errors(numerators, denomonators)
    return bincentres, numerators, denomonators, errors


def plot_calibration_curve_error_bars(a, p, n_bins=10, ax=None, alpha=1.0, label='',
    add_n=False):
    x, n, d, err = calibration_curve_error_bars(a, p, n_bins)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.errorbar(x, n/d, yerr=[n/d-err[0], err[1]-n/d],alpha=alpha,label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of positives",color='blue')
    ax.set_xlabel("Mean predicted value")
    if add_n:
        ax2 = ax.twinx()
        ax2.step(x,d,color='green',alpha=0.5)
        ax2.set_ylabel('N', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    ax.plot([0, 1], [0, 1], "k:")


def calc_metrics(y, preds_proba, thresh=0.5):
    m = {}
    preds = preds_proba[:, 1] > thresh
    cm = metrics.confusion_matrix(y, preds)
    m['proportion +'] = preds.mean()
    m['confusion_matrix'] = cm
    m['Accuracy'] = metrics.accuracy_score(y, preds)
    m['F1 score'] = (metrics.f1_score(y, preds))
    m['FP'] = (cm[0, 1]/(cm[:, 1].sum()*1.0))
    m['FN'] = (cm[1, 0]/(cm[:, 0].sum()*1.0))
    m['Specificity'] = (cm[0, 0]/(cm[0, :].sum()*1.0))
    m['Sensitivity'] = (cm[1, 1]/(cm[1, :].sum()*1.0))
    m['PPV'] = (cm[1, 1]/(cm[:, 1].sum()*1.0))
    m['NPV'] = (cm[0, 0]/(cm[:, 0].sum()*1.0))
    fpr, tpr, thresholds = metrics.roc_curve(y, preds_proba[:, 1])
    m['AUC'] = metrics.auc(fpr, tpr)
    return m


def plot_metrics(y, preds, thresh_range=(0, 1), ax=None):
    m_list = []
    for tr in np.linspace(thresh_range[0], thresh_range[1]):
        m = calc_metrics(y, preds, tr)
        m.update({'threshold': tr})
        m_list.append(m)

    m_df = pd.DataFrame(m_list)

    for metric in ['Sensitivity',
                   'Specificity',
                   'PPV',
                   'F1 score',
                   'proportion +']:  # ,'FN','FP'
        ax.plot(m_df['threshold'], m_df[metric], label=metric)

    ax.legend(bbox_to_anchor=(1.35, 1.0))
    return m_df


def auc_ci(y, p, bootstrap=1000, ci=95):
    n = y.shape[0]
    idx_choose = range(n)
    aucs = []
    for b in range(bootstrap):
        idx = np.random.choice(idx_choose, size=n, replace=True)
        yy = y[idx]
        pp = p[idx]
        if (yy.mean() != 0) and (yy.mean() != 1):
            #print(yy, pp)
            _, _, auc = calc_auc(yy, pp)
            aucs.append(auc)
        else:
            print('warning one class only in bootstrap')
    aucs = np.array(aucs)
    lower, upper = np.percentile(
        aucs,
        [(100 - ci) / 2.0,
         100 - (100 - ci) / 2.0])
    mean = aucs.mean()
    return mean, lower, upper


def calc_auc(y, p):
    fpr, tpr, thresholds = metrics.roc_curve(y, p)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc


def plt_auc(pred, actual, ax):
    fpr, tpr, auc = calc_auc(actual, pred[:, 1])
    ax.plot(fpr, tpr)
    mean, lower, upper, = auc_ci(actual.values, pred[:, 1])
    ax.plot([0, 1], [0, 1], '--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.text(0.7, 0.2, 'AUC = %0.2f [%0.3f-%0.3f]' % (auc, lower, upper))
