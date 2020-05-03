import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from IPython import get_ipython
from sklearn.metrics import roc_auc_score, roc_curve, auc
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
get_ipython().run_line_magic('matplotlib', 'inline')


def plot_histograms(df, features):
    n = len(features)
    fig = plt.figure(figsize=(12, n))

    for i, feature in enumerate(features):
        if n <= 2:
            r = 1
        else:
            r = int(np.ceil(n / 2)) + n % 2
        ax = fig.add_subplot(r, 2, i + 1)
        ax.hist(df[feature].dropna(), bins=20,
                color='#00A0A0')
        ax.set_title("'%s' Feature Distribution" % (feature), fontsize=12)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        t = np.linspace(0, np.max(df[feature]), 5)
        ax.set_xticks(t)
        ax.set_xticklabels(np.round(t, 2))

    # Only perform tight layout if there is more than 1 feature
    if n > 1:
        fig.tight_layout()
    fig.show()


def plot_counts(df, feature):
    if feature == 'TARGET':
        pal = {0: 'green', 1: 'red'}
    else:
        pal = sns.color_palette()
    n = len(df[feature])
    v = len(df[feature].unique())
    plt.figure(figsize=(12, 5))
    ax = sns.countplot(x=feature, data=df, palette=pal, alpha=.8)
    plt.title('Frequency and Counts')
    plt.xlabel(feature)

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax.set_ylabel('Count')
    ax2.set_ylabel('Frequency %')

    # if more than 6 values for feature rotate tick labels
    if v > 8:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                           fontsize=8)
    # Only add % labels if there are less than 6 features
    else:
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / n), (x.mean(), y),
                        ha='center', va='bottom')

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to 0-100
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, n)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax2.grid(None)

    plt.show()


def plot_roc_auc_curve(y_test, y_scores):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC Curve
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def evaluate(results, roc_auc):

    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']

    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'roc_auc_score_train',
                                   'pred_time', 'roc_auc_score_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j // 2, j % 2].bar(i + k * bar_width,
                                      results[learner][i][metric],
                                      width=bar_width, color=colors[k])
                ax[j // 2, j % 2].set_xticks([0.45, 1.45, 2.45])
                ax[j // 2, j % 2].set_xticklabels(["1%", "10%", "100%"])
                ax[j // 2, j % 2].set_xlabel("Training Set Size")
                ax[j // 2, j % 2].set_xlim((-0.1, 3.0))

    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("ROC AUC Score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("ROC AUC Score")

    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("ROC AUC Score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("ROC AUC Score on Testing Set")

    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y=roc_auc, xmin=-0.1, xmax=3.0, linewidth=1,
                     color='k', linestyle='dashed')
    ax[1, 1].axhline(y=roc_auc, xmin=-0.1, xmax=3.0, linewidth=1,
                     color='k', linestyle='dashed')

    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=learner))
    plt.legend(handles=patches, bbox_to_anchor=(-.25, 2.53),
               loc='upper center', borderaxespad=0., ncol=3,
               fontsize='x-large')

    # Aesthetics
    plt.suptitle("Performance Metrics for Three Supervised Learning Models",
                 fontsize=16, y=1.10)
    plt.tight_layout()
    plt.show()
