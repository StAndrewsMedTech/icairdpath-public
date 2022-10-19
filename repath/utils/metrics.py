import matplotlib.pyplot as plt
import numpy as np


def plotROC(xvalues, yvalues, summary_value, title, xlabel, ylabel, x_axis_lim=None, y_axis_lim=None):
    title_lab = f'{title}\n score = {round(summary_value, 4)}'
    fig = plt.figure(figsize=(8,6))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    fig.suptitle(title_lab, fontsize=14)
    plt.plot(xvalues, yvalues, '-', color='#000000')
    xleft, xright = plt.xlim()
    if x_axis_lim is not None:
        xleft = x_axis_lim[0]
        xright = x_axis_lim[1]
    ylow, yhigh = plt.ylim()
    if y_axis_lim is not None:
        ylow = y_axis_lim[0]
        yhigh = y_axis_lim[1]
    plt.xlim(xleft, xright)
    plt.ylim(ylow, yhigh)
    plt.tight_layout()
    return plt


def binary_curves(true, probabilities, pos_label):
    sort_order = np.argsort(-probabilities)
    prob_order = probabilities[sort_order]
    true_order = true[sort_order]
    pos_vals = true_order == pos_label
    tps = np.cumsum(pos_vals)
    n_pos_pred = np.add(list(range(len(probabilities))), 1)
    fps = np.subtract(n_pos_pred, tps)
    total_pos = np.sum(pos_vals)
    fns = np.subtract(total_pos, tps)
    total_vals = len(probabilities)
    tns = np.subtract(np.subtract(total_vals, n_pos_pred), fns)
    precisions = np.divide(tps, np.add(tps, fps))
    recalls = np.divide(tps, np.add(tps, fns))
    fprs = np.divide(fps, np.add(tns, fps))
    return precisions, recalls, fprs, prob_order


def pre_re_curve(true, probabilities, pos_label, recall_levels):
    precisions, recalls, fprs, po = binary_curves(true, probabilities, pos_label)
    p1000 = np.interp(recall_levels, recalls, precisions)
    return p1000


def fpr_tpr_curve(true, probabilities, pos_label, recall_levels):
    precisions, recalls, fprs, po = binary_curves(true, probabilities, pos_label)
    f1000 = np.interp(recall_levels, recalls, fprs)
    return f1000


def plotROCCI(xvalues, yvalues, xcivalues, ycivalues, summary_value, summary_valueCI, title, xlabel, ylabel,
              x_axis_lim=None):
    title_lab = f'{title} \n score = {round(summary_value, 4)}, (range {round(summary_valueCI[0], 4)}, {round(summary_valueCI[1], 4)})'
    fig = plt.figure(figsize=(8,6))
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14, x=1.005)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.suptitle(title_lab, fontsize=14)
    plt.plot(xvalues, yvalues, '-', color='#000000')
    plt.fill_between(x=xcivalues, y1=ycivalues[0, :], y2=ycivalues[1, :], facecolor='grey', alpha=0.5)
    xleft, xright = plt.xlim()
    if x_axis_lim is not None:
        xleft = x_axis_lim[0]
        xright = x_axis_lim[1]
    plt.xlim(xleft, xright)
    plt.tight_layout()
    return plt


def conf_mat_raw(true, predicted, labels):
    mat_out = np.empty((len(labels), len(labels)))
    for i, row in enumerate(labels):
        preds_row = predicted[true == row]
        for j, col in enumerate(labels):
            mat_out[i, j] = np.sum(preds_row == col)
    mat_out = np.array(mat_out, dtype=np.int)
    return mat_out


def conf_mat_plot_heatmap(cm, display_labels, title_in, heatmap_type='true'):
    fig, ax = plt.subplots(figsize=(8,8))
    n_classes = cm.shape[0]
    cmap = 'Greys'

    if heatmap_type == 'percent':
        sum_vals = np.sum(cm)
    elif heatmap_type == 'true':
        sum_vals = np.reshape(np.repeat(np.sum(cm, axis=1), n_classes), (n_classes, n_classes))
    elif heatmap_type == 'pred':
        sum_vals = np.reshape(np.tile(np.sum(cm, axis=0), n_classes), (n_classes, n_classes))
        print(sum_vals)

    color_mapping = np.array(np.multiply(np.divide(cm, sum_vals), 255), np.uint8)

    for i in range(n_classes):
        for j in range(n_classes):
            text_cm = format(cm[i, j], ',')
            txt_color = [1, 1, 1] if color_mapping[i, j] > 100 else [0, 0, 0]
            ax.text(j, i, text_cm, ha="center", va="center", color=txt_color, fontsize=16)
            ax.axhline(i - .5, color='black', linewidth=1.0)
            ax.axvline(j - .5, color='black', linewidth=1.0)

    ax.matshow(color_mapping, cmap=cmap, vmin=0, vmax=255)

    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(display_labels, fontsize=14)
    ax.set_yticklabels(display_labels, fontsize=14)
    ax.set_title(title_in, fontsize=14)
    ax.tick_params(bottom=True, labelbottom=True, top=False, labeltop=False)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.tight_layout()
    return ax


def save_conf_mat_plot(cm, labels, title, results_dir, percent=False):
    n_class = len(labels)
    cm_all = cm.loc['results', :].to_numpy()
    cm_all = np.reshape(np.array(cm_all, dtype=np.int), (n_class, n_class))
    out_path = 'confusion_matrix.png'
    if percent:
        cm_all = cm_all/cm_all.sum(axis=1, keepdims=True)
        cm_all = np.multiply(cm_all, 100)
        cm_all = np.around(cm_all, 3)  
        out_path = 'confusion_matrix_percent.png'
    cm_out = conf_mat_plot_heatmap(cm_all, labels, title)  
    cm_out.get_figure().savefig(results_dir / out_path)


def conf_mat_plot_heatmap_CI(cm, cm_ci, display_labels, title_in, heatmap_type='true'):
    fig, ax = plt.subplots(figsize=(8,8))
    n_classes = cm.shape[0]
    cmap = 'Greys'
    print(display_labels)

    if heatmap_type == 'percent':
        sum_vals = np.sum(cm)
    elif heatmap_type == 'true':
        sum_vals = np.reshape(np.repeat(np.sum(cm, axis=1), n_classes), (n_classes, n_classes))
    elif heatmap_type == 'pred':
        sum_vals = np.reshape(np.tile(np.sum(cm, axis=0), n_classes), (n_classes, n_classes))
        print(sum_vals)

    color_mapping = np.array(np.multiply(np.divide(cm, sum_vals), 255), np.uint8)

    nclass = len(display_labels)
    ci_lw = np.reshape(cm_ci[0, :], (nclass, nclass))
    ci_hi = np.reshape(cm_ci[1, :], (nclass, nclass))

    for i in range(n_classes):
        for j in range(n_classes):
            text_cm = f'{cm[i, j]} \n ({int(ci_lw[i, j])}, \n {int(ci_hi[i, j])})'
            txt_color = [1, 1, 1] if color_mapping[i, j] > 100 else [0, 0, 0]
            ax.text(j, i, text_cm, ha="center", va="center", color=txt_color, fontsize=14)
            ax.axhline(i - .5, color='black', linewidth=1.0)
            ax.axvline(j - .5, color='black', linewidth=1.0)

    ax.matshow(color_mapping, cmap=cmap, vmin=0, vmax=255)

    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(display_labels, fontsize=14)
    ax.set_yticklabels(display_labels, fontsize=14)
    ax.set_title(title_in, fontsize=14, y=1.05)
    ax.tick_params(bottom=True, labelbottom=True, top=False, labeltop=False)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.tight_layout()
    return ax

def save_conf_mat_plot_ci(cm, labels, title, results_dir):
    n_class = len(labels)
    cm_all = cm.loc['results', :].to_numpy()
    cm_all = np.reshape(np.array(cm_all, dtype=np.int), (n_class, n_class))
    ci_vec = cm.loc[['ci_lower_bound', 'ci_upper_bound'], :].to_numpy()
    cm_out = conf_mat_plot_heatmap_CI(cm_all, ci_vec, labels, title)
    out_path = 'confusion_matrix_with_ci.png'
    cm_out.get_figure().savefig(results_dir / out_path)

