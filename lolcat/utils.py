from sklearn.metrics import confusion_matrix
import numpy as np


def cm2str(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, format='.2f'):
    """Pretty print for confusion matrixes. Taken from https://gist.github.com/zachguo/10296432."""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    out = "    " + empty_cell + " "
    for label in labels:
        out += "%{0}s".format(columnwidth) % label
    out += "\n"
    for i, label1 in enumerate(labels):
        out += "    %{0}s ".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}{1}".format(columnwidth, format) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            out += cell + " "
        out += "\n"
    return out


def compute_isi_distribtuion(data: np.ndarray):
    """
    Computes the ISI distribution for each neuron in the dataset.
    data: neurons x trials x time array of events
    """
    isi_matrix = np.zeros_like(data)
    out = np.nonzero(data)
    trial_index = out[0] * data.shape[1] + out[1]
    time_index = trial_index * 2 * data.shape[2] + out[2]  # add a gap bigger than the duration of the trial

    isi = np.concatenate([np.zeros(1, dtype=np.int64), time_index[1:] - time_index[:-1]])
    isi[isi >= data.shape[2]] = 0
    np.add.at(isi_matrix, (out[0], out[1], isi), np.ones_like(isi))
    isi_matrix = isi_matrix[..., 1:]
    return isi_matrix