import torch

from .names import stoi, words

from .constants import context_size

def build_training_data(words):
    """
    Return X, Y tuple of training data and labels given words.

    X will contain one row for each example. Each example will contain `context_size`
    elements representing character indices.

    Y will contain a character index label for each example.
    """

    xs = []
    ys = []
    for word in words:
        context = [0] * context_size
        for ch in word:
            ich = stoi[ch]
            xs.append(context)
            ys.append(ich)
            context = context[1:] + [ich]
        xs.append(context)
        ys.append(0)
    assert len(xs) == len(ys)
    X = torch.tensor(xs)
    Y = torch.tensor(ys)
    return X, Y

train_cutoff = int(0.8 * len(words))
dev_cutoff = int(0.9 * len(words))

X_train, Y_train = build_training_data(words[:train_cutoff])
X_dev, Y_dev = build_training_data(words[train_cutoff:dev_cutoff])
X_test, Y_test = build_training_data(words[dev_cutoff:])
