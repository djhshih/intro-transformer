import torch
from torch.nn import functional as fn

def load_data(fname):
    with open(fname) as inf:
        sites = []
        max_len = 16
        for line in inf:
            # add initiation and termination symbol
            site = '^' + line.rstrip() + '$'
            # replace none with '_'
            site = site.replace('none', '_')
            # use only small enough restriction sites
            if len(site) <= max_len:
                sites.append(site)
        return sites

def get_vocab(rdata):
    vocab = set()
    for xs in rdata:
        for x in xs:
            vocab.add(x)
    vocab = list(vocab)
    vocab.sort()
    return vocab

def encode(xs, vocab):
    """
    Convert xs from tokens into integers using a vcoabulary.
    """
    return torch.tensor( [vocab.index(x) for x in xs] )

def decode(ys, vocab):
    """
    Convert ys from integers into tokens using a vocabulary.
    """
    return "".join( [vocab[y] for y in ys] )

def pad_left(xs, size, value):
    if len(xs) < size:
        return fn.pad(xs, (size - len(xs), 0),
            value = value)
    else:
        return xs

def pad_left(xs, size, value):
    if len(xs) < size:
        return fn.pad(xs, (size - len(xs), 0),
            value = value)
    else:
        return xs

def expand_sequence(xs, size, init_code):
    for i in range(0, len(xs)-1):
        context = pad_left(xs[:(i+1)], size, init_code)
        target = torch.tensor(xs[i+1])
        yield (context, target)

def get_batch(data, batch_size, context_size, init_code):
    for b in range(batch_size):
        idx = torch.randint(len(data), (batch_size, ) )
        xs_all = []
        ys_all = []
        for i in idx:
            for xs, ys in expand_sequence(data[i], context_size, init_code):
                xs_all.append(xs)
                ys_all.append(ys)
        return (torch.stack(xs_all), torch.stack(ys_all))

