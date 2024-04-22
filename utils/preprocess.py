import torch
from torch.nn import functional as fn
import torch.utils.data

class ShortSequenceDataset:
    """
    Input file stores one sequence per line.
    """

    def __init__(self, fname, context_size=16, batch_size=8):
        self.context_size = context_size
        seqs = self.load_data(fname)
        self.vocab = get_vocab(seqs)
        self.init_code = self.vocab.index('^')
        self.term_code = self.vocab.index('$')
        self.X, self.Y = self.prepare(seqs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def batch(self, batch_size):
        idx = torch.randint(len(self), batch_size)
        return self[idx]

    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        return self

    def load_data(self, fname):
        with open(fname) as inf:
            seqs = []
            for line in inf:
                # add initiation and termination symbol
                site = '^' + line.rstrip() + '$'
                # replace none with '_'
                site = site.replace('none', '_')
                # TODO keep long seqs but split than during expand
                # use only small enough seqs
                if len(site) <= self.context_size:
                    seqs.append(site)
            return seqs

    def encode(self, xs):
        """
        Convert xs from tokens into integers using a vcoabulary.
        """
        return torch.tensor( [self.vocab.index(x) for x in xs] )

    def decode(self, ys, strip=False):
        """
        Convert ys from integers into tokens using a vocabulary.
        """
        if ys.dim() == 0:
            return self.vocab[ys.item()]
        elif ys.dim() == 1:
            if strip:
                chars = [self.vocab[y] for y in ys
                    if y != self.init_code and y != self.term_code]
            else:
                chars = [self.vocab[y] for y in ys]
        elif ys.dim() == 2:
            return [ self.decode(ys[i,], strip=strip) for i in range(ys.shape[0]) ]
        return ''.join(chars)

    def expand(self, encoded):
        padded = pad_left(encoded, self.context_size+1, self.init_code)
        x = padded[:len(padded)-1]
        y = padded[1:]
        return (x, y)

    def prepare(self, seqs):
        xs_all = []
        ys_all = []
        for i in range(len(seqs)):
            xs, ys = self.expand(self.encode(seqs[i]))
            xs_all.append(xs)
            ys_all.append(ys)
        return (torch.stack(xs_all), torch.stack(ys_all))


def get_vocab(rdata):
    vocab = set()
    for xs in rdata:
        for x in xs:
            vocab.add(x)
    vocab = list(vocab)
    vocab.sort()
    return vocab

def pad_left(xs, size, value):
    if len(xs) < size:
        return fn.pad(xs, (size - len(xs), 0),
            value = value)
    else:
        return xs

def pad_right(xs, size, value):
    if len(xs) < size:
        return fn.pad(xs, (0, size - len(xs)),
            value = value)
    else:
        return xs

