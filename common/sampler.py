import  torch



class SubsetRandomSampler(object):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, indices, generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator))

    def __len__(self):
        return len(self.indices)
