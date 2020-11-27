import numpy as np

def listdict2dictnp(l, keepdims=False):
    '''
        Convert a list of dicts of numpy arrays to a dict of numpy arrays.
        If keepdims is False the new outer dimension in each dict element will be
            the length of the list
        If keepdims is True, then the new outdimension in each dict will be the sum of the
            outer dimensions of each item in the list
    '''
    if keepdims:
        return {k: np.concatenate([d[k] for d in l]) for k in l[0]}
    else:
        return {k: np.array([d[k] for d in l]) for k in l[0]}
