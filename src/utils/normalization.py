
def min_max(data):
    _min, _max = data.min(), data.max()
    return (data - _min) / (_max - _min)


def mean_std(data):
    return (data - data.mean()) / data.std()
