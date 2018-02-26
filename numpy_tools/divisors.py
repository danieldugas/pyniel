def divisors(x):
    if np.max(x) > 100000:
        print("Warning: this function is not optimized for large numbers. Aborting.")
        raise MemoryError
    if np.isscalar(x):
        return np.arange(2,x)[np.where((x % np.arange(2,x)) == 0)]
    else:
        return tuple([divisors(k) for k in x])
