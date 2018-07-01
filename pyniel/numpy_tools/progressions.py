import numpy as np

""" Arithmetic progressions
Daniel Dugas
"""


def divisors(n):
    """ lists all the divisors for the number n

    Example
    -------
    >>> divisors(20)
    array([ 2,  4,  5, 10])
    """
    x = np.abs(n)
    candidates = np.arange(2, np.floor(np.sqrt(x)) + 1)
    divisors = candidates[np.equal(np.mod(x, candidates), 0)]
    divisors2 = x * 1. / divisors[::-1]
    if divisors[-1] == divisors2[0]:
        divisors2 = divisors2[1:]
    return np.concatenate((divisors, divisors2), axis=0).astype(int)


def primes(n):
    """ Lists primes smaller than n
    using the sieve of Erasthothenes
    (half sieve, multiples of 2 already removed)

    Example
    -------
    >>> primes(120)
    array([  2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41,
            43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101,
           103, 107, 109, 113])
    """
    sieve = np.full((n // 2,), True)
    for i in range(3, int(np.floor(np.sqrt(n))) + 1, 2):
        if sieve[(i - 1) // 2]:
            sieve[(i - 1) // 2 + i :: i] = False
    primes = np.where(sieve)[0] * 2 + 1
    primes[0] = 2
    return primes


if __name__ == "__main__":
    import doctest

    doctest.testmod()
