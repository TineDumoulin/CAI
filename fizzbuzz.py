"""Docstring."""
from functools import partial

ORIGINAL_DICTIONARY = {}

def is_divisible(x, y):
    return len(x) % y == 0

def generalized_fizzbuzz(word, dictionary):
    """ returns THR33."""
    output = []
    for k, v in dictionary.items():
        # k = number
        # v = string
        if is_divisible(word, k):
            output.append(v)
    if output:
        return "AND".join(output)
    return word

fizzbuzz = partial(generalized_fizzbuzz, dictionary=ORIGINAL_DICTIONARY)