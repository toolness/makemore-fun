import random

from .constants import random_seed

words = open('names.txt', 'r').read().splitlines()

random.Random(random_seed).shuffle(words)

_chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(_chars)}
stoi['.'] = 0

itos = {i:s for s,i in stoi.items()}

# Number of characters in our alphabet (the very first one is the terminator character).
vocab_size = len(stoi)
