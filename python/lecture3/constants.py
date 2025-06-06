random_seed = 2147483647

# How many previous characters we will base our prediction on.
context_size = 4

# Number of dimensions in vector space that we map each character to.
embedding_dims = 3

# The length of a context as a "flattened" array of each of its character's embeddings.
embedded_context_dims = context_size * embedding_dims
