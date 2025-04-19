from dataclasses import dataclass

import torch
from torch import nn

from .names import vocab_size, stoi, itos
from .constants import embedding_dims, embedded_context_dims, random_seed, context_size
from .training_data import X_train, Y_train


class MakemoreModel:
    # Number of neurons in the hidden layer
    w1_neurons = 200

    def __init__(self, w2_scale=0.1, b2_scale=0):
        self.g = torch.Generator().manual_seed(random_seed)

        # Matrix containing a "lookup table" from character indices to their embeddings in the vector space.
        self.C = torch.randn((vocab_size, embedding_dims), dtype=torch.float, generator=self.g)

        # Hidden tanh layer
        self.W1 = torch.randn((embedded_context_dims, self.w1_neurons), dtype=torch.float, generator=self.g)

        self.b1 = torch.randn(self.w1_neurons, dtype=torch.float, generator=self.g)

        # Final softmax layer, scaled by w2_scale (0.1) to make the initial weights be as similar to
        # each other as possible to start, thus ultimately giving each character an equal
        # probability, which results in a much better initial loss (described in beginning
        # of lecture 4).
        self.W2 = torch.randn((self.w1_neurons, vocab_size), dtype=torch.float, generator=self.g) * w2_scale

        # Initialize softmax biases to b2_scale (0) so every character has equal probability (see above).
        self.b2 = torch.randn(vocab_size, dtype=torch.float, generator=self.g) * b2_scale

        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]

        self.training_loss = []

        for param in self.params:
            param.requires_grad = True

    def forward(self, X):
        num_examples = X.shape[0]

        # Each row is an example consisting of a "flattened" tensor of each character in the context.
        CX = self.C[X].view(num_examples, embedded_context_dims)

        # Make sure the very first example's first context item is the terminator character.
        # Commenting this out b/c we want this code to be used for more than just training!
        #terminator = C[0]
        #assert CX[0][:embedding_dims].tolist() == terminator.tolist()

        CXW1 = torch.tanh(CX @ self.W1 + self.b1)

        logits = CXW1 @ self.W2 + self.b2

        # I tried to use torch's softmax here to improve efficiency but it actually made things SLOWER (???).
        fake_counts = logits.exp()

        probs = fake_counts / torch.sum(fake_counts, dim=1, keepdim=True)

        return ForwardPassResult(
            CX=CX,
            CXW1=CXW1,
            logits=logits,
            probs=probs
        )

    def calc_loss(self, probs, Y):
        num_examples = probs.shape[0]
        assert num_examples == Y.shape[0]

        # I tried to use torch's cross-entropy loss here to improve efficiency but it actually made things SLOWER (???).
        loss = -probs[range(num_examples), Y].log().mean()

        return loss

    def train(self, rounds, first_half_lr=0.1, second_half_lr=0.01, minibatch_size=32, X=X_train, Y=Y_train):
        num_examples = X.shape[0]

        for i in range(rounds):
            minibatch_indexes = torch.randint(0, num_examples, (minibatch_size,), generator=self.g)
            minibatch = X[minibatch_indexes]

            probs = self.forward(minibatch).probs

            loss = self.calc_loss(probs, Y[minibatch_indexes])

            self.training_loss.append(loss.item())

            learning_rate = first_half_lr if i < rounds / 2 else second_half_lr

            if i % 1_000 == 0:
                print(f"{i:7d} / {rounds:7d} LR={learning_rate:.2f} minibatch loss: {loss.item():.4f}")

            for param in self.params:
                param.grad = None
            
            loss.backward()

            for param in self.params:
                param.data += -learning_rate * param.grad

    @torch.no_grad
    def calc_loss_for_dataset(self, X, Y):
        return self.calc_loss(self.forward(X).probs, Y).item()

    def predict(self, context_str='', num_chars=1000, stop_on_terminator=True, greedy=False):
        """
        Given an optional starting context, predicts next character(s) in the sequence.
        """

        while num_chars > 0:
            context = ([0] * context_size + [stoi[ch] for ch in context_str])[-context_size:]
            X = torch.tensor([context])
            probs = self.forward(X).probs
            if greedy:
                next_idx = probs[0].argmax().item()
            else:
                next_idx = torch.multinomial(probs[0], 1, replacement=True, generator=self.g).item()
            if next_idx == 0 and stop_on_terminator:
                break
            context_str = context_str + itos[next_idx]
            num_chars -= 1
        return context_str


@dataclass
class ForwardPassResult:
    CX: torch.Tensor

    CXW1: torch.Tensor

    logits: torch.Tensor

    probs: torch.Tensor


def test_model():
    from .training_data import X_test

    model = MakemoreModel()

    X = X_test
    num_examples = X_test.shape[0]

    result = model.forward(X)

    # Ensure the probabilities of all characters in the first example sum to approximately 1.0.
    assert result.probs[0].sum() - 1.0 < 0.00000

    assert list(result.CXW1.shape) == [num_examples, model.w1_neurons]

    assert list(result.logits.shape) == [num_examples, vocab_size]
