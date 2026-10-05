import torch
import torch.nn as nn


def train_linear_probe(train_embeddings, train_labels, test_embeddings, test_labels,
                        num_classes, device, steps=300, lr=1e-2):
    """Freeze encoder, train ONLY a linear readout on top. This is the valid
    dependent variable (per Opus audit) -- NOT raw self-supervised loss,
    which is not comparable across architectures with different embedding
    scales/norms."""
    in_dim = train_embeddings.shape[1]
    probe = nn.Linear(in_dim, num_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)
    test_embeddings = test_embeddings.to(device)
    test_labels = test_labels.to(device)

    for _ in range(steps):
        opt.zero_grad()
        logits = probe(train_embeddings)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

    with torch.no_grad():
        test_logits = probe(test_embeddings)
        test_acc = (test_logits.argmax(dim=1) == test_labels).float().mean().item()
        train_logits = probe(train_embeddings)
        train_acc = (train_logits.argmax(dim=1) == train_labels).float().mean().item()

    return {"train_acc": train_acc, "test_acc": test_acc, "final_train_loss": loss.item()}


class TrivialConstantEncoder(nn.Module):
    """The mandatory control baseline (per Opus audit): outputs the SAME
    vector regardless of input. If a real encoder doesn't beat this on the
    downstream probe, it has demonstrated zero value -- no matter how low its
    self-supervised loss was."""

    def __init__(self, repr_dim):
        super().__init__()
        self.repr_dim = repr_dim
        self.constant = nn.Parameter(torch.randn(repr_dim) * 0.01, requires_grad=False)

    def forward(self, x):
        batch = x.shape[0]
        return self.constant.unsqueeze(0).expand(batch, -1).clone()
