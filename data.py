import torch


def make_sequence_batch(batch_size, seq_len, dim, device, difficulty="easy"):
    """Synthetic world to predict. `difficulty="easy"` = single damped sinusoid
    per dimension (original skeleton task). `difficulty="hard"` = two coupled
    frequency components per dimension PLUS cross-dimension coupling (dim i's
    phase is nudged by dim i-1's value) and heavier noise -- removes the
    "trivially near-zero loss for everyone" ceiling effect."""
    t = torch.arange(seq_len, device=device).float().unsqueeze(0).unsqueeze(-1)

    if difficulty == "easy":
        freq = torch.rand(batch_size, 1, dim, device=device) * 0.3 + 0.05
        phase = torch.rand(batch_size, 1, dim, device=device) * 2 * torch.pi
        decay = torch.rand(batch_size, 1, dim, device=device) * 0.02
        amp = torch.rand(batch_size, 1, dim, device=device) * 0.8 + 0.2
        signal = amp * torch.exp(-decay * t) * torch.sin(freq * t + phase)
        noise_std = 0.03
    elif difficulty == "hard":
        freq1 = torch.rand(batch_size, 1, dim, device=device) * 0.3 + 0.05
        freq2 = torch.rand(batch_size, 1, dim, device=device) * 0.9 + 0.3
        phase1 = torch.rand(batch_size, 1, dim, device=device) * 2 * torch.pi
        phase2 = torch.rand(batch_size, 1, dim, device=device) * 2 * torch.pi
        decay = torch.rand(batch_size, 1, dim, device=device) * 0.015
        amp1 = torch.rand(batch_size, 1, dim, device=device) * 0.6 + 0.2
        amp2 = torch.rand(batch_size, 1, dim, device=device) * 0.4 + 0.1

        base = amp1 * torch.sin(freq1 * t + phase1) + amp2 * torch.sin(freq2 * t + phase2)
        base = base * torch.exp(-decay * t)

        # cross-dimension coupling: shift each dim's phase by the previous
        # dim's running value, so dims are NOT independent (harder to predict
        # from a context window alone without actually modeling interaction)
        coupled = base.clone()
        for d in range(1, dim):
            coupled[:, :, d] = coupled[:, :, d] + 0.25 * torch.roll(base[:, :, d - 1], shifts=1, dims=1)

        noise_std = 0.08
        signal = coupled
    else:
        raise ValueError(f"unknown difficulty: {difficulty}")

    noise = torch.randn_like(signal) * noise_std
    return signal + noise  # (batch, seq_len, dim)


def split_context_target(batch, context_frac=0.6):
    seq_len = batch.shape[1]
    ctx_len = int(seq_len * context_frac)
    context = batch[:, :ctx_len, :]
    target = batch[:, ctx_len:, :]
    return context, target
