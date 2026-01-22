from typing import Literal
import torch
from einops import rearrange


class DispersionLoss(torch.nn.Module):
    '''
    Variants (exactly as in the table):

      Decorrelation:      \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization
      l2_repel:           log E_{i,j}[exp(-D(z_i, z_j) / \tau_l2)], D(z_i, z_j) = pdist(z_i, z_j, p=2)**2
      Angular spread:     log E_{i,j}[exp(-D(z_i, z_j) / \tau_cos)], D(z_i, z_j) = - z_i z_j / (||z_i|| ||z_j||)
      Orthogonalization:  E_{i,j}[max(0, margin - D(z_i, z_j))^2]
      perplexity entropy: \sum_{(i, j)} p_{j|i} log_2 p_{j|i}, p_{j|i} = exp(-||x_i - x_j||^2 / \sigma^2)

    Notes:
      - \tau_l2, \tau_cos and margin are kept as internal constants for simplicity.
    '''
    def __init__(self,
                 variant: Literal["decorrelation", "l2_repel", "angular_spread", "orthogonalization", "perplexity_entropy"] = "angular_spread",
                 tau_l2: float = 0.5,
                 tau_cos: float = 0.5,
                 margin: float = 0.5,  # NOTE: 0.5 angular cosine distance = orthogonal.
                 epsilon: float = 1e-4):
        super().__init__()
        variant = variant.lower()
        assert variant in {"decorrelation", "l2_repel", "angular_spread", "orthogonalization", "perplexity_entropy"}
        self.variant = variant
        self.tau_l2 = float(tau_l2)
        self.tau_cos = float(tau_cos)
        self.margin = float(margin)
        self.epsilon = float(epsilon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        z: [B, L, F],
            where B: batch size. L: sequence length. F: feature dimension.
        '''
        if z.dim() != 3:
            raise ValueError(f'DispersionLoss only supports 3D [B, L, F]; got {tuple(z.shape)}.')

        B, L, F = z.shape

        if F < 2:
            raise ValueError(f'DispersionLoss expects F >= 2 in [B, L, F]; got {F}.')

        if self.variant == "decorrelation":
            # NOTE: The covariance matrix `Cov` has shape [B, L, L].
            # \sum_{(m,n), m != n} Cov_{mn}^2, after l2-normalization
            z_centered = (z - z.mean(dim=2, keepdim=True)) / z.std(dim=2, keepdim=True)
            Cov = torch.matmul(z_centered, rearrange(z_centered, 'b l f -> b f l')) / (F - 1)
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            return Cov.pow(2)[:, non_diag].mean()

        elif self.variant == "l2_repel":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = torch.cdist(z, z)**2
            logit = -D / self.tau_l2
            # Norm regularization to prevent blowing up L2 distance too much.
            norm_regularization = (z ** 2).mean() * 1e-1
            # NOTE: log-sum-exp trick for `log(mean(exp(logit)))`, only differ by a constant: -log(logit.size(1))
            constant_diff = torch.log(torch.tensor(L**2))
            return (torch.logsumexp(logit + self.epsilon, dim=(1, 2)) - constant_diff).mean() + norm_regularization

        elif self.variant == "angular_spread":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            z_norm = z / (torch.linalg.norm(z, dim=2, keepdim=True) + self.epsilon)
            cossim = z_norm @ rearrange(z_norm, 'b l f -> b f l')
            cossim = torch.clamp(cossim, -1 + self.epsilon, 1 - self.epsilon)
            D = torch.arccos(cossim) / torch.pi
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            logit = -D[:, non_diag] / self.tau_cos
            # NOTE: log-sum-exp trick for `log(mean(exp(logit)))`, only differ by a constant: -log(logit.size(1))
            constant_diff = torch.log(torch.tensor(L * (L - 1)))
            return (torch.logsumexp(logit + self.epsilon, dim=1) - constant_diff).mean()

        elif self.variant == 'orthogonalization':
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            z_norm = z / (torch.linalg.norm(z, dim=2, keepdim=True) + self.epsilon)
            cossim = z_norm @ rearrange(z_norm, 'b l f -> b f l')
            cossim = torch.clamp(cossim, -1 + self.epsilon, 1 - self.epsilon)
            D = torch.arccos(cossim) / torch.pi
            non_diag = ~torch.eye(L, dtype=torch.bool, device=z.device)
            diff = torch.clamp(self.margin - D, min=0.0)
            return diff.pow(2)[:, non_diag].mean()

        elif self.variant == "perplexity_entropy":
            # NOTE: The distance matrix matrix `D` has shape [B, L, L].
            D = torch.cdist(z, z)**2
            # Use fixed sigma (bandwidth)
            sigma_sq = self.tau_l2
            logit = -D / sigma_sq
            # Set diagonal to -inf so p_{i|i} = 0 after softmax
            mask = torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0)
            logit = logit.masked_fill(mask, float('-inf'))
            # Compute conditional probabilities p_{j|i} using softmax
            P = torch.softmax(logit, dim=2)
            # Compute entropy: - sum_j p_{j|i} log p_{j|i}
            log_P = torch.log2(P + self.epsilon)
            entropy_per_point = -torch.sum(P * log_P, dim=2)  # [B, L]
            # Minimize negative entropy to maximize entropy.
            return -entropy_per_point.mean()


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from midtrain_gpt2_huggingface.midtrain_gpt2 import CausalLMLoss

    import os
    import sys
    import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    sys.path.insert(0, os.path.join(import_dir, 'prelim'))
    from utils.text_data import get_random_long_text

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # grab the first non-empty line
    text = get_random_long_text('wikipedia', min_word_count=1200, max_word_count=1500)
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        add_special_tokens=False
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.train()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    base_loss_fn = CausalLMLoss()
    base_loss = base_loss_fn(out.logits, input_ids)

    # Use all layer activations as z: [B, L, F]
    # Detach so each test is a fresh leaf tensor.
    z_base_list = [vec.detach() for vec in out.hidden_states]

    # 3) Your exact test loop, but with real hidden states
    for variant in ["decorrelation", "l2_repel", "angular_spread", "orthogonalization", "perplexity_entropy"]:
        print(f"\nVariant: {variant}")
        loss_fn = DispersionLoss(variant=variant)

        print(f"Base loss: {base_loss.item():.3f}")
        loss = 0
        for z_base in z_base_list:
            # fresh leaf w/ grads each time
            z = z_base.clone().requires_grad_(True)
            loss += loss_fn(z)
        loss /= len(z_base_list)
        print(f"Dispersion loss: {loss.item():.3f}")

        loss.backward()
        print(f"Gradient norm: {torch.norm(z.grad).item():.6f}")
