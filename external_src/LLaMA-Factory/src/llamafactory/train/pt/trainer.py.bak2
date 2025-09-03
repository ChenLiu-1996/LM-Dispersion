# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Aug 28, 2025: Adjusted to support dispersion loss.
see https://github.com/ChenLiu-1996/Transformer-Dispersion/blob/main/transformer_dispersion/midtrain_gpt2_huggingface/midtrain_gpt2.py#L362
"""

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


class DispersionLoss(torch.nn.Module):
    '''
    Variants (exactly as in the table):

      InfoNCE:     log E_{i,j}[ exp( - D(z_i, z_j) / \tau ) ]
      Hinge:       E_{i,j}[ max(0, margin - D(z_i, z_j))^2 ]
      Covariance:  \sum_{m,n} Cov_{mn}^2

    Notes:
      - D is Euclidean distance (L2), not squared.
      - \tau and margin are kept as internal constants for simplicity.
      - To avoid OOM, pairwise computations are chunked in tiles.
    '''
    def __init__(self, variant: str, tau: float = 1.0, margin: float = 1.0, tiny: float = 1e-9,
                 block_size: int = 512):
        super().__init__()
        v = (variant or "").lower()
        assert v in {"infonce", "hinge", "covariance"}
        self.variant = v
        self.tau = float(tau)
        self.margin = float(margin)
        self.tiny = tiny
        self.block_size = int(block_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        z: [N, feature] features from a **single sequence** (non-ignored tokens only).
        '''
        if z.dim() != 2 or z.size(0) < 2:
            # If fewer than 2 tokens, no pairwise terms; return 0
            return z.new_zeros(())

        if self.variant == "covariance":
            # \sum_{m,n} Cov_{mn}^2   (sample covariance over tokens in the sequence)
            n = z.size(0)
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.t() @ zc) / (n - 1)
            return (cov * cov).sum()

        # For InfoNCE / Hinge: compute over all unordered pairs i<j in CHUNKS
        return self._pairwise_reduced_loss(z)

    def _pairwise_reduced_loss(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Chunked upper-triangular pairwise reduction to avoid materializing all pairs.
        '''
        n = z.size(0)
        bs = self.block_size
        total_sum = z.new_zeros(())
        total_cnt = 0

        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            zi = z[i0:i1]  # [bi, F]
            for j0 in range(i0, n, bs):
                j1 = min(j0 + bs, n)
                zj = z[j0:j1]  # [bj, F]

                # pairwise Euclidean distances for the tile: [bi, bj]
                d = torch.cdist(zi, zj, p=2)

                if i0 == j0:
                    # keep only upper triangle without diagonal
                    tri = torch.triu_indices(d.size(0), d.size(1), offset=1, device=d.device)
                    d = d[tri[0], tri[1]]

                if self.variant == "infonce":
                    contrib = torch.exp(-d / max(self.tau, self.tiny)).sum()
                else:  # 'hinge'
                    margin = torch.clamp(self.margin - d, min=0.0)
                    contrib = (margin * margin).sum()

                total_sum = total_sum + contrib
                total_cnt += d.numel()

        return total_sum / max(total_cnt, 1)

class CausalLMLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, seq_len, V], labels: [B, seq_len]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss

class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Initialize dispersion loss
        self.use_disp = (finetuning_args.dispersion is not None and 
                        finetuning_args.dispersion_coeff > 0.0)
        self.disp_coeff = finetuning_args.dispersion_coeff
        self.disp_loc = finetuning_args.dispersion_loc
        self.disp_eval = finetuning_args.dispersion_eval  # Add this line

        if self.use_disp:
            variant = finetuning_args.dispersion.lower()
            assert variant in {"infonce", "hinge", "covariance"}
            self._disp_fn = DispersionLoss(variant=variant)
        else:
            self._disp_fn = None

        self.loss_fn = CausalLMLoss()
        
        # Track logging to avoid duplicate logs per global step
        self._last_logged_step = -1
        self._current_accumulation_step = 0

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs["labels"]
        
        # Determine if we should compute dispersion
        # Training: use dispersion if enabled
        # Evaluation: use dispersion only if both enabled AND disp_eval is True
        should_compute_disp = (self.use_disp and 
                              (model.training or self.disp_eval))
        
        outputs = model(**inputs, output_hidden_states=should_compute_disp)
        
        # Compute standard loss - either use model's built-in loss or compute manually
        if hasattr(outputs, "loss") and outputs.loss is not None:
            raw_loss = outputs.loss
        else:
            # Fallback: compute manually
            logits = outputs.logits
            raw_loss = self.loss_fn(logits, labels)

        # Apply gradient accumulation scaling ONLY during training
        if model.training and self.args.gradient_accumulation_steps > 1:
            scaled_loss = raw_loss / self.args.gradient_accumulation_steps
        else:
            scaled_loss = raw_loss
        
        total = scaled_loss

        # print(f"DEBUG: Raw loss: {float(raw_loss)}")
        # print(f"DEBUG: Scaled loss: {float(scaled_loss)}")

        # Add dispersion if enabled
        if should_compute_disp:
            disp_val = self._dispersion_from_hidden_states(outputs.hidden_states, labels)
            # Scale dispersion loss the same way as standard loss (only during training)
            if model.training and self.args.gradient_accumulation_steps > 1:
                scaled_disp_val = disp_val / self.args.gradient_accumulation_steps
            else:
                scaled_disp_val = disp_val
            
            # print(f"DEBUG: Dispersion loss: {float(disp_val)}")
            total = scaled_loss + self.disp_coeff * scaled_disp_val
            # print(f"DEBUG: Total loss: {float(total)}")
            
            # Track gradient accumulation step
            if model.training:
                self._current_accumulation_step = (self._current_accumulation_step + 1) % self.args.gradient_accumulation_steps
                is_last_accumulation_step = (self._current_accumulation_step == 0)
            else:
                is_last_accumulation_step = True  # Always log during eval
            
            # Log metrics only at proper intervals to avoid spam
            # For training: only log on last accumulation step and at logging intervals
            # For eval: always log
            should_log = (not model.training or 
                         (is_last_accumulation_step and
                          hasattr(self.state, 'global_step') and 
                          self.state.global_step > 0 and 
                          self.state.global_step % self.args.logging_steps == 0 and
                          self.state.global_step != self._last_logged_step))
            
            if should_log:
                if model.training:
                    self._last_logged_step = self.state.global_step  # Mark as logged
                    self.log({
                        "train/dispersion_loss": float(disp_val.detach()),
                        "train/standard_loss": float(raw_loss.detach()),
                        "train/total_loss": float(total.detach())
                    })
                else:
                    # Evaluation metrics
                    self.log({
                        "eval/dispersion_loss": float(disp_val.detach()),
                        "eval/standard_loss": float(raw_loss.detach()),
                        "eval/total_loss": float(total.detach())
                    })
            
            # Store for logging (optional)
            if hasattr(outputs, '__dict__'):
                outputs.dispersion_loss = disp_val.detach()
        else:
            # Track gradient accumulation step even when dispersion is disabled
            if model.training:
                self._current_accumulation_step = (self._current_accumulation_step + 1) % self.args.gradient_accumulation_steps
                is_last_accumulation_step = (self._current_accumulation_step == 0)
            else:
                is_last_accumulation_step = True  # Always log during eval
            
            # When dispersion is disabled, provide basic logging at proper intervals
            # For training: only log on last accumulation step and at logging intervals
            should_log = (not model.training or 
                         (is_last_accumulation_step and
                          hasattr(self.state, 'global_step') and 
                          self.state.global_step > 0 and 
                          self.state.global_step % self.args.logging_steps == 0 and
                          self.state.global_step != self._last_logged_step))
            
            if should_log and model.training:
                self._last_logged_step = self.state.global_step  # Mark as logged
                self.log({"loss": float(raw_loss.detach())})
            # For eval, let Transformers handle eval/loss automatically

        # import pdb; pdb.set_trace()
        # print(f"DEBUG: Final total loss: {float(total)}")
        
        if return_outputs:
            return total, outputs
        else:
            return total

    @staticmethod
    def _seq_token_features(hidden: torch.Tensor, labels: torch.Tensor,
                            max_tokens_per_seq: int = 512):
        '''
        hidden: [B, seq_len, feature], labels: [B, seq_len]
        Returns a list of per-sequence token features:
            [ [n1, feature], [n2, feature], ... ]   (one tensor per example)
        Only non-ignored tokens are kept per sequence, and each sequence
        is capped to at most max_tokens_per_seq tokens (uniform subsample) to avoid OOM.
        '''
        B, T, F = hidden.shape
        feats = []
        for b in range(B):
            mask_b = labels[b] != -100
            if mask_b.any():
                zb = hidden[b][mask_b]  # [n_b, F]
                n = zb.size(0)
                if n > max_tokens_per_seq:
                    idx = torch.randperm(n, device=zb.device)[:max_tokens_per_seq]
                    zb = zb.index_select(0, idx)
                feats.append(zb)
        if not feats:
            # If no valid tokens anywhere; return a single dummy so loss=0
            feats = [hidden.new_zeros((1, F))]
        return feats

    def _dispersion_from_hidden_states(self, hidden_states, labels) -> torch.Tensor:
        '''
        hidden_states: tuple of [B, seq_len, feature] with length (layer + 1) for most HF models.
        labels: [B, seq_len]

        Computes dispersion **within each sequence** (no cross-sequence pairs),
        averages over sequences, and if dispersion_loc == "all", also averages over layers.
        '''
        def per_layer_loss(h):
            zs = self._seq_token_features(h, labels)
            vals = []
            for z in zs:
                # z: [n_seq_tokens, feature]; each sequence handled independently
                if z.size(0) >= 2:
                    vals.append(self._disp_fn(z))
            if not vals:
                return labels.new_zeros(())
            return torch.stack(vals).mean()

        if self.disp_loc == "last":
            return per_layer_loss(hidden_states[-1])

        # "all": average across all transformer layers (skip embedding layer at index 0)
        layer_vals = []
        for h in hidden_states[1:]:
            layer_vals.append(per_layer_loss(h))
        return torch.stack(layer_vals).mean() if layer_vals else labels.new_zeros(())