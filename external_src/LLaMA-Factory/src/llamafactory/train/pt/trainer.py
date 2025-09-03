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
Sep 2, 2025: adujusted to the updated version of the dispersion loss.
"""

from types import MethodType
from typing import TYPE_CHECKING, Optional, List

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .dispersion import DispersionLoss


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments




class CausalLMLoss(torch.nn.Module):
    """
    This is a fallback for non-llama models that does not compute the loss automatically.
    Note: this has not been tested yet.
    """
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
        self.disp_eval = finetuning_args.dispersion_eval

        if self.use_disp:
            variant = finetuning_args.dispersion.lower()
            self.disp_loss_fn = DispersionLoss(variant=variant,
                                          tau_l2=finetuning_args.tau_infonce_l2,
                                          tau_cos=finetuning_args.tau_infonce_cos)
        else:
            self.disp_loss_fn = None

        self.loss_fn = CausalLMLoss()
        
        # Track logging to avoid duplicate logs per global step
        self._last_logged_step = -1
        self._current_accumulation_step = 0

        # import pdb; pdb.set_trace()

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
            # Debug: Check for NaN in hidden states before dispersion computation
            for i, h in enumerate(outputs.hidden_states):
                if torch.isnan(h).any():
                    raise ValueError(f"NaN detected in hidden_states[{i}] before dispersion computation")

            disp_val = self.disperse_hidden_states(outputs.hidden_states)
            
            # Debug: Check if dispersion computation produced NaN
            if torch.isnan(disp_val).any():
                raise ValueError(f"NaN detected in dispersion loss: {disp_val}")
                # import pdb; pdb.set_trace()
            
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
                    "dispersion_loss": float(disp_val.detach()),
                    "standard_loss": float(raw_loss.detach()),
                    "total_loss": float(total.detach())
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

    def disperse_hidden_states(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        '''
        Computes dispersion for last layer or averages across all layers (excluding emb layer at index 0),
        with embeddings rearranged to [num_samples, sequence_length].

        hidden_states: tuple of tensors, each [B, seq_len, feature_dim]
        '''
        assert self.disp_loss_fn is not None, "disp_loss_fn is None"
        if self.disp_loc == "last":
            return self.disp_loss_fn(hidden_states[-1])

        # Average across transformer layers (skipping embedding layer)
        loss_values = []
        assert len(hidden_states) > 1
        for idx, h in enumerate(hidden_states):
            if idx == 0:
                # skipping embedding layer
                continue
            loss_values.append(self.disp_loss_fn(h))
        return torch.stack(loss_values).mean()
