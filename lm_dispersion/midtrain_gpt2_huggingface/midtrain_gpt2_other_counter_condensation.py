"""
Mid-train GPT-2 with either noisy embedding or active forgetting.
Mutually exclusive with dispersion loss; training objective is standard CE only.
"""

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

try:
    from transformers.pytorch_utils import get_parameter_names
except ImportError:
    from transformers.trainer_pt_utils import get_parameter_names

import_dir = "/".join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, os.path.join(import_dir))

import midtrain_gpt2 as mt


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _collect_input_embedding_trainable_params(model) -> List[torch.nn.Parameter]:
    """Input token embeddings only (Chen et al.); lm_head / output weights are not reset."""
    m = _unwrap_model(model)
    out: List[torch.nn.Parameter] = []
    seen = set()
    for p in m.get_input_embeddings().parameters():
        if p.requires_grad:
            oid = id(p)
            if oid not in seen:
                seen.add(oid)
                out.append(p)
    return out


def _reinit_input_token_embeddings_synced(model, pad_token_id: Optional[int]):
    """N(0, 0.02) reinit for input embedding weights only; zero pad row (Chen et al.)."""
    m = _unwrap_model(model)
    emb = m.get_input_embeddings()
    w = emb.weight.data
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            torch.nn.init.normal_(w, mean=0.0, std=0.02)
            pidx = getattr(emb, "padding_idx", None)
            if pidx is not None:
                w[pidx].zero_()
            elif pad_token_id is not None and 0 <= int(pad_token_id) < w.size(0):
                w[int(pad_token_id)].zero_()
        dist.broadcast(w, src=0)
    else:
        torch.nn.init.normal_(w, mean=0.0, std=0.02)
        pidx = getattr(emb, "padding_idx", None)
        if pidx is not None:
            w[pidx].zero_()
        elif pad_token_id is not None and 0 <= int(pad_token_id) < w.size(0):
            w[int(pad_token_id)].zero_()


def _clear_adam_states_for_params(optimizer, params: Sequence[torch.nn.Parameter]) -> None:
    """Match language-model-plasticity AdamEF: zero momentum / variance for reset embeddings."""
    want = {id(p) for p in params}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) not in want:
                continue
            st = optimizer.state.get(p)
            if not st:
                continue
            if "exp_avg" in st and st["exp_avg"] is not None:
                st["exp_avg"].zero_()
            if "exp_avg_sq" in st and st["exp_avg_sq"] is not None:
                st["exp_avg_sq"].zero_()
            if "max_exp_avg_sq" in st and st["max_exp_avg_sq"] is not None:
                st["max_exp_avg_sq"].zero_()
            if "step" in st:
                s = st["step"]
                if isinstance(s, torch.Tensor):
                    s.zero_()
                else:
                    st["step"] = 0


def _linear_warmup_decay_mult(current_step: int, num_training_steps: int, warmup_ratio: float) -> float:
    """LR multiplier matching language-model-plasticity Fairseq polynomial_decay with power=1 (linear post-warmup).

    Same curve as HF `get_linear_schedule_with_warmup` / polynomial schedule with power=1 and lr_end=0.
    """
    if num_training_steps <= 0:
        return 1.0
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0,
        float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
    )


def _plasticity_emb_schedule_step(body_num_updates: int, max_update: int, K: int) -> int:
    """Match language-model-plasticity fairseq `Trainer.lr_step_update` for adamef (embedding index).

    See https://github.com/facebookresearch/language-model-plasticity - `language/fairseq/trainer.py`:
    `speed = tot // K`, `emb_num_updates = (body_num_updates * speed) % tot`.
    """
    if max_update <= 0 or K <= 0:
        return 0
    speed = max_update // K
    return (body_num_updates * speed) % max_update


def _cosine_warmup_mult(
    current_step: int, num_training_steps: int, warmup_ratio: float, num_cycles: float = 0.5
) -> float:
    """Same multiplier shape as HF ``get_cosine_schedule_with_warmup`` (body LR when active forgetting)."""
    if num_training_steps <= 0:
        return 1.0
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


try:
    from torch.optim.lr_scheduler import LRScheduler as _TorchLRSchedulerBase
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as _TorchLRSchedulerBase


class ActiveForgettingLRScheduler(_TorchLRSchedulerBase):
    """Absolute per-group learning rates for active forgetting (plasticity-style emb schedule + HF-shaped body cosine)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        every_k: int,
        body_base_lr: float,
        body_warmup_ratio: float,
        emb_peak_lr: float,
        emb_warmup_ratio: float,
        last_epoch: int = -1,
    ):
        self.af_max_steps = int(max_steps)
        self.af_every_k = max(1, int(every_k))
        self.af_body_base_lr = float(body_base_lr)
        self.af_body_warmup_ratio = float(body_warmup_ratio)
        self.af_emb_peak_lr = float(emb_peak_lr)
        self.af_emb_warmup_ratio = float(emb_warmup_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Body: same step index as HF ``LambdaLR`` / ``get_cosine_schedule_with_warmup``. Embedding plasticity:
        # ``last_epoch + 1`` = completed optimizer steps when this lr is applied on the next step.
        le = self.last_epoch
        T = self.af_max_steps
        K = self.af_every_k
        emb_g = le + 1
        out: List[float] = []
        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            if name.startswith("embedding"):
                ei = _plasticity_emb_schedule_step(emb_g, T, K)
                m = _linear_warmup_decay_mult(
                    min(ei, max(0, T - 1)), T, self.af_emb_warmup_ratio
                )
                out.append(self.af_emb_peak_lr * m)
            else:
                m = _cosine_warmup_mult(le, T, self.af_body_warmup_ratio)
                out.append(self.af_body_base_lr * m)
        return out


class ActiveForgettingCallback(TrainerCallback):
    """Periodic input embedding reinit + Adam state clear (Chen et al.). LRs come from ``ActiveForgettingLRScheduler``.

    References: Chen et al. 2023; https://github.com/facebookresearch/language-model-plasticity
    """

    def __init__(self, trainer: "PerturbationTrainer"):
        self.trainer = trainer
        self.every_k = trainer.active_forget_every_k
        self.log_path = trainer.af_log_path
        self.pad_token_id = trainer.af_pad_token_id

    def on_step_end(self, args, state, control, **kwargs):
        """After completing step gs where gs > 0 and gs % K == 0 (num_updates % K == 0 in official code)."""
        if self.every_k <= 0:
            return control
        K = self.every_k
        gs = state.global_step
        if gs <= 0 or (gs % K) != 0:
            return control
        model = kwargs.get("model", self.trainer.model)
        opt = kwargs.get("optimizer", self.trainer.optimizer)
        if model is None or opt is None or not model.training:
            return control
        _reinit_input_token_embeddings_synced(model, pad_token_id=self.pad_token_id)
        emb_params = _collect_input_embedding_trainable_params(model)
        _clear_adam_states_for_params(opt, emb_params)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            mt.log(
                f"[active_forgetting] Reset embeddings + cleared Adam states after step {gs} "
                f"(gs % {K} == 0; Chen et al. 2023 / language-model-plasticity train.py timing)",
                filepath=self.log_path,
            )
        return control


class PerturbationTrainer(Trainer):
    """CE only; optional NEFTune-style noise on input embeddings during training."""

    def __init__(
        self,
        *args,
        neftune_alpha: float = 0.0,
        use_embed_noise: bool = False,
        active_forgetting: bool = False,
        active_forget_every_k: int = 1000,
        af_log_path: Optional[str] = None,
        af_body_base_lr: float = 5e-5,
        af_body_warmup_ratio: float = 0.2,
        af_emb_peak_lr: float = 1e-4,
        af_emb_warmup_ratio: float = 0.08,
        af_pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.neftune_alpha = neftune_alpha
        self.use_embed_noise = use_embed_noise
        self.loss_fn = mt.CausalLMLoss()
        self.active_forgetting = active_forgetting
        self.active_forget_every_k = active_forget_every_k
        self.af_log_path = af_log_path
        self.af_body_base_lr = af_body_base_lr
        self.af_body_warmup_ratio = af_body_warmup_ratio
        self.af_emb_peak_lr = af_emb_peak_lr
        self.af_emb_warmup_ratio = af_emb_warmup_ratio
        self.af_pad_token_id = af_pad_token_id

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.active_forgetting:
            optimizer = optimizer or self.optimizer
            K = self.active_forget_every_k
            self.lr_scheduler = ActiveForgettingLRScheduler(
                optimizer,
                max_steps=int(num_training_steps),
                every_k=max(1, K),
                body_base_lr=self.af_body_base_lr,
                body_warmup_ratio=self.af_body_warmup_ratio,
                emb_peak_lr=self.af_emb_peak_lr,
                emb_warmup_ratio=self.af_emb_warmup_ratio,
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer=optimizer)

    def create_optimizer(self):
        if not self.active_forgetting:
            return super().create_optimizer()

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [n for n in decay_parameters if "bias" not in n]
        emb_ids = {id(p) for p in _collect_input_embedding_trainable_params(opt_model)}

        def pick(names_in_decay: bool):
            rows = []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                in_decay = n in decay_parameters
                if in_decay != names_in_decay:
                    continue
                if id(p) in emb_ids:
                    rows.append(p)
            return rows

        def pick_body(names_in_decay: bool):
            rows = []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                in_decay = n in decay_parameters
                if in_decay != names_in_decay:
                    continue
                if id(p) in emb_ids:
                    continue
                rows.append(p)
            return rows

        optimizer_grouped_parameters = []
        bd, ed = pick_body(True), pick(True)
        bnd, end = pick_body(False), pick(False)
        # Match HF ``Trainer`` + cosine warmup: initial param group lr is full peak; scheduler updates after
        # each step. mult(0)==0 as initial lr made the first optimizer step a no-op vs ``midtrain_gpt2.py``.
        lr_body = self.af_body_base_lr
        lr_emb = self.af_emb_peak_lr
        wd = self.args.weight_decay
        if bd:
            optimizer_grouped_parameters.append(
                {"params": bd, "weight_decay": wd, "lr": lr_body, "name": "body_decay"}
            )
        if bnd:
            optimizer_grouped_parameters.append(
                {"params": bnd, "weight_decay": 0.0, "lr": lr_body, "name": "body_nd"}
            )
        if ed:
            optimizer_grouped_parameters.append(
                {"params": ed, "weight_decay": wd, "lr": lr_emb, "name": "embedding_decay"}
            )
        if end:
            optimizer_grouped_parameters.append(
                {"params": end, "weight_decay": 0.0, "lr": lr_emb, "name": "embedding_nd"}
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        train_noise = self.use_embed_noise and model.training and self.neftune_alpha > 0

        if train_noise:
            input_ids = inputs["input_ids"]
            m = _unwrap_model(model)
            emb_layer = m.get_input_embeddings()
            embeds = emb_layer(input_ids)

            # NEFTune: uniform noise, mag_norm = alpha / sqrt(|positions| * |hidden|) for shape (B, S, H).
            d = float(embeds.size(1) * embeds.size(2))
            mag_norm = self.neftune_alpha / math.sqrt(d)
            noise = torch.empty_like(embeds).uniform_(-mag_norm, mag_norm)
            inputs_embeds = embeds + noise

            fwd = dict(inputs)
            fwd.pop("input_ids", None)
            fwd["inputs_embeds"] = inputs_embeds
            outputs = model(**fwd)
        else:
            outputs = model(**inputs)

        logits = outputs.logits
        loss = self.loss_fn(logits, labels)

        if (
            model.training
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            self.log(
                {
                    "train/default_loss": loss.detach().item(),
                    "train/total_loss": loss.detach().item(),
                }
            )

        return (loss, outputs) if return_outputs else loss


def main(args):
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token, cache_dir=args.cache_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(args.model_name, token=args.hf_token, cache_dir=args.cache_dir)
    if hasattr(config, "loss_type"):
        delattr(config, "loss_type")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, token=args.hf_token, cache_dir=args.cache_dir
    )
    model.gradient_checkpointing_enable()

    max_position_embeddings = getattr(model.config, "max_position_embeddings")
    context_len = 1024
    max_gen_tokens = 256
    assert max_gen_tokens <= context_len and context_len <= max_position_embeddings
    tokenizer.model_max_length = context_len

    lm_train, lm_val = mt.make_splits(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        tokenizer=tokenizer,
        context_len=context_len,
        seed=args.seed,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = args.per_device_train_batch_size * context_len * args.gradient_accumulation_steps * world_size
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step computed as 0; check batch size/accumulation/context length.")
    max_steps = math.ceil(args.train_tokens / tokens_per_step)
    mt.log(f"Training for {args.train_tokens} tokens, which is {max_steps} steps.", filepath=args.log_path)
    log_every_n_steps = max_steps // args.num_ckpt + 1

    fp16, bf16 = mt.compute_precision_flags()
    learning_rate = args.lr * math.sqrt(world_size)

    training_args_kw = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        warmup_ratio=0.2,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        log_level="info",
        logging_steps=max(1, max_steps // 20),
        log_on_each_node=False,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=True,
    )
    training_args = TrainingArguments(**training_args_kw)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_noise = args.noisy_embedding
    trainer = PerturbationTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        neftune_alpha=args.neftune_alpha,
        use_embed_noise=use_noise,
        active_forgetting=args.active_forgetting,
        active_forget_every_k=args.active_forget_every_k_steps,
        af_log_path=args.log_path,
        af_body_base_lr=learning_rate,
        af_body_warmup_ratio=0.2,
        af_emb_peak_lr=args.af_emb_peak_lr,
        af_emb_warmup_ratio=args.af_emb_warmup_ratio,
        af_pad_token_id=int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
    )

    perturb_tag = "noise" if use_noise else "active_forget"
    mt.log("=== Mid-training setup (NEFTune noise or active forgetting) ===", filepath=args.log_path)
    mt.log(f"Perturbation mode: {perturb_tag}", filepath=args.log_path)
    if use_noise:
        mt.log(
            f"NEFTune embedding noise: alpha={args.neftune_alpha} "
            f"(uniform in [-mag_norm, mag_norm], mag_norm=alpha/sqrt(S*H), S=seq len, H=hidden dim)",
            filepath=args.log_path,
        )
    else:
        mt.log(
            f"Active forgetting (Chen et al. 2023; https://arxiv.org/pdf/2307.01163 ; "
            f"code https://github.com/facebookresearch/language-model-plasticity ): "
            f"K={args.active_forget_every_k_steps}, input emb only reinit N(0,0.02); "
            f"body LR cosine+warmup (peak scaled lr={learning_rate:.2e}); "
            f"emb LR peak={args.af_emb_peak_lr:.2e}, linear warmup/decay, idx (g*floor(T/K))%T; "
            f"TrainingArgs cosine warmup 0.2 max_grad_norm 1.0 (global); "
            f"reset+Adam clear emb when gs>0 and gs%K==0.",
            filepath=args.log_path,
        )
    mt.log(f"Model: {args.model_name}", filepath=args.log_path)
    mt.log(str(model.config), filepath=args.log_path)
    mt.log(f"Dataset: {args.dataset_name} ({args.dataset_config})", filepath=args.log_path)
    mt.log(f"Context length: {context_len}", filepath=args.log_path)
    mt.log(f"Max gen tokens: {max_gen_tokens}", filepath=args.log_path)
    mt.log(
        f"Per-device batch: {args.per_device_train_batch_size} | Grad accum: {args.gradient_accumulation_steps} | World size: {world_size}",
        filepath=args.log_path,
    )
    mt.log(f"Token budget: {args.train_tokens} | Tokens/step: {tokens_per_step} | Max steps: {max_steps}", filepath=args.log_path)
    mt.log(f"Precision: {'bf16' if bf16 else ('fp16' if fp16 else 'fp32')}", filepath=args.log_path)

    zeroshot_tasks = [
        "anli",
        "hellaswag",
        "lambada",
        "openbookqa",
        "paloma_wikitext_103",
        "piqa",
        "truthfulqa_mc2",
        "winogrande",
    ]
    fewshot_tasks = [
        "arc_challenge",
        "arc_easy",
        "mmlu",
        "medmcqa",
    ]
    lm_eval_callback = mt.LMEvalCallback(
        tokenizer,
        zeroshot_tasks,
        fewshot_tasks,
        log_path=args.log_path,
        max_gen_tokens=max_gen_tokens,
        num_fewshot=args.num_fewshot,
        max_eval_samples=args.max_eval_samples,
        every_n_steps=log_every_n_steps if args.train_tokens > 0 else None,
        eval_at_begin=args.eval_at_begin,
        eval_at_end=args.train_tokens > 0,
        save_on_eval=not args.no_save_model,
    )
    trainer.add_callback(lm_eval_callback)

    if args.active_forgetting:
        trainer.add_callback(ActiveForgettingCallback(trainer))

    train_t0 = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_t0
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        eval_sec = lm_eval_callback.eval_wall_seconds
        train_wo_eval = max(0.0, train_elapsed - eval_sec)
        mt.log(
            f"Training wall time: {train_elapsed:.2f}s ({train_elapsed / 60:.2f} min, {train_elapsed / 3600:.4f} h)",
            filepath=args.log_path,
        )
        mt.log(
            f"LMEvalCallback wall time (rank 0 eval work): {eval_sec:.2f}s ({eval_sec / 60:.2f} min, {eval_sec / 3600:.4f} h)",
            filepath=args.log_path,
        )
        mt.log(
            f"Training wall time minus LMEvalCallback: {train_wo_eval:.2f}s ({train_wo_eval / 60:.2f} min, {train_wo_eval / 3600:.4f} h)",
            filepath=args.log_path,
        )

    mt.log(f"Done. Saved to {args.output_dir}", filepath=args.log_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Mid-train GPT-2 with NEFTune noise or active forgetting (mutually exclusive)."
    )
    ap.add_argument("--model_name", type=str, default="gpt2", help="HF model id.")
    ap.add_argument("--cache_dir", type=str, default="./.cache/", help="HF cache dir.")
    ap.add_argument("--dataset_name", type=str, default="Salesforce/wikitext", help="HF dataset id.")
    ap.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="Dataset config name.")
    ap.add_argument("--hf_token", type=str, default=None, help="HF token if needed.")
    ap.add_argument("--lr", type=float, default=5e-5, help="Base LR (scaled by sqrt(world_size)).")
    ap.add_argument("--train_tokens", type=int, required=True, help="Total training token budget.")
    ap.add_argument(
        "--noisy_embedding",
        action="store_true",
        help="NEFTune-style uniform noise on token embeddings during training (train only).",
    )
    ap.add_argument("--active_forgetting", action="store_true", help="Active forgetting (Chen et al. 2023).")
    ap.add_argument(
        "--neftune_alpha",
        type=float,
        default=1.0,
        help="NEFTune noise_alpha if --noisy_embedding (bound = alpha/sqrt(S H) on token embeddings).",
    )
    ap.add_argument("--active_forget_every_k_steps", type=int, default=1000, help="K if --active_forgetting.")
    ap.add_argument(
        "--af-emb-peak-lr",
        type=float,
        default=5e-5,
        help="Peak LR for embedding param groups under active forgetting.",
    )
    ap.add_argument(
        "--af-emb-warmup-ratio",
        type=float,
        default=0.08,
        help="Embedding linear schedule: warmup fraction of max_steps (~10K/125K in Chen et al.).",
    )
    ap.add_argument("--num_fewshot", type=int, default=1, help="lm-eval fewshot count.")
    ap.add_argument("--max_eval_samples", type=int, default=500, help="lm-eval limit per task.")
    ap.add_argument("--num_ckpt", type=int, default=5, help="Eval/save intervals from token budget.")
    ap.add_argument("--no_save_model", action="store_true", help="Skip saving eval checkpoints.")
    ap.add_argument("--num_workers", type=int, default=8, help="Dataloader workers.")
    ap.add_argument("--per_device_train_batch_size", type=int, default=16, help="Train batch size per device.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Grad accumulation steps.")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed.")
    ap.add_argument("--eval_at_begin", action="store_true", help="Run lm-eval at step 0.")

    args = ap.parse_args()

    n_modes = int(args.noisy_embedding) + int(args.active_forgetting)
    if n_modes != 1:
        ap.error("Specify exactly one of --noisy_embedding and --active_forgetting.")

    ds = "-".join(args.dataset_name.split("/"))
    if args.noisy_embedding:
        mid = f"ccnoise-{args.neftune_alpha}"
    else:
        mid = f"ccforget-{args.active_forget_every_k_steps}"
    args.output_dir = (
        f"./results/midtrain_{args.model_name}_{ds}_lr-{args.lr}_token-{args.train_tokens}_"
        f"{mid}_fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_seed-{args.seed}"
    )
    args.log_path = os.path.join(args.output_dir, "log.txt")
    main(args)
