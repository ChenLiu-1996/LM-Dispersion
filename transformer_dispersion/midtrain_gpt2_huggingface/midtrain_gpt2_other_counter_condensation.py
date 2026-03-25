"""
Mid-train GPT-2 with anti-condensation baselines from the reviewer discussion:
  --perturb_random_noise: add Gaussian noise to token embeddings on each train forward.
  --perturb_active_forgetting: reinitialize token embeddings every K steps (Chen et al., NeurIPS 2023).

Mutually exclusive with dispersion loss; training objective is standard CE only.
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import_dir = "/".join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, os.path.join(import_dir))

import midtrain_gpt2 as mt


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _reinit_token_embeddings_synced(model, std: float):
    """Chen et al.-style reset; identical weights on all DDP ranks via broadcast."""
    m = _unwrap_model(model)
    emb = m.get_input_embeddings()
    w = emb.weight.data
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            torch.nn.init.normal_(w, mean=0.0, std=std)
        dist.broadcast(w, src=0)
    else:
        torch.nn.init.normal_(w, mean=0.0, std=std)


class ActiveForgettingCallback(TrainerCallback):
    def __init__(self, every_k_steps: int, log_path: Optional[str]):
        self.every_k_steps = every_k_steps
        self.log_path = log_path

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_k_steps <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.every_k_steps != 0:
            return control
        model = kwargs["model"]
        if not model.training:
            return control
        std = getattr(_unwrap_model(model).config, "initializer_range", 0.02)
        _reinit_token_embeddings_synced(model, std=std)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0:
            mt.log(
                f"[active_forgetting] Reinitialized token embeddings at step {state.global_step} (every_k={self.every_k_steps})",
                filepath=self.log_path,
            )
        return control


class PerturbationTrainer(Trainer):
    """CE only; optional additive noise on input embeddings during training."""

    def __init__(self, *args, embed_noise_std: float = 0.0, use_embed_noise: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_noise_std = embed_noise_std
        self.use_embed_noise = use_embed_noise
        self.loss_fn = mt.CausalLMLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        train_noise = self.use_embed_noise and model.training and self.embed_noise_std > 0

        if train_noise:
            input_ids = inputs["input_ids"]
            m = _unwrap_model(model)
            emb_layer = m.get_input_embeddings()
            embeds = emb_layer(input_ids)

            noise = torch.randn(embeds.shape, device=embeds.device, dtype=embeds.dtype)
            inputs_embeds = embeds + noise * self.embed_noise_std

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

    training_args = TrainingArguments(
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_noise = args.perturb_random_noise
    trainer = PerturbationTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        embed_noise_std=args.embed_noise_std,
        use_embed_noise=use_noise,
    )

    perturb_tag = "noise" if use_noise else "active_forget"
    mt.log("=== Mid-training setup (counter-condensation baselines) ===", filepath=args.log_path)
    mt.log(f"Perturbation mode: {perturb_tag}", filepath=args.log_path)
    if use_noise:
        mt.log(f"Embedding noise std: {args.embed_noise_std}", filepath=args.log_path)
    else:
        mt.log(f"Active forgetting every {args.active_forget_every_k_steps} optimizer steps", filepath=args.log_path)
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

    if args.perturb_active_forgetting:
        trainer.add_callback(
            ActiveForgettingCallback(
                every_k_steps=args.active_forget_every_k_steps,
                log_path=args.log_path,
            )
        )

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
    ap = argparse.ArgumentParser(description="Mid-train GPT-2: embedding noise or active forgetting baselines.")
    ap.add_argument("--model_name", type=str, default="gpt2", help="HF model id.")
    ap.add_argument("--cache_dir", type=str, default="./.cache/", help="HF cache dir.")
    ap.add_argument("--dataset_name", type=str, default="Salesforce/wikitext", help="HF dataset id.")
    ap.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1", help="Dataset config name.")
    ap.add_argument("--hf_token", type=str, default=None, help="HF token if needed.")
    ap.add_argument("--lr", type=float, default=5e-5, help="Base LR (scaled by sqrt(world_size)).")
    ap.add_argument("--train_tokens", type=int, required=True, help="Total training token budget.")
    ap.add_argument("--perturb_random_noise", action="store_true", help="Gaussian noise on token embeddings (train).")
    ap.add_argument("--perturb_active_forgetting", action="store_true", help="Reinit embeddings every K steps (Chen et al.).")
    ap.add_argument("--embed_noise_std", type=float, default=0.02, help="Noise std if --perturb_random_noise.")
    ap.add_argument("--active_forget_every_k_steps", type=int, default=1000, help="Reinit period if --perturb_active_forgetting.")
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

    n_modes = int(args.perturb_random_noise) + int(args.perturb_active_forgetting)
    if n_modes != 1:
        ap.error("Specify exactly one of --perturb_random_noise and --perturb_active_forgetting.")

    ds = "-".join(args.dataset_name.split("/"))
    if args.perturb_random_noise:
        mid = f"ccnoise-{args.embed_noise_std}"
    else:
        mid = f"ccforget-{args.active_forget_every_k_steps}"
    args.output_dir = (
        f"./results/midtrain_{args.model_name}_{ds}_lr-{args.lr}_token-{args.train_tokens}_"
        f"{mid}_fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_seed-{args.seed}"
    )
    args.log_path = os.path.join(args.output_dir, "log.txt")
    main(args)
