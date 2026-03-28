"""
Toy from-scratch pre-training on a GPT-2-style (decoder-only) transformer.
Same data / Trainer / optional dispersion path as pretrain_toy_depth / midtrain_gpt2, except:

  - model weights are randomly initialized (no pretrained checkpoint);
  - MLP (FFN) hidden width is controlled via --ffn_intermediate (GPT-2 config field n_inner).
    Default n_inner in HF is 4 * n_embd; here you set it explicitly.
  - n_embd and per-layer residual shapes stay fixed, so hidden_states remain (B, T, n_embd).
  - optional --num_layers overrides the template depth; otherwise the template's n_layer is kept.

Sweeping --ffn_intermediate with n_layer and n_embd fixed isolates "parameter count in the FFN"
from depth and from representation dimension.
"""
from typing import List, Tuple
import os
import gc
import sys
import json
import math
import argparse
import time
import torch
import torch.distributed as dist
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

import_dir = "/".join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, os.path.join(import_dir))
from dispersion import DispersionLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(s, filepath=None, to_console=True):
    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, "w+") as o:
                o.write(s + "\n")
        else:
            with open(filepath, "a+") as o:
                o.write(s + "\n")


def filter_non_empty(example):
    txt = example.get("text", "")
    return bool(txt and txt.strip())


def tokenize_batch(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, context_len):
    concatenated = {}
    for k in examples.keys():
        all_vals = []
        for seq in examples[k]:
            all_vals.extend(seq)
        concatenated[k] = all_vals
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // context_len) * context_len
    result = {k: [t[i : i + context_len] for i in range(0, total_len, context_len)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result


def compute_precision_flags():
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.is_bf16_supported():
        return False, True
    return True, False


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    if hasattr(model, "num_parameters"):
        return (
            int(model.num_parameters(only_trainable=False)),
            int(model.num_parameters(only_trainable=True)),
        )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_splits(dataset_name, dataset_config, cache_dir, hf_token, tokenizer, context_len, seed):
    if dataset_config is None or str(dataset_config).strip() == "":
        ds = load_dataset(dataset_name, streaming=False, token=hf_token, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_name, dataset_config, streaming=False, token=hf_token, cache_dir=cache_dir)

    if "train" in ds:
        ds_train = ds["train"]
    else:
        parts = [s for s in ("validation", "test") if s in ds]
        assert parts, "No 'train' split and no alternative splits available."
        ds_train = concatenate_datasets([ds[s] for s in parts])

    ds_train = ds_train.filter(filter_non_empty)

    if "validation" in ds:
        ds_val = ds["validation"].filter(filter_non_empty)
    elif "test" in ds:
        ds_val = ds["test"].filter(filter_non_empty)
    else:
        ds_val = ds_train

    tok_train = ds_train.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=[c for c in ds_train.column_names if c != "text"],
        desc="Tokenizing train",
    )
    tok_val = ds_val.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=[c for c in ds_val.column_names if c != "text"],
        desc="Tokenizing val",
    )

    lm_train = tok_train.map(
        lambda b: group_texts(b, context_len),
        batched=True,
        desc=f"Grouping train into blocks of {context_len}",
    ).shuffle(seed=seed)

    lm_val = tok_val.map(
        lambda b: group_texts(b, context_len),
        batched=True,
        desc=f"Grouping val into blocks of {context_len}",
    )

    return lm_train, lm_val


def save_pretrained_eval_checkpoint(
    args,
    state,
    model,
    tokenizer,
    stage: str,
    log_path,
    msg_prefix: str = "[LMEval]",
    sync_ddp: bool = True,
):
    """Same eval_ckpt_* path naming as LMEvalCallback.save_on_eval. If sync_ddp, wrap in barriers (for callbacks that only call this)."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    if sync_ddp and ddp:
        dist.barrier()
    try:
        if local_rank == 0:
            ckpt_dir = os.path.join(
                args.output_dir,
                f"eval_ckpt_{stage or 'interval'}_step{state.global_step}",
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            save_st = getattr(args, "save_safetensors", True)
            if hasattr(model, "module"):
                model.module.save_pretrained(ckpt_dir, save_safetensors=save_st)
            else:
                model.save_pretrained(ckpt_dir, save_safetensors=save_st)
            tokenizer.save_pretrained(ckpt_dir)
            log(f"{msg_prefix} Weights saved to {ckpt_dir}", filepath=log_path)
    finally:
        if sync_ddp and ddp:
            dist.barrier()


class ModelSaveCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        log_path,
        every_n_steps=None,
        save_at_begin=True,
        save_at_end=True,
    ):
        self.tok = tokenizer
        self.log_path = log_path
        self.every_n_steps = every_n_steps
        self.save_at_begin = save_at_begin
        self.save_at_end = save_at_end
        self.has_run_begin = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.save_at_begin and not self.has_run_begin:
            model = kwargs["model"]
            save_pretrained_eval_checkpoint(
                args,
                state,
                model,
                self.tok,
                "begin",
                self.log_path,
                msg_prefix="[skip_eval]",
                sync_ddp=True,
            )
            self.has_run_begin = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps is None:
            return
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        model = kwargs["model"]
        save_pretrained_eval_checkpoint(
            args,
            state,
            model,
            self.tok,
            "interval",
            self.log_path,
            msg_prefix="[skip_eval]",
            sync_ddp=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        if self.save_at_end:
            model = kwargs["model"]
            save_pretrained_eval_checkpoint(
                args,
                state,
                model,
                self.tok,
                "end",
                self.log_path,
                msg_prefix="[skip_eval]",
                sync_ddp=True,
            )


class LMEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        zeroshot_tasks,
        fewshot_tasks,
        log_path,
        max_gen_tokens,
        num_fewshot,
        max_eval_samples=None,
        eval_at_begin=True,
        eval_at_end=True,
        every_n_steps=None,
        save_on_eval=True,
    ):
        self.tok = tokenizer
        self.zeroshot_tasks = zeroshot_tasks
        self.fewshot_tasks = fewshot_tasks
        self.log_path = log_path
        self.max_gen_tokens = max_gen_tokens
        self.num_fewshot = num_fewshot
        self.max_eval_samples = max_eval_samples
        self.eval_at_begin = eval_at_begin
        self.eval_at_end = eval_at_end
        self.every_n_steps = every_n_steps
        self.save_on_eval = save_on_eval
        self.has_run_begin = False
        self.eval_wall_seconds = 0.0

    def _run_evaluation(self, args, state, model, stage=""):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size > 1 and dist.is_available() and dist.is_initialized()

        if ddp:
            dist.barrier()

        try:
            if local_rank == 0:
                _lm_eval_t0 = time.perf_counter()
                try:
                    stage_str = f" ({stage})" if stage else ""
                    if torch.cuda.is_available() and world_size > 1:
                        device_str = "cuda"
                    elif torch.cuda.is_available():
                        device = next(model.parameters()).device
                        device_str = f"cuda:{device.index}" if device.index is not None else "cuda:0"
                    else:
                        device_str = "cpu"

                    log(
                        f"[LMEval] Running evaluation{stage_str} at step {state.global_step} "
                        f"(world_size={world_size}, device={device_str})...",
                        filepath=self.log_path,
                    )

                    eval_model = model.module if hasattr(model, "module") else model

                    was_training = eval_model.training
                    eval_model.eval()

                    try:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        gc.collect()

                        with torch.inference_mode():
                            wrapped_model = HFLM(pretrained=eval_model, tokenizer=self.tok, batch_size=1)

                            res_zeroshot = simple_evaluate(
                                model=wrapped_model,
                                tasks=self.zeroshot_tasks,
                                num_fewshot=0,
                                device=device_str,
                                limit=self.max_eval_samples,
                                gen_kwargs={"max_gen_toks": self.max_gen_tokens, "do_sample": False},
                                log_samples=False,
                                random_seed=args.seed,
                                numpy_random_seed=args.seed,
                                torch_random_seed=args.seed,
                                fewshot_random_seed=args.seed,
                            )

                            res_fewshot = simple_evaluate(
                                model=wrapped_model,
                                tasks=self.fewshot_tasks,
                                num_fewshot=self.num_fewshot,
                                device=device_str,
                                limit=self.max_eval_samples,
                                gen_kwargs={"max_gen_toks": self.max_gen_tokens, "do_sample": False},
                                log_samples=False,
                                random_seed=args.seed,
                                numpy_random_seed=args.seed,
                                torch_random_seed=args.seed,
                                fewshot_random_seed=args.seed,
                            )

                        assert "results" in res_zeroshot and "results" in res_fewshot
                        filename = (
                            f"lm_eval_{stage}_{state.global_step}.json"
                            if stage
                            else f"lm_eval_step{state.global_step}.json"
                        )
                        out = os.path.join(args.output_dir, filename)
                        merged_dict = {**res_zeroshot["results"], **res_fewshot["results"]}
                        with open(out, "w") as f:
                            json.dump({"results": merged_dict}, f, indent=2)
                        log(f"[LMEval] Results saved to {out}", filepath=self.log_path)

                        for task, metrics in merged_dict.items():
                            if isinstance(metrics, dict):
                                for metric_name, value in metrics.items():
                                    if isinstance(value, (int, float)):
                                        log(f"[LMEval] {task}.{metric_name}: {value:.4f}", filepath=self.log_path)

                        if self.save_on_eval:
                            save_pretrained_eval_checkpoint(
                                args, state, model, self.tok, stage, self.log_path, msg_prefix="[LMEval]", sync_ddp=False
                            )

                    except Exception as e:
                        log(
                            f"[LMEval] Error during evaluation{stage_str} at step {state.global_step}: {e}",
                            filepath=self.log_path,
                        )

                    finally:
                        if was_training:
                            eval_model.train()
                finally:
                    self.eval_wall_seconds += time.perf_counter() - _lm_eval_t0

        finally:
            if ddp:
                dist.barrier()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_at_begin and not self.has_run_begin:
            model = kwargs["model"]
            self._run_evaluation(args, state, model, "begin")
            self.has_run_begin = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps is None:
            return

        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return

        model = kwargs["model"]
        self._run_evaluation(args, state, model, "interval")

    def on_train_end(self, args, state, control, **kwargs):
        if self.eval_at_end:
            model = kwargs["model"]
            self._run_evaluation(args, state, model, "end")


class CausalLMLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss


class CustomLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_fn: torch.nn.Module,
        dispersion: str,
        dispersion_coeff: float,
        dispersion_loc: str,
        tau_l2: float,
        tau_cos: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

        self.use_disp = dispersion is not None and dispersion_coeff > 0.0
        self.disp_coeff = dispersion_coeff
        self.disp_loc = dispersion_loc

        if self.use_disp:
            variant = dispersion.lower()
            self.disp_loss_fn = DispersionLoss(variant=variant, tau_l2=tau_l2, tau_cos=tau_cos)

    def disperse_hidden_states(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        if self.disp_loc == "last":
            return self.disp_loss_fn(hidden_states[-1])

        loss_values = []
        assert len(hidden_states) > 1
        for idx, h in enumerate(hidden_states):
            if idx == 0:
                continue
            loss_values.append(self.disp_loss_fn(h))
        return torch.stack(loss_values).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        want_disp = self.use_disp and model.training
        outputs = model(**inputs, output_hidden_states=want_disp)
        logits = outputs.logits

        default_loss = self.loss_fn(logits, labels)

        if want_disp:
            disp_loss = self.disperse_hidden_states(outputs.hidden_states)
            total_loss = default_loss + self.disp_coeff * disp_loss
            outputs.dispersion_loss = disp_loss.detach()
        else:
            disp_loss = torch.zeros_like(default_loss)
            total_loss = default_loss

        if (
            model.training
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
        ):
            custom_losses = {
                "train/dispersion_loss": disp_loss.detach().item(),
                "train/default_loss": default_loss.detach().item(),
                "train/total_loss": total_loss.detach().item(),
            }
            self.log(custom_losses)

        return (total_loss, outputs) if return_outputs else total_loss


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

    if args.num_layers is not None:
        config.n_layer = int(args.num_layers)
    # FFN width: HuggingFace GPT-2 uses config.n_inner (None -> 4 * n_embd in the model).
    config.n_inner = int(args.ffn_intermediate)
    config.vocab_size = len(tokenizer)

    skip_tag = "skipeval_" if args.skip_eval else ""
    args.output_dir = (
        f"./results/pretrain_toy_ffn_{args.model_name}_nlayers-{config.n_layer}_ninner-{config.n_inner}_"
        f'{"-".join(args.dataset_name.split("/"))}_lr-{args.lr}_token-{args.train_tokens}_'
        f"disp-{args.dispersion}-{args.dispersion_coeff}-{args.dispersion_loc}-tau_cos-{args.tau_cos}-tau_l2-{args.tau_l2}_"
        f"fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_{skip_tag}seed-{args.seed}"
    )
    args.log_path = os.path.join(args.output_dir, "log.txt")

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_config(config)
    model.gradient_checkpointing_enable()

    npos = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    context_len = min(1024, int(npos))
    max_gen_tokens = min(256, context_len)
    assert max_gen_tokens <= context_len
    tokenizer.model_max_length = context_len

    lm_train, lm_val = make_splits(
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
    log(f"Training for {args.train_tokens} tokens, which is {max_steps} steps.", filepath=args.log_path)
    log_every_n_steps = max_steps // args.num_ckpt + 1

    fp16, bf16 = compute_precision_flags()
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

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        loss_fn=CausalLMLoss(),
        dispersion=args.dispersion,
        dispersion_coeff=args.dispersion_coeff,
        dispersion_loc=args.dispersion_loc,
        tau_cos=args.tau_cos,
        tau_l2=args.tau_l2,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    log("=== Toy pre-training (from scratch, GPT-2-style, FFN width sweep) ===", filepath=args.log_path)
    log(
        f"Template config from: {args.model_name} | n_layer = {config.n_layer} | "
        f"n_inner (FFN) = {config.n_inner} | n_embd = {config.n_embd}",
        filepath=args.log_path,
    )
    n_params, n_trainable = count_model_parameters(model)
    log(f"Parameters: {n_params:,} total, {n_trainable:,} trainable", filepath=args.log_path)
    log(str(model.config), filepath=args.log_path)
    log(f"Dataset: {args.dataset_name} ({args.dataset_config})", filepath=args.log_path)
    log(f"Context length: {context_len}", filepath=args.log_path)
    log(f"Max gen tokens: {max_gen_tokens}", filepath=args.log_path)
    log(
        f"Per-device batch: {args.per_device_train_batch_size} | Grad accum: {args.gradient_accumulation_steps} | "
        f"World size: {world_size}",
        filepath=args.log_path,
    )
    log(
        f"Token budget: {args.train_tokens} | Tokens/step: {tokens_per_step} | Max steps: {max_steps}",
        filepath=args.log_path,
    )
    log(f"Precision: {'bf16' if bf16 else ('fp16' if fp16 else 'fp32')}", filepath=args.log_path)

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

    lm_eval_callback = None
    if not args.skip_eval:
        lm_eval_callback = LMEvalCallback(
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
    else:
        log(
            "Skipping LMEvalCallback (--skip_eval); checkpoints at same steps as eval intervals.",
            filepath=args.log_path,
        )
        if not args.no_save_model:
            trainer.add_callback(
                ModelSaveCallback(
                    tokenizer,
                    log_path=args.log_path,
                    every_n_steps=log_every_n_steps if args.train_tokens > 0 else None,
                    save_at_begin=True,
                    save_at_end=args.train_tokens > 0,
                )
            )

    train_t0 = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_t0
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        eval_sec = 0.0 if lm_eval_callback is None else lm_eval_callback.eval_wall_seconds
        train_wo_eval = max(0.0, train_elapsed - eval_sec)
        log(
            f"Training wall time: {train_elapsed:.2f}s ({train_elapsed / 60:.2f} min, {train_elapsed / 3600:.4f} h)",
            filepath=args.log_path,
        )
        if lm_eval_callback is not None:
            log(
                f"LMEvalCallback wall time (rank 0 eval work): {eval_sec:.2f}s ({eval_sec / 60:.2f} min, {eval_sec / 3600:.4f} h)",
                filepath=args.log_path,
            )
            log(
                f"Training wall time minus LMEvalCallback: {train_wo_eval:.2f}s ({train_wo_eval / 60:.2f} min, {train_wo_eval / 3600:.4f} h)",
                filepath=args.log_path,
            )

    log(f"Done. Saved to {args.output_dir}", filepath=args.log_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Toy GPT-2-style pretrain from scratch; FFN width = --ffn_intermediate (config.n_inner)."
    )
    ap.add_argument("--model_name", type=str, default="gpt2", help="HF id: tokenizer + config template (no weights).")
    ap.add_argument("--num_layers", type=int, default=None, help="If set, override config.n_layer; otherwise default depth")
    ap.add_argument(
        "--ffn_intermediate",
        type=int,
        required=True,
        help="MLP hidden size (GPT-2 n_inner). Residual stream stays n_embd; shapes (B,T,n_embd) unchanged.",
    )

    ap.add_argument("--cache_dir", type=str, default="./.cache/", help="HF datasets/models cache.")
    ap.add_argument("--dataset_name", type=str, default="codelion/fineweb-edu-1B", help="HF dataset id.")
    ap.add_argument("--dataset_config", type=str, default="", help="Dataset config.")
    ap.add_argument("--hf_token", type=str, default=None, help="HF token for gated data.")
    ap.add_argument("--lr", type=float, default=1e-5, help="Base LR; scaled by sqrt(world_size).")
    ap.add_argument("--train_tokens", type=int, required=True, help="Token budget for training.")
    ap.add_argument("--dispersion", type=str, default=None, help="Dispersion variant; omit for CE only.")
    ap.add_argument("--dispersion_coeff", type=float, default=1.0, help="Dispersion loss weight.")
    ap.add_argument("--dispersion_loc", type=str, default="all", help="Where to apply dispersion.")
    ap.add_argument("--tau_l2", type=float, default=1.0, help="Dispersion tau (L2).")
    ap.add_argument("--tau_cos", type=float, default=1.0, help="Dispersion tau (cos).")
    ap.add_argument("--num_fewshot", type=int, default=1, help="lm-eval few-shot k.")
    ap.add_argument("--max_eval_samples", type=int, default=500, help="lm-eval sample limit per task.")
    ap.add_argument("--num_ckpt", type=int, default=2, help="Eval intervals = budget / num_ckpt.")
    ap.add_argument("--no_save_model", action="store_true", help="Do not save ckpt on lm-eval.")
    ap.add_argument("--num_workers", type=int, default=8, help="Dataloader workers.")
    ap.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed.")
    ap.add_argument("--eval_at_begin", action="store_true", help="lm-eval at step 0 (slow on random init).")
    ap.add_argument("--skip_eval", action="store_true", help="Do not register LMEvalCallback.")

    main(ap.parse_args())
