from typing import List
import os
import json
import tempfile
import math
import argparse
import torch
from einops import rearrange
from lm_eval import simple_evaluate
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from dispersion import DispersionLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console
    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
                o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def filter_non_empty(example):
    txt = example.get("text", "")
    return bool(txt and txt.strip())

def tokenize_batch(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length)

def group_texts(examples, block_size):
    concatenated = {}
    for k in examples.keys():
        all_vals = []
        for seq in examples[k]:
            all_vals.extend(seq)
        concatenated[k] = all_vals
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // block_size) * block_size
    result = {k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result

def compute_precision_flags():
    if not torch.cuda.is_available():
        return False, False
    major = torch.cuda.get_device_capability(0)[0]
    fp16 = major < 8
    bf16 = major >= 8
    return fp16, bf16

def make_splits(dataset_name, dataset_config, tokenizer, block_size, seed, 
                log_path=None, use_streaming=False, num_proc=1):
    # Use streaming only if explicitly requested
    ds = load_dataset(dataset_name, dataset_config, streaming=use_streaming)

    if "train" in ds:
        ds_train = ds["train"]
    else:
        parts = [s for s in ("validation", "test") if s in ds]
        assert parts, "No 'train' split and no alternative splits available."
        ds_train = concatenate_datasets([ds[s] for s in parts])

    ds_train = ds_train.filter(filter_non_empty)

    if "validation" in ds:
        print("Using validation split from dataset...")
        ds_val = ds["validation"].filter(filter_non_empty)
    elif "test" in ds:
        print("Using test split from dataset...")
        ds_val = ds["test"].filter(filter_non_empty)
    else:
        print("[Warning] No validation/test split found, train")
        ds_val = ds_train

    # Handle column names for streaming vs non-streaming datasets
    train_cols_to_remove = []
    val_cols_to_remove = []
    
    if hasattr(ds_train, 'column_names') and ds_train.column_names is not None:
        train_cols_to_remove = [c for c in ds_train.column_names if c != "text"]
    
    if hasattr(ds_val, 'column_names') and ds_val.column_names is not None:
        val_cols_to_remove = [c for c in ds_val.column_names if c != "text"]
    
    # Map with or without desc based on dataset type
    if use_streaming:
        tok_train = ds_train.map(
            lambda b: tokenize_batch(b, tokenizer),
            batched=True,
            remove_columns=train_cols_to_remove,
        )
        tok_val = ds_val.map(
            lambda b: tokenize_batch(b, tokenizer),
            batched=True,
            remove_columns=val_cols_to_remove,
        )
    else:
        tok_train = ds_train.map(
            lambda b: tokenize_batch(b, tokenizer),
            batched=True,
            remove_columns=train_cols_to_remove,
            desc="Tokenizing train",
            num_proc=num_proc,
        )
        tok_val = ds_val.map(
            lambda b: tokenize_batch(b, tokenizer),
            batched=True,
            remove_columns=val_cols_to_remove,
            desc="Tokenizing val",
            num_proc=num_proc,
        )

    if use_streaming:
        lm_train = tok_train.map(
            lambda b: group_texts(b, block_size),
            batched=True,
        ).shuffle(seed=seed)

        lm_val = tok_val.map(
            lambda b: group_texts(b, block_size),
            batched=True,
        )
    else:
        lm_train = tok_train.map(
            lambda b: group_texts(b, block_size),
            batched=True,
            desc=f"Grouping train into blocks of {block_size}",
            num_proc=num_proc,
        ).shuffle(seed=seed)

        lm_val = tok_val.map(
            lambda b: group_texts(b, block_size),
            batched=True,
            desc=f"Grouping val into blocks of {block_size}",
            num_proc=num_proc,
        )

    return lm_train, lm_val

class LMEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, zeroshot_tasks, fewshot_tasks, log_path, num_fewshot,
                 max_eval_samples=None,
                 eval_at_begin=True, eval_at_end=True,
                 every_n_steps=None, save_on_eval=True):
        self.tok = tokenizer
        self.zeroshot_tasks = zeroshot_tasks
        self.fewshot_tasks = fewshot_tasks
        self.log_path = log_path
        self.num_fewshot = num_fewshot
        self.max_eval_samples = max_eval_samples
        self.eval_at_begin = eval_at_begin
        self.eval_at_end = eval_at_end
        self.every_n_steps = every_n_steps
        self.save_on_eval = save_on_eval
        self.has_run_begin = False

    def _run_evaluation(self, args, state, model, stage=""):
        # Only run evaluation on main process in distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank != 0:
            return

        if args.bf16:
            dtype = "bfloat16"
        elif args.fp16:
            dtype = "float16"
        else:
            dtype = "float32"

        # Handle multi-GPU setup
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # Determine device configuration
        if torch.cuda.is_available() and world_size > 1:
            # Multi-GPU setup - use parallelization in lm_eval
            device_str = "cuda"
            model_args = f"pretrained={{tmp}},dtype={dtype},parallelize=True"
        elif torch.cuda.is_available():
            # Single GPU
            device = next(model.parameters()).device
            device_str = f"cuda:{device.index}" if device.index is not None else "cuda:0"
            model_args = f"pretrained={{tmp}},dtype={dtype}"
        else:
            # CPU
            device_str = "cpu"
            model_args = f"pretrained={{tmp}},dtype={dtype}"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                stage_str = f" ({stage})" if stage else ""
                log(f"[LMEval] Running evaluation{stage_str} at step {state.global_step} (world_size={world_size}, device={device_str})...", filepath=self.log_path)

                # Save model - handle distributed training
                if hasattr(model, 'module'):
                    # Model is wrapped (e.g., DDP, FSDP)
                    save_model = model.module
                else:
                    save_model = model

                # Check if this is a PEFT model (LoRA)
                is_peft_model = hasattr(save_model, 'peft_config')
                if is_peft_model:
                    log(f"[LMEval] Detected PEFT model, merging adapters for evaluation...", filepath=self.log_path)
                    save_model = save_model.merge_and_unload()

                save_model.save_pretrained(tmp)

                self.tok.save_pretrained(tmp)

                res_zeroshot = simple_evaluate(
                    model="hf",
                    model_args=model_args.format(tmp=tmp),
                    tasks=self.zeroshot_tasks,
                    num_fewshot=0,
                    batch_size="auto",
                    device=device_str,
                    limit=self.max_eval_samples,
                    log_samples=False,  # Otherwise, will log individual samples in the JSON.
                    random_seed=args.seed,
                    numpy_random_seed=args.seed,
                    torch_random_seed=args.seed,
                    fewshot_random_seed=args.seed,
                )

                res_fewshot = simple_evaluate(
                    model="hf",
                    model_args=model_args.format(tmp=tmp),
                    tasks=self.fewshot_tasks,
                    num_fewshot=self.num_fewshot,
                    batch_size="auto",
                    device=device_str,
                    limit=self.max_eval_samples,
                    log_samples=False,  # Otherwise, will log individual samples in the JSON.
                    random_seed=args.seed,
                    numpy_random_seed=args.seed,
                    torch_random_seed=args.seed,
                    fewshot_random_seed=args.seed,
                )

                assert "results" in res_zeroshot and "results" in res_fewshot
                filename = f"lm_eval_{stage}_{state.global_step}.json" if stage else f"lm_eval_step{state.global_step}.json"
                out = os.path.join(args.output_dir, filename)
                merged_dict = {**res_zeroshot["results"], **res_fewshot["results"]}
                with open(out, "w") as f:
                    json.dump({"results": merged_dict}, f, indent=2)
                log(f"[LMEval] Results saved to {out}", filepath=self.log_path)

                # Log to wandb if available
                wandb_metrics = {}
                for task, metrics in merged_dict.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                log(f"[LMEval] {task}.{metric_name}: {value:.4f}", filepath=self.log_path)
                                # Add to wandb metrics with eval prefix
                                wandb_key = f"eval/{task}.{metric_name}"
                                wandb_metrics[wandb_key] = value
                
                # Log to wandb if initialized
                if wandb.run is not None:
                    wandb.log(wandb_metrics, step=state.global_step)
                    log(f"[LMEval] Logged {len(wandb_metrics)} metrics to wandb", filepath=self.log_path)

                if self.save_on_eval:
                    ckpt_dir = os.path.join(args.output_dir, f"eval_ckpt_{stage or 'interval'}_step{state.global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    if hasattr(model, 'module'):
                        model.module.save_pretrained(ckpt_dir, save_safetensors=getattr(args, "save_safetensors", True))
                    else:
                        model.save_pretrained(ckpt_dir, save_safetensors=getattr(args, "save_safetensors", True))
                    self.tok.save_pretrained(ckpt_dir)
                    log(f"[LMEval] Weights saved to {ckpt_dir}", filepath=self.log_path)

        except Exception as e:
            log(f"[LMEval] Error during evaluation{stage_str} at step {state.global_step}: {e}", filepath=self.log_path)

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

@torch.no_grad()
def eval_perplexity_with_trainer(trainer, eval_dataset):
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    loss = metrics.get("eval_loss", None)
    if loss is None:
        return None, metrics
    loss_clamped = min(loss, 20.0)
    ppl = math.exp(loss_clamped)
    metrics["eval_ppl"] = ppl
    return ppl, metrics

def choice_logprob(model, device, tokenizer, prompt_text, choice_text):
    combined = prompt_text + " " + str(choice_text)
    enc = tokenizer(
        combined,
        add_special_tokens=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    choice_ids = tokenizer(" " + str(choice_text), add_special_tokens=False)["input_ids"]
    k = min(len(choice_ids), input_ids.size(-1))
    labels = torch.full_like(input_ids, -100)
    labels[:, -k:] = input_ids[:, -k:]
    out = model(input_ids=input_ids, labels=labels)
    total_logprob = -out.loss.item() * k
    return total_logprob


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


class CustomLossTrainer(Trainer):
    def __init__(self,
                 *args,
                 loss_fn: torch.nn.Module,
                 dispersion: str,
                 dispersion_coeff: float,
                 dispersion_loc: str,
                 tau_infonce_l2: float,
                 tau_infonce_cos: float,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

        self.use_disp = dispersion is not None and dispersion_coeff > 0.0
        self.disp_coeff = dispersion_coeff
        self.disp_loc = dispersion_loc

        if self.use_disp:
            variant = dispersion.lower()
            self.disp_loss_fn = DispersionLoss(variant=variant,
                                               tau_l2=tau_infonce_l2,
                                               tau_cos=tau_infonce_cos)

    def disperse_hidden_states(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        '''
        Computes dispersion for last layer or averages across all layers (excluding emb layer at index 0),
        with embeddings rearranged to [num_samples, sequence_length].

        hidden_states: tuple of tensors, each [B, seq_len, feature_dim]
        '''
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        # Use hidden states ONLY if we're training AND dispersion is on.
        want_disp = self.use_disp and model.training
        outputs = model(**inputs, output_hidden_states=want_disp)
        logits = outputs.logits

        default_loss = self.loss_fn(logits, labels)

        # Add dispersion ONLY in training
        if want_disp:
            disp_loss = self.disperse_hidden_states(outputs.hidden_states)
            total_loss = default_loss + self.disp_coeff * disp_loss
            outputs.dispersion_loss = disp_loss.detach()
        else:
            disp_loss = torch.zeros_like(default_loss)
            total_loss = default_loss

        if (model.training and
            self.state.global_step > 0 and
            self.state.global_step % self.args.logging_steps == 0):

            custom_losses = {
                "train/dispersion_loss": disp_loss.detach().item(),
                "train/default_loss": default_loss.detach().item(),
                "train/total_loss": total_loss.detach().item(),
            }

            # Log to trainer's system
            self.log(custom_losses)

        return (total_loss, outputs) if return_outputs else total_loss


def main(args):
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_name)
    if hasattr(config, "loss_type"):
        delattr(config, "loss_type")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)

    max_ctx = getattr(model.config, "n_positions",
              getattr(model.config, "max_position_embeddings",
              getattr(model.config, "max_sequence_length", 1024)))
    tokenizer.model_max_length = max_ctx
    print(f"Tokenizer model max length: {tokenizer.model_max_length}")

    block_size = args.block_size or getattr(tokenizer, "model_max_length", 8192)
    print(f"Block size: {block_size}")

    lm_train, lm_val = make_splits(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        block_size=block_size,
        seed=args.seed,
        log_path=args.log_path,
        use_streaming=args.use_streaming,
        num_proc=args.num_workers,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_step = args.per_device_train_batch_size * block_size * args.gradient_accumulation_steps * world_size
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step computed as 0; check batch size/accumulation/block_size.")
    max_steps = math.ceil(args.train_tokens / tokens_per_step)
    log(f"Training for {args.train_tokens} tokens, which is {max_steps} steps.", filepath=args.log_path)
    log_every_n_steps = max_steps // args.num_ckpt + 1

    fp16, bf16 = compute_precision_flags()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_steps=max_steps,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        log_level="info",
        logging_steps=max(1, max_steps // 20),
        log_on_each_node=False,
        save_strategy="no",  # We will save checkpoints using LMEvalCallback.
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=True,
        warmup_ratio=0.1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model.resize_token_embeddings(len(tokenizer))

    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        loss_fn=CausalLMLoss(),
        dispersion=args.dispersion,
        dispersion_coeff=args.dispersion_coeff,
        dispersion_loc=args.dispersion_loc,
        tau_infonce_cos=args.tau_infonce_cos,
        tau_infonce_l2=args.tau_infonce_l2,
        train_dataset=lm_train,
        eval_dataset=lm_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    log("=== Mid-training setup ===", filepath=args.log_path)
    log(f"Model: {args.model_name}", filepath=args.log_path)
    log(str(model.config), filepath=args.log_path)
    log(f"Dataset: {args.dataset_name} ({args.dataset_config})", filepath=args.log_path)
    log(f"Block size: {block_size}", filepath=args.log_path)
    log(f"Per-device batch: {args.per_device_train_batch_size} | Grad accum: {args.gradient_accumulation_steps} | World size: {world_size}", filepath=args.log_path)
    log(f"Token budget: {args.train_tokens} | Tokens/step: {tokens_per_step} | Max steps: {max_steps}", filepath=args.log_path)
    log(f"Precision: {'bf16' if bf16 else ('fp16' if fp16 else 'fp32')}", filepath=args.log_path)

    log(f"\n\nEvaluation before mid-training.", filepath=args.log_path)
    ppl, eval_metrics = eval_perplexity_with_trainer(trainer, lm_val)
    if ppl is not None:
        log(f"[Eval] Validation Perplexity: {ppl:.3f}", filepath=args.log_path)
    if eval_metrics:
        log(f"[Eval] Raw eval metrics: {eval_metrics}", filepath=args.log_path)

    # https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
    zeroshot_tasks = [
        "hellaswag",
        "lambada",
        "paloma_wikitext_103",
        "piqa",
        "truthfulqa_mc2",
        "winogrande",
    ]
    fewshot_tasks = [
        "arc_challenge",
        "gsm8k",
        "mmlu",
        "medmcqa",
    ]
    trainer.add_callback(LMEvalCallback(tokenizer,
                                        zeroshot_tasks,
                                        fewshot_tasks,
                                        log_path=args.log_path,
                                        num_fewshot=args.num_fewshot,
                                        max_eval_samples=args.max_eval_samples,
                                        every_n_steps=log_every_n_steps,
                                        save_on_eval=not args.no_save_model))

    trainer.train()
    # trainer.save_model(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    log(f"\n\nEvaluation after mid-training.", filepath=args.log_path)
    ppl, eval_metrics = eval_perplexity_with_trainer(trainer, lm_val)
    if ppl is not None:
        log(f"[Eval] Validation Perplexity: {ppl:.3f}", filepath=args.log_path)
    if eval_metrics:
        log(f"[Eval] Raw eval metrics: {eval_metrics}", filepath=args.log_path)

    log(f"Done. Saved to {args.output_dir}", filepath=args.log_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mid-train with a token budget.")
    ap.add_argument("--model_name", type=str, default="allenai/OLMo-2-0425-1B",
                    help="Hugging Face model id to start from (pretrained).")
    ap.add_argument("--dataset_name", type=str, default="Salesforce/wikitext",
                    help="Hugging Face dataset id.")
    ap.add_argument("--dataset_config", type=str,
                    default=os.environ.get("DATASET_CONFIG", "wikitext-103-raw-v1"),
                    help="Dataset config (e.g., wikitext-2-raw-v1).")
    # Removed --hf_token argument since we rely on huggingface-cli login
    ap.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")
    ap.add_argument("--wandb_project", type=str, default="transformer-dispersion", help="Wandb project name")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (auto-generated if None)")
    ap.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    ap.add_argument("--train_tokens", type=int, required=True,
                    help="Total number of tokens to train on (token budget).")
    ap.add_argument("--dispersion", type=str, default=None, help="Dispersion loss.")
    ap.add_argument("--dispersion_coeff", type=float, default=1, help="Dispersion loss weight.")
    ap.add_argument("--dispersion_loc", type=str, default='last', help="Dispersion loss location.")
    ap.add_argument("--tau_infonce_l2", type=float, default=0.5, help="Temperature.")
    ap.add_argument("--tau_infonce_cos", type=float, default=0.5, help="Temperature.")
    ap.add_argument("--num_fewshot", type=int, default=1, help="Eval num_fewshot.")
    ap.add_argument("--max_eval_samples", type=int, default=200, help="Eval max_eval_samples.")
    ap.add_argument("--num_ckpt", type=int, default=10, help="Number of checkpoints.")
    ap.add_argument("--no_save_model", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    ap.add_argument("--block_size", type=int, default=None,
                    help="Context length (default: tokenizer max).")
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_streaming", action="store_true",
                    help="Use streaming dataset loading (default: False). Useful for very large datasets.")

    args = ap.parse_args()

    args.output_dir = f'./results/midtrain_{args.model_name}_{"-".join(args.dataset_name.split("/"))}_lr-{args.lr}_token-{args.train_tokens}_disp-{args.dispersion}-{args.dispersion_coeff}-{args.dispersion_loc}_fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_seed-{args.seed}'
    args.log_path = os.path.join(args.output_dir, 'log.txt')
    main(args)
