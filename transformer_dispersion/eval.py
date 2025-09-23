import os
import json
import tempfile
from copy import deepcopy
import torch
from lm_eval import simple_evaluate
import wandb
from transformers import TrainerCallback


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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LMEvalCallback(TrainerCallback):
    def __init__(self,
                 tokenizer,
                 zeroshot_tasks, fewshot_tasks,
                 log_path,
                 max_gen_tokens,
                 num_fewshot,
                 max_eval_samples=None,
                 eval_at_begin=True, eval_at_end=True,
                 every_n_steps=None, save_on_eval=True):
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
                    model_copy = deepcopy(save_model).to("cpu")
                    merged = model_copy.merge_and_unload()
                    merged.save_pretrained(tmp)
                    del model_copy, merged
                else:
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
                    gen_kwargs = {"max_gen_toks": self.max_gen_tokens},
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
                    gen_kwargs = {"max_gen_toks": self.max_gen_tokens},
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

                for task, metrics in merged_dict.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                log(f"[LMEval] {task}.{metric_name}: {value:.4f}", filepath=self.log_path)

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