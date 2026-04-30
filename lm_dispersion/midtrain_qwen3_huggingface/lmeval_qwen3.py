import os
import json
import argparse
import torch
from lm_eval import simple_evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(s, filepath=None, to_console=True):
    if to_console:
        print(s)
    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a+") as o:
            o.write(s + "\n")


def pick_device(device_arg: str) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return "cuda"
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def pick_dtype(dtype_arg: str, device: str) -> str:
    if dtype_arg and dtype_arg != "auto":
        if device == "cpu" and dtype_arg in {"float16", "bfloat16"}:
            return "float32"
        return dtype_arg
    if device == "cpu":
        return "float32"
    return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"


def build_model_args(args, dtype: str, parallelize: bool) -> str:
    parts = [
        f"pretrained={args.model_name}",
        f"dtype={dtype}",
        "trust_remote_code=True",
    ]
    if args.cache_dir:
        parts.append(f"cache_dir={args.cache_dir}")
    if args.hf_token and args.hf_token.strip():
        parts.append(f"token={args.hf_token.strip()}")
    if parallelize:
        parts.append("parallelize=True")
    return ",".join(parts)


def run_eval(args):
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    parallelize = args.parallelize
    if parallelize is None:
        parallelize = torch.cuda.is_available() and torch.cuda.device_count() > 1

    model_args = build_model_args(args, dtype, parallelize)

    log("=== LMEval (pretrained) ===", filepath=args.log_path)
    log(f"Model: {args.model_name}", filepath=args.log_path)
    log(f"Device: {device} | Dtype: {dtype} | Parallelize: {parallelize}", filepath=args.log_path)
    log(f"Zero-shot tasks: {args.zeroshot_tasks}", filepath=args.log_path)
    log(f"Few-shot tasks: {args.fewshot_tasks}", filepath=args.log_path)
    log(f"Num fewshot: {args.num_fewshot}", filepath=args.log_path)
    log(f"Max eval samples: {args.max_eval_samples}", filepath=args.log_path)
    log(f"Context length: {args.context_len}", filepath=args.log_path)
    log(f"Max gen tokens: {args.max_gen_tokens}", filepath=args.log_path)
    log(f"Batch size: {args.batch_size}", filepath=args.log_path)
    log(f"Seed: {args.seed}", filepath=args.log_path)

    if args.max_gen_tokens > args.context_len:
        raise ValueError("max_gen_tokens must be <= context_len.")

    with torch.inference_mode():
        res_zeroshot = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=args.zeroshot_tasks,
            num_fewshot=0,
            batch_size=args.batch_size,
            device=device,
            limit=args.max_eval_samples,
            gen_kwargs={"max_gen_toks": args.max_gen_tokens, "do_sample": False},
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

        res_fewshot = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=args.fewshot_tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=device,
            limit=args.max_eval_samples,
            gen_kwargs={"max_gen_toks": args.max_gen_tokens, "do_sample": False},
            log_samples=False,
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )

    if "results" not in res_zeroshot or "results" not in res_fewshot:
        raise RuntimeError("LMEval did not return expected results.")

    merged_dict = {**res_zeroshot["results"], **res_fewshot["results"]}
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "lm_eval_begin_0.json")
    with open(out, "w") as f:
        json.dump({"results": merged_dict}, f, indent=2)
    log(f"Results saved to {out}", filepath=args.log_path)

    for task, metrics in merged_dict.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    log(f"{task}.{metric_name}: {value:.4f}", filepath=args.log_path)


def main():
    ap = argparse.ArgumentParser(description="Run lm-eval on a pretrained Qwen3 model.")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B",
                    help="Hugging Face model id to evaluate.")
    ap.add_argument("--hf_token", type=str, default=None,
                    help="HF token if needed for gated/private models.")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="HF cache dir.")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Output directory for lm_eval JSON and logs.")
    ap.add_argument("--log_path", type=str, default=None,
                    help="Path to log file (default: <output_dir>/log.txt).")
    ap.add_argument("--num_fewshot", type=int, default=5)
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument("--context_len", type=int, default=4096,
                    help="Context length used during training (for consistency checks).")
    ap.add_argument("--max_gen_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=str, default="auto",
                    help="Batch size for lm_eval ('auto' or integer).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto",
                    help="Device for lm_eval (e.g. 'cuda', 'cuda:0', 'cpu').")
    ap.add_argument("--dtype", type=str, default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--parallelize", action="store_true", default=None,
                    help="Enable model parallelism across GPUs.")
    ap.add_argument("--no_parallelize", dest="parallelize", action="store_false",
                    help="Disable model parallelism across GPUs.")
    ap.add_argument("--zeroshot_tasks", type=str, nargs="+",
                    default=[
                        "anli",
                        "hellaswag",
                        "lambada",
                        "openbookqa",
                        "paloma_wikitext_103",
                        "piqa",
                        "truthfulqa_mc2",
                        "winogrande",
                    ])
    ap.add_argument("--fewshot_tasks", type=str, nargs="+",
                    default=[
                        "arc_challenge",
                        "arc_easy",
                        "mathqa",
                        "mmlu",
                        "medmcqa",
                    ])
    args = ap.parse_args()

    model_str = args.model_name.replace("/", "-")
    args.dataset_name = "Salesforce/wikitext"
    args.lr = 0
    args.train_tokens = 0
    args.dispersion = "none"
    args.dispersion_coeff = 0
    args.dispersion_loc = "all"
    args.tau_cos = 1.0
    args.tau_l2 = 1.0
    if args.output_dir is None:
        args.output_dir = f'./results/midtrain_{model_str}_{"-".join(args.dataset_name.split("/"))}_lr-{args.lr}_token-{args.train_tokens}_disp-{args.dispersion}-{args.dispersion_coeff}-{args.dispersion_loc}-tau_cos-{args.tau_cos}-tau_l2-{args.tau_l2}_fewshot-{args.num_fewshot}_maxsample-{args.max_eval_samples}_seed-{args.seed}'
    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, "log.txt")

    run_eval(args)


if __name__ == "__main__":
    main()
