#!/usr/bin/env python3
"""
Standalone script to evaluate checkpoints using the same logic as LMEvalCallback.
This script maximally reuses the evaluation code from midtrain.py.
"""

import os
import json
import tempfile
import argparse
import glob
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import simple_evaluate


def log(s, filepath=None, to_console=True):
    """
    Logs a string to either file or console
    """
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


class CheckpointEvaluator:
    """
    Evaluator class that reuses the evaluation logic from LMEvalCallback
    """
    
    def __init__(self, 
                 zeroshot_tasks: List[str],
                 fewshot_tasks: List[str],
                 num_fewshot: int = 1,
                 max_eval_samples: Optional[int] = None,
                 log_path: Optional[str] = None,
                 seed: int = 1):
        self.zeroshot_tasks = zeroshot_tasks
        self.fewshot_tasks = fewshot_tasks
        self.num_fewshot = num_fewshot
        self.max_eval_samples = max_eval_samples
        self.log_path = log_path
        self.seed = seed
        
    def evaluate_checkpoint(self, checkpoint_path: str, output_dir: str, 
                          checkpoint_name: str = None) -> dict:
        """
        Evaluate a single checkpoint using the same logic as LMEvalCallback._run_evaluation
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            output_dir: Directory to save evaluation results
            checkpoint_name: Name for the checkpoint (used in output filename)
            
        Returns:
            Dictionary containing evaluation results
        """
        if checkpoint_name is None:
            checkpoint_name = os.path.basename(checkpoint_path)
            
        log(f"[Eval] Loading checkpoint from {checkpoint_path}", filepath=self.log_path)
        
        # Load tokenizer and model
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        except Exception as e:
            log(f"[Eval] Error loading checkpoint {checkpoint_path}: {e}", filepath=self.log_path)
            return {}
        
        # Determine precision and device setup
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_str = f"cuda:{device}"
            
            # Check device capability for precision
            major = torch.cuda.get_device_capability(device)[0]
            if major >= 8:
                dtype = "bfloat16"
            else:
                dtype = "float16"
        else:
            device_str = "cpu"
            dtype = "float32"
            
        # Handle multi-GPU setup
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if torch.cuda.is_available() and world_size > 1:
            device_str = "cuda"
            model_args = f"pretrained={{tmp}},dtype={dtype},parallelize=True"
        else:
            model_args = f"pretrained={{tmp}},dtype={dtype}"
            
        log(f"[Eval] Evaluating {checkpoint_name} (device={device_str}, dtype={dtype})", 
            filepath=self.log_path)
        
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # Check if this is a PEFT model (LoRA)
                is_peft_model = hasattr(model, 'peft_config')
                if is_peft_model:
                    log(f"[Eval] Detected PEFT model, merging adapters for evaluation...", 
                        filepath=self.log_path)
                    model = model.merge_and_unload()
                
                # Save model and tokenizer to temporary directory
                model.save_pretrained(tmp)
                tokenizer.save_pretrained(tmp)
                
                # Run zero-shot evaluation
                log(f"[Eval] Running zero-shot tasks: {self.zeroshot_tasks}", filepath=self.log_path)
                res_zeroshot = simple_evaluate(
                    model="hf",
                    model_args=model_args.format(tmp=tmp),
                    tasks=self.zeroshot_tasks,
                    num_fewshot=0,
                    batch_size="auto",
                    device=device_str,
                    limit=self.max_eval_samples,
                    log_samples=False,
                    random_seed=self.seed,
                    numpy_random_seed=self.seed,
                    torch_random_seed=self.seed,
                    fewshot_random_seed=self.seed,
                )
                
                # Run few-shot evaluation
                log(f"[Eval] Running few-shot tasks: {self.fewshot_tasks}", filepath=self.log_path)
                res_fewshot = simple_evaluate(
                    model="hf",
                    model_args=model_args.format(tmp=tmp),
                    tasks=self.fewshot_tasks,
                    num_fewshot=self.num_fewshot,
                    batch_size="auto",
                    device=device_str,
                    limit=self.max_eval_samples,
                    log_samples=False,
                    random_seed=self.seed,
                    numpy_random_seed=self.seed,
                    torch_random_seed=self.seed,
                    fewshot_random_seed=self.seed,
                )
                
                # Merge results
                assert "results" in res_zeroshot and "results" in res_fewshot
                merged_dict = {**res_zeroshot["results"], **res_fewshot["results"]}
                
                # Save results
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"lm_eval_{checkpoint_name}.json")
                with open(output_file, "w") as f:
                    json.dump({"results": merged_dict}, f, indent=2)
                    
                log(f"[Eval] Results saved to {output_file}", filepath=self.log_path)
                
                # Log individual metrics
                for task, metrics in merged_dict.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                log(f"[Eval] {task}.{metric_name}: {value:.4f}", filepath=self.log_path)
                
                return merged_dict
                
        except Exception as e:
            log(f"[Eval] Error during evaluation of {checkpoint_name}: {e}", filepath=self.log_path)
            return {}


def find_checkpoints(base_dir: str, patterns: List[str] = None) -> List[str]:
    """
    Find all checkpoint directories matching the patterns
    """
    if patterns is None:
        patterns = ["eval_ckpt_*"]
    
    checkpoints = []
    for pattern in patterns:
        search_pattern = os.path.join(base_dir, pattern)
        checkpoints.extend(glob.glob(search_pattern))
    
    # Remove duplicates and sort
    checkpoints = list(set(checkpoints))
    checkpoints.sort()  # Sort by name for consistent ordering
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints using LMEval")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing checkpoints or single checkpoint path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--checkpoint_patterns", type=str, nargs="+", default=["eval_ckpt_*"],
                        help="Patterns to match checkpoint directories (can specify multiple)")
    parser.add_argument("--num_fewshot", type=int, default=1,
                        help="Number of few-shot examples")
    parser.add_argument("--max_eval_samples", type=int, default=200,
                        help="Maximum number of samples to evaluate per task")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to log file")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for evaluation (default: 1, matching training default)")
    
    # Task selection arguments
    parser.add_argument("--zeroshot_tasks", type=str, nargs="+",
                        default=["hellaswag", "lambada", "paloma_wikitext_103", "piqa", "truthfulqa_mc2", "winogrande"],
                        help="Zero-shot evaluation tasks")
    parser.add_argument("--fewshot_tasks", type=str, nargs="+",
                        default=["arc_challenge", "gsm8k", "mmlu", "medmcqa"],
                        help="Few-shot evaluation tasks")
    
    # Single checkpoint evaluation
    parser.add_argument("--single_checkpoint", action="store_true",
                        help="Treat checkpoint_dir as a single checkpoint rather than a directory of checkpoints")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, "eval_log.txt")
    
    log(f"=== Checkpoint Evaluation ===", filepath=args.log_path)
    log(f"Checkpoint directory: {args.checkpoint_dir}", filepath=args.log_path)
    log(f"Checkpoint patterns: {args.checkpoint_patterns}", filepath=args.log_path)
    log(f"Output directory: {args.output_dir}", filepath=args.log_path)
    log(f"Zero-shot tasks: {args.zeroshot_tasks}", filepath=args.log_path)
    log(f"Few-shot tasks: {args.fewshot_tasks}", filepath=args.log_path)
    log(f"Num fewshot: {args.num_fewshot}", filepath=args.log_path)
    log(f"Max eval samples: {args.max_eval_samples}", filepath=args.log_path)
    log(f"Random seed: {args.seed}", filepath=args.log_path)
    
    # Initialize evaluator
    evaluator = CheckpointEvaluator(
        zeroshot_tasks=args.zeroshot_tasks,
        fewshot_tasks=args.fewshot_tasks,
        num_fewshot=args.num_fewshot,
        max_eval_samples=args.max_eval_samples,
        log_path=args.log_path,
        seed=args.seed
    )
    
    # Find checkpoints
    if args.single_checkpoint:
        if not os.path.exists(args.checkpoint_dir):
            log(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist", filepath=args.log_path)
            return
        checkpoints = [args.checkpoint_dir]
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir, args.checkpoint_patterns)
        if not checkpoints:
            log(f"No checkpoints found in {args.checkpoint_dir} with patterns {args.checkpoint_patterns}", 
                filepath=args.log_path)
            return
    
    log(f"Found {len(checkpoints)} checkpoint(s) to evaluate", filepath=args.log_path)
    
    # Evaluate each checkpoint
    all_results = {}
    for i, checkpoint_path in enumerate(checkpoints, 1):
        checkpoint_name = os.path.basename(checkpoint_path)
        log(f"\n[{i}/{len(checkpoints)}] Evaluating checkpoint: {checkpoint_name}", filepath=args.log_path)
        
        results = evaluator.evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            checkpoint_name=checkpoint_name
        )
        
        if results:
            all_results[checkpoint_name] = results
            log(f"Successfully evaluated {checkpoint_name}", filepath=args.log_path)
        else:
            log(f"Failed to evaluate {checkpoint_name}", filepath=args.log_path)
    
    # Save summary of all results
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    log(f"\nEvaluation complete. Summary saved to {summary_file}", filepath=args.log_path)
    log(f"Individual results saved in {args.output_dir}", filepath=args.log_path)


if __name__ == "__main__":
    main()
