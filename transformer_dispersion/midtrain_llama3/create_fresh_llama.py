#!/usr/bin/env python3
"""
Create a fresh Llama model with random weights for true "from scratch" pretraining.
This copies the config and tokenizer from an existing model but reinitializes all weights.
"""

import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    LlamaForCausalLM,
    LlamaConfig
)
import os
import argparse

def create_fresh_llama_model(
    source_model: str = "meta-llama/Llama-3.2-1B",
    output_dir: str = "./fresh_llama_1b",
    seed: int = 42
):
    """
    Create a fresh Llama model with random weights.
    
    Args:
        source_model: HuggingFace model ID to copy config/tokenizer from
        output_dir: Directory to save the fresh model
        seed: Random seed for weight initialization
    """
    print(f"🚀 Creating fresh Llama model from {source_model}")
    print(f"📁 Output directory: {output_dir}")
    
    # Set random seed for reproducible initialization
    torch.manual_seed(seed)
    
    # Load config and tokenizer from source model
    print("📋 Loading config and tokenizer...")
    config = AutoConfig.from_pretrained(source_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(source_model, trust_remote_code=True)
    
    print(f"✅ Config loaded: {config.model_type}")
    print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"🔢 Vocab size: {config.vocab_size}")
    print(f"🧠 Hidden size: {config.hidden_size}")
    print(f"🔄 Num layers: {config.num_hidden_layers}")
    print(f"🎯 Num attention heads: {config.num_attention_heads}")
    
    # Create fresh model with random weights
    print("🎲 Initializing model with random weights...")
    model = LlamaForCausalLM(config)
    
    # Verify it's actually random (check a few weight values)
    first_layer_weight = model.model.layers[0].self_attn.q_proj.weight
    print(f"🔍 Sample weights from first layer: {first_layer_weight[0, :5].tolist()}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the fresh model and tokenizer
    print(f"💾 Saving fresh model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save some metadata
    metadata = {
        "source_model": source_model,
        "seed": seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "config_class": config.__class__.__name__,
        "model_class": model.__class__.__name__,
        "created_from_scratch": True
    }
    
    import json
    with open(os.path.join(output_dir, "creation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Fresh model created successfully!")
    print(f"📝 You can now use: model_name_or_path: {output_dir}")
    print("🎯 This model has completely random weights - true 'from scratch' training!")
    
    return output_dir

def verify_model_is_fresh(model_path: str, source_model: str = "meta-llama/Llama-3.2-1B"):
    """Compare weights between fresh model and source to verify they're different."""
    print(f"\n🔍 VERIFICATION: Comparing {model_path} vs {source_model}")
    
    try:
        # Load both models
        fresh_model = LlamaForCausalLM.from_pretrained(model_path)
        source_model_obj = LlamaForCausalLM.from_pretrained(source_model)
        
        # Compare a few weight tensors
        fresh_weights = fresh_model.model.layers[0].self_attn.q_proj.weight
        source_weights = source_model_obj.model.layers[0].self_attn.q_proj.weight
        
        # Check if they're different
        are_different = not torch.allclose(fresh_weights, source_weights, atol=1e-6)
        
        print(f"🎲 Fresh model weights (sample): {fresh_weights[0, :3].tolist()}")
        print(f"🏗️  Source model weights (sample): {source_weights[0, :3].tolist()}")
        print(f"✅ Weights are different: {are_different}")
        
        if are_different:
            print("🎉 SUCCESS: Model has fresh random weights!")
        else:
            print("⚠️  WARNING: Weights appear to be the same as source model!")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a fresh Llama model with random weights")
    parser.add_argument("--source", default="meta-llama/Llama-3.2-1B", 
                       help="Source model to copy config/tokenizer from")
    parser.add_argument("--output", default="./fresh_llama_1b",
                       help="Output directory for fresh model")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for weight initialization")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the created model has different weights")
    
    args = parser.parse_args()
    
    print("🎯 FRESH LLAMA MODEL CREATOR")
    print("="*50)
    
    output_path = create_fresh_llama_model(
        source_model=args.source,
        output_dir=args.output, 
        seed=args.seed
    )
    
    if args.verify:
        verify_model_is_fresh(output_path, args.source)
    
    print("\n" + "="*50)
    print("📋 NEXT STEPS:")
    print("="*50)
    print("1. Update your YAML config:")
    print(f"   model_name_or_path: {output_path}")
    print("2. Run your pretraining as usual")
    print("3. The model will start with completely random weights!")
