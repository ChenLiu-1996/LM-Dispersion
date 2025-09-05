#!/usr/bin/env python3
"""
Simple script to check if a model's tokenizer has a chat_template.
This helps understand why LLaMA-Factory defaults to "empty" template for stage: pt.
"""

from transformers import AutoTokenizer
import sys

def check_tokenizer_template(model_name_or_path):
    """Load tokenizer and check if it has a chat_template."""
    print(f"Loading tokenizer from: {model_name_or_path}")
    
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"📝 Tokenizer class: {tokenizer.__class__.__name__}")
        print(f"🔤 Vocab size: {tokenizer.vocab_size}")
        print(f"🏷️  Model max length: {getattr(tokenizer, 'model_max_length', 'Not set')}")
        
        # Check the chat_template attribute
        chat_template = getattr(tokenizer, 'chat_template', None)
        
        print("\n" + "="*60)
        print("🎯 CHAT TEMPLATE ANALYSIS")
        print("="*60)
        
        if chat_template is None:
            print("❌ chat_template is None")
            print("📋 This means LLaMA-Factory will:")
            print("   1. Check: isinstance(tokenizer.chat_template, str) → False")
            print("   2. Fall back to: TEMPLATES['empty']")
            print("   3. Show warning: 'template was not specified, use empty template.'")
        elif isinstance(chat_template, str):
            print("✅ chat_template is a string")
            print(f"📏 Length: {len(chat_template)} characters")
            print("📋 This means LLaMA-Factory will:")
            print("   1. Check: isinstance(tokenizer.chat_template, str) → True") 
            print("   2. Try to: parse_template(tokenizer)")
            print("   3. Show warning: 'template was not specified, try parsing...'")
            print("\n🔍 Chat template content (first 200 chars):")
            print("-" * 40)
            print(repr(chat_template[:200]))
            if len(chat_template) > 200:
                print("... (truncated)")
        else:
            print(f"⚠️  chat_template has unexpected type: {type(chat_template)}")
            print(f"📋 Value: {repr(chat_template)}")
        
        # Check special tokens
        print("\n" + "="*60)
        print("🔧 SPECIAL TOKENS")
        print("="*60)
        print(f"🔚 EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
        print(f"🔜 BOS token: {repr(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
        print(f"📄 PAD token: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")
        print(f"❓ UNK token: {repr(tokenizer.unk_token)} (ID: {getattr(tokenizer, 'unk_token_id', 'N/A')})")
        
        return chat_template
        
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

def simulate_llamafactory_logic(chat_template):
    """Simulate the exact logic from get_template_and_fix_tokenizer."""
    print("\n" + "="*60)
    print("🔮 SIMULATING LLAMAFACTORY LOGIC")
    print("="*60)
    
    # This simulates the logic from template.py lines 593-599
    template_is_none = True  # Assume data_args.template is None (not specified)
    
    if template_is_none:
        print("🔍 data_args.template is None (not specified in YAML)")
        if isinstance(chat_template, str):
            print("✅ isinstance(tokenizer.chat_template, str) → True")
            print("📝 LLaMA-Factory would: parse_template(tokenizer)")
            print("⚠️  Warning: 'template was not specified, try parsing the chat template from the tokenizer.'")
            result = "parsed_from_tokenizer"
        else:
            print("❌ isinstance(tokenizer.chat_template, str) → False")
            print("📝 LLaMA-Factory would: use TEMPLATES['empty']")
            print("⚠️  Warning: 'template was not specified, use empty template.'")
            result = "empty"
    else:
        print("✅ data_args.template is specified")
        result = "user_specified"
    
    print(f"\n🎯 FINAL RESULT: Would use '{result}' template")
    return result

if __name__ == "__main__":
    # Default model from your config
    model_name = "meta-llama/Llama-3.2-1B"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print("🚀 TOKENIZER TEMPLATE CHECKER")
    print("="*60)
    
    chat_template = check_tokenizer_template(model_name)
    simulate_llamafactory_logic(chat_template)
    
    print("\n" + "="*60)
    print("💡 CONCLUSION")
    print("="*60)
    print("This explains why LLaMA-Factory behavior with stage: pt:")
    print("• If chat_template is None → defaults to 'empty' template")
    print("• If chat_template exists → tries to parse it")
    print("• Either way, 'empty' template is perfect for pretraining!")
