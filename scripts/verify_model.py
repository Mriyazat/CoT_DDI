#!/usr/bin/env python3
"""Verify Qwen3-8B compatibility: LoRA target modules + chat template.

Run on the cluster after downloading the model:
  python3 scripts/verify_model.py [--model Qwen/Qwen3-8B]
"""
import argparse
import json
import sys
from pathlib import Path

EXPECTED_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"}


def verify_lora_targets(model_name: str) -> bool:
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"\nModel: {model_name}")
    print(f"Architecture: {config.architectures}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    named_modules = {n.split(".")[-1] for n, _ in model.named_modules()}
    linear_names = set()
    for name, module in model.named_modules():
        if hasattr(module, "weight") and len(module.weight.shape) == 2:
            short = name.split(".")[-1]
            linear_names.add(short)

    print(f"\nLinear module names found: {sorted(linear_names)}")

    missing = EXPECTED_TARGETS - linear_names
    if missing:
        print(f"\nWARNING: Expected LoRA targets NOT found: {missing}")
        print("  You must update configs/config.yaml student.lora.target_modules")
        return False
    else:
        print(f"\nAll expected LoRA targets present: {sorted(EXPECTED_TARGETS)}")
        return True


def verify_chat_template(model_name: str) -> bool:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nPad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"Vocab size: {len(tokenizer)}")

    messages = [
        {"role": "system", "content": "You are a pharmacology expert."},
        {"role": "user", "content": "Predict the DDI between Drug A and Drug B."},
        {"role": "assistant", "content": "Summary here.\n\nClassification: Y=1\nSeverity: Major"},
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"\nChat template output ({len(text)} chars):")
        print("-" * 60)
        print(text[:500])
        if len(text) > 500:
            print(f"... ({len(text) - 500} more chars)")
        print("-" * 60)

        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"Token count: {len(ids)}")

        gen_prompt = tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        print(f"\nGeneration prompt ends with: ...{gen_prompt[-100:]}")

        if "Classification:" in text and "Severity:" in text:
            print("\nChat template preserves structured output markers.")
        else:
            print("\nWARNING: Structured markers may be altered by chat template!")

        return True
    except Exception as e:
        print(f"\nERROR applying chat template: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    print("=" * 60)
    print("DDI V3: Model Compatibility Verification")
    print("=" * 60)

    ok_lora = verify_lora_targets(args.model)
    ok_chat = verify_chat_template(args.model)

    print("\n" + "=" * 60)
    if ok_lora and ok_chat:
        print("ALL CHECKS PASSED -- model is compatible with DDI V3 pipeline")
    else:
        print("SOME CHECKS FAILED -- review output above before proceeding")
    print("=" * 60)

    sys.exit(0 if (ok_lora and ok_chat) else 1)


if __name__ == "__main__":
    main()
