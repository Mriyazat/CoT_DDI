"""
Clinical DDI Chatbot Demo – Gradio interface for the distilled student model.

Inputs:  Two drug names (autocomplete from 4,628 drugs)
Outputs: Interaction type, mechanism explanation, severity, drug profiles
"""

import os
import re
import json
import gradio as gr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"


def load_resources():
    """Load all static resources at startup."""
    with open(PROCESSED / "drug_profiles.json") as f:
        profiles = json.load(f)

    with open(PROCESSED / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    severity_map = {}
    sev_path = PROCESSED / "severity_map.json"
    if sev_path.exists():
        with open(sev_path) as f:
            severity_map = json.load(f)

    name_to_id = {}
    for did, prof in profiles.items():
        name = prof.get("name", did)
        name_to_id[name.lower()] = did
        name_to_id[did.lower()] = did

    syn_path = PROCESSED / "drug_synonyms.json"
    if syn_path.exists():
        with open(syn_path) as f:
            synonyms = json.load(f)
        for name, did in synonyms.items():
            name_to_id[name.lower()] = did

    drug_names = sorted(set(
        prof.get("name", did) for did, prof in profiles.items()
    ))

    return profiles, label_map, severity_map, name_to_id, drug_names


def format_profile(prof: dict) -> str:
    lines = []
    if prof.get("description"):
        lines.append(f"**Description:** {prof['description'][:300]}")
    if prof.get("mechanism_of_action"):
        lines.append(f"**Mechanism:** {prof['mechanism_of_action'][:200]}")
    if prof.get("enzymes"):
        lines.append(f"**Enzymes:** {'; '.join(prof['enzymes'][:5])}")
    if prof.get("transporters"):
        lines.append(f"**Transporters:** {'; '.join(prof['transporters'][:3])}")
    if prof.get("targets"):
        lines.append(f"**Targets:** {'; '.join(prof['targets'][:3])}")
    return "\n".join(lines) if lines else "No detailed profile available."


def resolve_drug(name: str, name_to_id: dict, profiles: dict):
    """Resolve a user-entered drug name to a DrugBank ID."""
    key = name.strip().lower()
    did = name_to_id.get(key)
    if did:
        return did, profiles.get(did, {}).get("name", did)
    for k, v in name_to_id.items():
        if key in k or k in key:
            return v, profiles.get(v, {}).get("name", v)
    return None, None


def get_severity(d1_id: str, d2_id: str, severity_map: dict) -> str:
    """Look up DDInter severity for a pair."""
    key1 = f"{d1_id}_{d2_id}"
    key2 = f"{d2_id}_{d1_id}"
    sev = severity_map.get(key1) or severity_map.get(key2)
    return sev if sev else "Not clinically rated"


MODEL = None
TOKENIZER = None


def load_model():
    global MODEL, TOKENIZER
    if MODEL is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    ckpt = os.environ.get("DDI_CHECKPOINT", str(BASE_DIR / "outputs" / "checkpoints" / "C_summary_s42" / "final"))
    model_name = os.environ.get("DDI_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    if os.path.exists(ckpt):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",
        )
        MODEL = PeftModel.from_pretrained(base, ckpt)
        MODEL.eval()
    else:
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map="auto",
        )
        MODEL.eval()


def predict_interaction(drug1: str, drug2: str,
                        profiles, label_map, severity_map, name_to_id):
    """Run the student model on a drug pair and return formatted results."""
    d1_id, d1_name = resolve_drug(drug1, name_to_id, profiles)
    d2_id, d2_name = resolve_drug(drug2, name_to_id, profiles)

    if not d1_id:
        return f"Drug not found: **{drug1}**", "", "", ""
    if not d2_id:
        return f"Drug not found: **{drug2}**", "", "", ""

    load_model()

    p1 = profiles.get(d1_id, {})
    p2 = profiles.get(d2_id, {})
    prompt_lines = [
        f"Drug 1: {d1_name} ({d1_id})",
    ]
    for field, key in [("Description", "description"),
                       ("Mechanism", "mechanism_of_action"),
                       ("Enzymes", "enzymes"),
                       ("Transporters", "transporters")]:
        val = p1.get(key)
        if val:
            if isinstance(val, list):
                val = "; ".join(val[:5])
            prompt_lines.append(f"  {field}: {str(val)[:200]}")

    prompt_lines.append(f"Drug 2: {d2_name} ({d2_id})")
    for field, key in [("Description", "description"),
                       ("Mechanism", "mechanism_of_action"),
                       ("Enzymes", "enzymes"),
                       ("Transporters", "transporters")]:
        val = p2.get(key)
        if val:
            if isinstance(val, list):
                val = "; ".join(val[:5])
            prompt_lines.append(f"  {field}: {str(val)[:200]}")

    prompt_lines.append("")
    prompt_lines.append("Predict the interaction type, explain the mechanism briefly, "
                        "and state the severity.")
    user_msg = "\n".join(prompt_lines)

    system_prompt = (
        "You are an expert pharmacologist specialising in drug-drug interactions. "
        "Given two drugs with their pharmacological profiles, analyse their "
        "mechanisms step-by-step and predict their interaction type. "
        "Include the severity if known."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    text = TOKENIZER.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    import torch
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(**inputs, max_new_tokens=512,
                             temperature=0.1, do_sample=True, top_p=0.95)
    response = TOKENIZER.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    label_match = re.findall(r"Y\s*=\s*(\d+)", response)
    if label_match:
        pred_label = int(label_match[-1])
        label_text = label_map.get(pred_label, "Unknown interaction")
        if "#Drug1" in label_text:
            label_text = label_text.replace("#Drug1", d1_name).replace("#Drug2", d2_name)
    else:
        label_text = "Could not parse prediction"

    severity = get_severity(d1_id, d2_id, severity_map)

    d1_profile = f"### {d1_name}\n{format_profile(p1)}"
    d2_profile = f"### {d2_name}\n{format_profile(p2)}"

    return (
        f"**Interaction:** {label_text}\n\n**Severity:** {severity}",
        response,
        d1_profile,
        d2_profile,
    )


def build_app():
    profiles, label_map, severity_map, name_to_id, drug_names = load_resources()

    with gr.Blocks(title="DDI Clinical Chatbot", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Drug-Drug Interaction Predictor\n"
                    "Enter two drugs to predict their interaction, mechanism, "
                    "and clinical severity.")

        with gr.Row():
            drug1_input = gr.Dropdown(
                choices=drug_names, label="Drug 1",
                allow_custom_value=True, filterable=True,
            )
            drug2_input = gr.Dropdown(
                choices=drug_names, label="Drug 2",
                allow_custom_value=True, filterable=True,
            )

        predict_btn = gr.Button("Predict Interaction", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                result_box = gr.Markdown(label="Prediction")
                reasoning_box = gr.Textbox(label="Model Reasoning", lines=10,
                                           interactive=False)
            with gr.Column(scale=1):
                d1_profile_box = gr.Markdown(label="Drug 1 Profile")
                d2_profile_box = gr.Markdown(label="Drug 2 Profile")

        predict_btn.click(
            fn=lambda d1, d2: predict_interaction(
                d1, d2, profiles, label_map, severity_map, name_to_id
            ),
            inputs=[drug1_input, drug2_input],
            outputs=[result_box, reasoning_box, d1_profile_box, d2_profile_box],
        )

        gr.Markdown(
            "---\n"
            "*Drug data from DrugBank v5.1.14. Severity labels from DDInter 2.0. "
            "Predictions from a fine-tuned Qwen2.5-7B model (C_summary condition). "
            "For research purposes only — not for clinical use.*"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
