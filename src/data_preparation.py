"""
Phase 1 – Data loading, drug-name resolution, stratified splitting, and prompt construction.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils import load_config, setup_logging, set_seed, ensure_dirs, categorize_interaction


FEW_SHOT_EXAMPLES = [

    {
        "drug1_name": "Warfarin",
        "drug1_smiles": "CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=CC=CC=C2OC1=O",
        "drug2_name": "Aspirin",
        "drug2_smiles": "CC(=O)OC1=CC=CC=C1C(O)=O",
        "label": 49,
        "label_text": "The risk or severity of adverse effects can be increased when Warfarin is combined with Aspirin.",
        "cot": (
            "Step 1: Warfarin is a vitamin K antagonist that inhibits clotting factor synthesis, "
            "producing an anticoagulant effect and increasing bleeding risk.\n"
            "Step 2: Aspirin irreversibly inhibits cyclooxygenase (COX-1/COX-2), blocking "
            "thromboxane A2 production and impairing platelet aggregation.\n"
            "Step 3: Co-administration amplifies bleeding risk through dual mechanisms – "
            "reduced clotting factor synthesis plus impaired platelet function – "
            "and aspirin can also displace warfarin from plasma protein binding sites, "
            "raising free warfarin concentration."
        ),
    },
    {
        "drug1_name": "Metformin",
        "drug1_smiles": "CN(C)C(=N)NC(N)=N",
        "drug2_name": "Furosemide",
        "drug2_smiles": "NS(=O)(=O)C1=CC(C(=O)O)=CC(NCC2=CC=CO2)=C1Cl",
        "label": 49,
        "label_text": "The risk or severity of adverse effects can be increased when Metformin is combined with Furosemide.",
        "cot": (
            "Step 1: Metformin inhibits hepatic gluconeogenesis and increases peripheral "
            "glucose uptake; a rare but serious side effect is lactic acidosis, especially "
            "when renal clearance is impaired.\n"
            "Step 2: Furosemide is a loop diuretic that can cause volume depletion and "
            "electrolyte imbalances, potentially reducing renal perfusion.\n"
            "Step 3: Furosemide-induced renal impairment and dehydration can decrease "
            "metformin clearance, increasing metformin plasma levels and the risk of "
            "lactic acidosis – a life-threatening adverse effect."
        ),
    },
    {
        "drug1_name": "Simvastatin",
        "drug1_smiles": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)CC(O)CC(O)=O)[C@@H]21",
        "drug2_name": "Itraconazole",
        "drug2_smiles": "CCC(C)N1N=CN(C1=O)C1=CC=C(C=C1)N1CCN(CC1)C1=CC=C(OC[C@H]2CO[C@@](CN3C=NC=N3)(O2)C2=CC=C(Cl)C=C2Cl)C=C1",
        "label": 73,
        "label_text": "The serum concentration of Simvastatin can be increased when it is combined with Itraconazole.",
        "cot": (
            "Step 1: Simvastatin is extensively metabolised by cytochrome P450 3A4 (CYP3A4) "
            "in the liver, and its plasma levels are normally kept low by this first-pass metabolism.\n"
            "Step 2: Itraconazole is a potent inhibitor of CYP3A4, blocking the primary "
            "metabolic clearance pathway of simvastatin.\n"
            "Step 3: When co-administered, itraconazole inhibits CYP3A4-mediated metabolism "
            "of simvastatin, causing a marked increase in simvastatin serum concentration, "
            "which elevates the risk of myopathy and rhabdomyolysis."
        ),
    },
    {
        "drug1_name": "Digoxin",
        "drug1_smiles": "C[C@@H]1O[C@@H](O[C@@H]2[C@@H](O)C[C@H](O[C@@H]3[C@@H](O)C[C@H](O[C@@H]4CC[C@@]5(C)[C@H](CC[C@@H]6[C@@H]5C[C@H](O)[C@]5(C)[C@@H](C7=CC(=O)OC7)CC[C@]65O)C4)OC3C)OC2C)C[C@@H](O)[C@@H]1O",
        "drug2_name": "Amiodarone",
        "drug2_smiles": "CCCCC1=C(C=C(C=C1I)C(=O)C1=CC=C(OCCN(CC)CC)C=C1)I",
        "label": 73,
        "label_text": "The serum concentration of Digoxin can be increased when it is combined with Amiodarone.",
        "cot": (
            "Step 1: Digoxin is a cardiac glycoside with a narrow therapeutic index; "
            "it is eliminated mainly by renal excretion via P-glycoprotein (P-gp) "
            "mediated efflux in the renal tubules.\n"
            "Step 2: Amiodarone is a potent inhibitor of P-glycoprotein, blocking "
            "the transporter responsible for digoxin renal and biliary clearance.\n"
            "Step 3: P-gp inhibition by amiodarone reduces digoxin elimination, "
            "raising serum digoxin concentration by 50-100%, which can cause "
            "dangerous cardiac arrhythmias, nausea, and visual disturbances."
        ),
    },

    {
        "drug1_name": "Erythromycin",
        "drug1_smiles": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
        "drug2_name": "Carbamazepine",
        "drug2_smiles": "NC(=O)N1C2=CC=CC=C2C=CC2=CC=CC=C12",
        "label": 47,
        "label_text": "The metabolism of Carbamazepine can be decreased when combined with Erythromycin.",
        "cot": (
            "Step 1: Carbamazepine is an anticonvulsant primarily metabolised by CYP3A4 "
            "in the liver to its active epoxide metabolite; its clearance is highly dependent "
            "on CYP3A4 activity.\n"
            "Step 2: Erythromycin is a macrolide antibiotic and a well-established mechanism-based "
            "inhibitor of CYP3A4, forming a nitroso-alkane complex with the enzyme's haem group.\n"
            "Step 3: CYP3A4 inhibition by erythromycin decreases the hepatic metabolism of "
            "carbamazepine, leading to elevated plasma levels and an increased risk of "
            "neurotoxicity (dizziness, ataxia, diplopia)."
        ),
    },

    {
        "drug1_name": "Rifampin",
        "drug1_smiles": "CO[C@H]1\\C=C\\O[C@@]2(C)OC3=C(C2=O)C2=C(O)C(\\C=N\\N4CCN(C)CC4)=C(NC(=O)\\C(C)=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)C(O)=C2C(O)=C3C",
        "drug2_name": "Dexamethasone",
        "drug2_smiles": "[H][C@@]12C[C@@H](C)[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)C=C[C@]12C",
        "label": 75,
        "label_text": "The serum concentration of Dexamethasone can be decreased when it is combined with Rifampin.",
        "cot": (
            "Step 1: Dexamethasone is a synthetic corticosteroid that is extensively metabolised "
            "by CYP3A4 in the liver; its bioavailability depends on this metabolic pathway.\n"
            "Step 2: Rifampin is one of the most potent known inducers of CYP3A4 (and CYP2C9, "
            "CYP2C19, and P-glycoprotein), upregulating enzyme expression via activation of "
            "the pregnane X receptor (PXR).\n"
            "Step 3: Rifampin-induced CYP3A4 upregulation accelerates dexamethasone metabolism, "
            "substantially lowering its serum concentration and potentially rendering the "
            "anti-inflammatory therapy ineffective."
        ),
    },
    {
        "drug1_name": "Phenobarbital",
        "drug1_smiles": "CCC1(C(=O)NC(=O)NC1=O)C1=CC=CC=C1",
        "drug2_name": "Metoprolol",
        "drug2_smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "label": 4,
        "label_text": "The metabolism of Metoprolol can be increased when combined with Phenobarbital.",
        "cot": (
            "Step 1: Metoprolol is a beta-1 selective adrenergic blocker metabolised primarily "
            "by CYP2D6, with additional contributions from CYP3A4; it undergoes extensive "
            "hepatic first-pass metabolism.\n"
            "Step 2: Phenobarbital is a barbiturate and potent inducer of multiple hepatic "
            "cytochrome P450 enzymes (CYP3A4, CYP2C, CYP1A2) via constitutive androstane "
            "receptor (CAR) activation, increasing enzyme protein levels.\n"
            "Step 3: Chronic phenobarbital administration upregulates CYP enzymes that "
            "contribute to metoprolol clearance, increasing its metabolism and reducing "
            "plasma concentrations, which may compromise beta-blocker efficacy."
        ),
    },
    {
        "drug1_name": "Methylprednisolone",
        "drug1_smiles": "[H][C@@]12CC[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1([H])[C@@]2([H])C[C@H](C)C2=CC(=O)C=C[C@]12C",
        "drug2_name": "Metformin",
        "drug2_smiles": "CN(C)C(=N)NC(N)=N",
        "label": 70,
        "label_text": "The therapeutic efficacy of Metformin can be decreased when used in combination with Methylprednisolone.",
        "cot": (
            "Step 1: Metformin is a biguanide antidiabetic that lowers blood glucose by "
            "suppressing hepatic gluconeogenesis, enhancing insulin sensitivity, and "
            "increasing peripheral glucose uptake.\n"
            "Step 2: Methylprednisolone is a synthetic corticosteroid that stimulates hepatic "
            "gluconeogenesis, promotes glycogenolysis, and induces peripheral insulin resistance "
            "as part of its metabolic effects.\n"
            "Step 3: The hyperglycaemic action of methylprednisolone directly opposes the "
            "glucose-lowering mechanism of metformin, reducing its therapeutic efficacy "
            "and potentially leading to loss of glycaemic control in diabetic patients."
        ),
    },
    {
        "drug1_name": "Lithium",
        "drug1_smiles": "[Li+]",
        "drug2_name": "Ibuprofen",
        "drug2_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",
        "label": 72,
        "label_text": "Lithium may decrease the excretion rate of Ibuprofen which could result in a higher serum level.",
        "cot": (
            "Step 1: Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) that is "
            "eliminated through renal excretion of both parent compound and glucuronide "
            "conjugates; its clearance depends on adequate renal function.\n"
            "Step 2: Lithium can impair renal concentrating ability and reduce glomerular "
            "filtration rate (GFR) through its effects on aquaporin-2 channels and renal "
            "tubular function, particularly with chronic use.\n"
            "Step 3: Lithium-induced reduction in renal function decreases the excretion rate "
            "of ibuprofen, leading to accumulation and higher serum levels, increasing the "
            "risk of NSAID-related gastrointestinal and renal adverse effects."
        ),
    },
]


def load_raw_data(cfg: dict) -> pd.DataFrame:
    path = cfg["data"]["raw_path"]
    df = pd.read_csv(path, sep="\t", dtype={"ID1": str, "ID2": str, "Y": int})
    df.columns = ["drug1_id", "drug2_id", "label", "label_text_template", "drug1_smiles", "drug2_smiles"]
    return df


def load_name_map(cfg: dict) -> dict:
    path = cfg["data"]["name_map_path"]
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def resolve_drug_names(df: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    """Add drug name columns.  Falls back to DrugBank ID when name is unknown."""
    df["drug1_name"] = df["drug1_id"].map(lambda x: name_map.get(x, x))
    df["drug2_name"] = df["drug2_id"].map(lambda x: name_map.get(x, x))
    return df


def build_label_map(df: pd.DataFrame) -> dict:
    """Y-integer  →  canonical interaction text (one per label)."""
    label_map = {}
    for _, row in df.drop_duplicates("label").iterrows():
        label_map[int(row["label"])] = row["label_text_template"]
    return label_map


def fill_template(template: str, name1: str, name2: str) -> str:
    return template.replace("#Drug1", name1).replace("#Drug2", name2)


def stratified_split(df: pd.DataFrame, train_ratio: float, seed: int):
    train_df, test_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)



SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs with their molecular structures (SMILES notation), "
    "analyse their pharmacological mechanisms step-by-step and predict "
    "their interaction type."
)

TEACHER_SYSTEM_PROMPT = (
    "You are an expert pharmacologist specialising in drug-drug interactions. "
    "Given two drugs and their known interaction type, explain the "
    "pharmacological mechanisms step-by-step. Discuss each drug's mechanism "
    "of action and how their combination produces the stated interaction. "
    "Structure your reasoning as numbered steps."
)


def _format_example(ex: dict, include_cot: bool = True) -> str:
    """Format a single few-shot example."""
    lines = [
        f"Drug 1: {ex['drug1_name']}",
        f"SMILES: {ex['drug1_smiles']}",
        f"Drug 2: {ex['drug2_name']}",
        f"SMILES: {ex['drug2_smiles']}",
    ]
    if include_cot:
        lines.append("")
        lines.append(ex["cot"])
        lines.append(f"\nClassification: Y={ex['label']} — \"{ex['label_text']}\"")
    return "\n".join(lines)


def build_teacher_prompt(row: pd.Series, label_map: dict) -> str:
    """
    Construct the CoT rationale-generation prompt for the teacher.

    The teacher is given the CORRECT interaction label and asked to explain
    the pharmacological reasoning behind it. This is the standard approach
    for CoT distillation -- the teacher generates reasoning, not predictions.
    """
    parts = []

    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"--- Example {i} ---")
        parts.append(f"Drug 1: {ex['drug1_name']}")
        parts.append(f"SMILES: {ex['drug1_smiles']}")
        parts.append(f"Drug 2: {ex['drug2_name']}")
        parts.append(f"SMILES: {ex['drug2_smiles']}")
        parts.append(f"Known interaction: Y={ex['label']} — \"{ex['label_text']}\"")
        parts.append(f"\n{ex['cot']}")
        parts.append(f"\nClassification: Y={ex['label']} — \"{ex['label_text']}\"")
        parts.append("")

    parts.append("--- Your turn ---")
    parts.append(f"Drug 1: {row['drug1_name']} ({row['drug1_id']})")
    parts.append(f"SMILES: {row['drug1_smiles']}")
    parts.append(f"Drug 2: {row['drug2_name']} ({row['drug2_id']})")
    parts.append(f"SMILES: {row['drug2_smiles']}")
    parts.append(f"Known interaction: Y={row['label']} — \"{row['label_text']}\"")
    parts.append("")
    parts.append(
        "Explain step-by-step the pharmacological mechanisms behind this "
        "drug-drug interaction. Discuss each drug's mechanism of action and "
        "how they combine to produce this effect. End with the classification line."
    )
    return "\n".join(parts)


def build_student_input(row: pd.Series) -> str:
    """Input portion of the student prompt (no answer)."""
    return (
        f"Drug 1: {row['drug1_name']} ({row['drug1_id']})\n"
        f"SMILES: {row['drug1_smiles']}\n"
        f"Drug 2: {row['drug2_name']} ({row['drug2_id']})\n"
        f"SMILES: {row['drug2_smiles']}\n\n"
        "Explain step-by-step why these drugs interact and state the interaction type."
    )


def build_student_target_cot(row: pd.Series) -> str:
    """Target output for CoT distillation training (populated after teacher generation)."""
    return f"{row['teacher_cot']}\n\nClassification: Y={row['label']} — \"{row['label_text']}\""


def build_student_target_label_only(row: pd.Series) -> str:
    """Target output for label-only baseline (Condition B)."""
    return f"Classification: Y={row['label']} — \"{row['label_text']}\""



def prepare_data(cfg: dict):
    logger = setup_logging("data_preparation")
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)

    logger.info("Loading raw DrugBank DDI data …")
    df = load_raw_data(cfg)
    logger.info(f"  Total pairs : {len(df):,}")
    logger.info(f"  Label range : {df['label'].min()} – {df['label'].max()}")
    logger.info(f"  Unique labels: {df['label'].nunique()}")

    name_map = load_name_map(cfg)
    logger.info(f"  Drug-name map covers {len(name_map)} drugs")
    df = resolve_drug_names(df, name_map)

    unique_drugs = set(df["drug1_id"]) | set(df["drug2_id"])
    resolved = sum(1 for d in unique_drugs if d in name_map)
    logger.info(f"  Unique drugs: {len(unique_drugs)}, resolved names: {resolved} "
                f"({100*resolved/len(unique_drugs):.1f}%)")

    label_map = build_label_map(df)
    df["label_text"] = df.apply(
        lambda r: fill_template(label_map[r["label"]], r["drug1_name"], r["drug2_name"]),
        axis=1,
    )
    df["category"] = df["label_text_template"].apply(categorize_interaction)

    logger.info("Stratified train/test split …")
    train_df, test_df = stratified_split(df, cfg["data"]["train_ratio"], cfg["project"]["seed"])
    logger.info(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    out = Path(cfg["data"]["processed_dir"])
    train_df.to_json(out / "train.jsonl", orient="records", lines=True)
    test_df.to_json(out / "test.jsonl", orient="records", lines=True)
    with open(out / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved processed data to {out}")
    return train_df, test_df, label_map
