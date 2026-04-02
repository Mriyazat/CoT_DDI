"""
Extract the full DDI dataset, drug profiles, and DDInter severity labels
from DrugBank XML v5.1.14 and DDInter 2.0 CSVs.

Three extraction passes on the XML:
  Pass 1: Collect all drug names + synonyms by ID
  Pass 2: Extract all DDI pairs with interaction text
  Pass 3: Extract pharmacological profiles for interacting drugs

Then cross-reference DDInter severity labels via synonym resolution.

No dependency on v2 data. Pure extraction from authoritative sources.
"""
import xml.etree.ElementTree as ET
import json
import os
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

XML_PATH = "data/raw/drugbank_full.xml"
DDINTER_DIR = "data/raw/ddinter"
OUT_DIR = "data/processed"
NS = {"db": "http://www.drugbank.ca"}


def extract_template(text, drug1_name, drug2_name):
    t = text.strip()
    t1 = t.replace(drug1_name, "#Drug1").replace(drug2_name, "#Drug2")
    t2 = t.replace(drug2_name, "#Drug1").replace(drug1_name, "#Drug2")

    pos1_d1 = t1.find("#Drug1")
    pos1_d2 = t1.find("#Drug2")
    pos2_d1 = t2.find("#Drug1")
    pos2_d2 = t2.find("#Drug2")

    if pos1_d1 != -1 and pos1_d2 != -1 and pos1_d1 < pos1_d2:
        return t1, drug1_name, drug2_name
    elif pos2_d1 != -1 and pos2_d2 != -1 and pos2_d1 < pos2_d2:
        return t2, drug2_name, drug1_name
    elif pos1_d1 != -1 and pos1_d2 != -1:
        return t1, drug1_name, drug2_name
    elif pos2_d1 != -1 and pos2_d2 != -1:
        return t2, drug2_name, drug1_name
    else:
        return None, drug1_name, drug2_name


def _get_text(elem, xpath):
    el = elem.find(xpath, NS)
    if el is not None and el.text:
        return el.text.strip()
    return ""


def _extract_polypeptide_targets(elem, tag):
    """Extract targets/enzymes/transporters with their actions."""
    results = []
    for entry in elem.findall(f"db:{tag}/db:{tag[:-1]}", NS):
        name_el = entry.find("db:name", NS)
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        actions = []
        for a in entry.findall("db:actions/db:action", NS):
            if a.text:
                actions.append(a.text.strip().lower())
        pp = entry.find("db:polypeptide", NS)
        gene = ""
        if pp is not None:
            gene_el = pp.find("db:gene-name", NS)
            if gene_el is not None and gene_el.text:
                gene = gene_el.text.strip()
        entry_str = name
        if gene:
            entry_str += f" ({gene})"
        if actions:
            entry_str += f": {', '.join(actions)}"
        results.append(entry_str)
    return results


def pass1_names_and_synonyms():
    """Collect all drug names, synonyms, and secondary IDs."""
    print("=== Pass 1: Collecting drug names + synonyms ===")
    drug_names = {}
    drug_synonyms = defaultdict(set)  # name_lower -> set of DrugBank IDs
    drug_count = 0

    for event, elem in ET.iterparse(XML_PATH, events=("end",)):
        if elem.tag != f'{{{NS["db"]}}}drug':
            continue
        pk = elem.find('db:drugbank-id[@primary="true"]', NS)
        if pk is None:
            elem.clear()
            continue
        dbid = pk.text
        name_elem = elem.find("db:name", NS)
        if name_elem is not None and name_elem.text:
            official_name = name_elem.text.strip()
            drug_names[dbid] = official_name
            drug_synonyms[official_name.lower()].add(dbid)

        for syn_el in elem.findall("db:synonyms/db:synonym", NS):
            if syn_el.text:
                drug_synonyms[syn_el.text.strip().lower()].add(dbid)

        for prod_el in elem.findall("db:international-brands/db:international-brand/db:name", NS):
            if prod_el.text:
                drug_synonyms[prod_el.text.strip().lower()].add(dbid)

        drug_count += 1
        if drug_count % 5000 == 0:
            print(f"  Pass 1: {drug_count:,} drugs...", file=sys.stderr)
        elem.clear()

    n_synonyms = sum(len(v) for v in drug_synonyms.values())
    print(f"  Total drugs: {len(drug_names):,}")
    print(f"  Synonym entries: {len(drug_synonyms):,} names -> {n_synonyms:,} mappings")
    return drug_names, drug_synonyms


def pass2_interactions(drug_names):
    """Extract all DDI pairs with interaction text and templates."""
    print("\n=== Pass 2: Extracting interactions ===")
    interactions = []
    template_counter = Counter()
    failed_template = 0
    drug_count = 0

    for event, elem in ET.iterparse(XML_PATH, events=("end",)):
        if elem.tag != f'{{{NS["db"]}}}drug':
            continue
        pk = elem.find('db:drugbank-id[@primary="true"]', NS)
        if pk is None:
            elem.clear()
            continue

        source_id = pk.text
        source_name = drug_names.get(source_id, "?")
        drug_count += 1

        for ix in elem.findall("db:drug-interactions/db:drug-interaction", NS):
            partner_id_elem = ix.find("db:drugbank-id", NS)
            desc_elem = ix.find("db:description", NS)
            if partner_id_elem is None or desc_elem is None or not desc_elem.text:
                continue

            partner_id = partner_id_elem.text
            partner_name_elem = ix.find("db:name", NS)
            partner_name = (partner_name_elem.text if partner_name_elem is not None
                            else drug_names.get(partner_id, "?"))
            text = desc_elem.text.strip()

            template, d1_name, d2_name = extract_template(text, source_name, partner_name)
            if template is None:
                failed_template += 1
                continue

            template_counter[template] += 1
            interactions.append({
                "drug1_id": source_id,
                "drug2_id": partner_id,
                "drug1_name": d1_name,
                "drug2_name": d2_name,
                "text": text,
                "template": template,
            })

        if drug_count % 2000 == 0:
            print(f"  Pass 2: {drug_count:,} drugs, {len(interactions):,} interactions...",
                  file=sys.stderr)
        elem.clear()

    print(f"  Raw interactions: {len(interactions):,}")
    print(f"  Failed templates: {failed_template:,}")
    print(f"  Unique templates: {len(template_counter):,}")

    seen_pairs = set()
    unique_interactions = []
    for ix in interactions:
        pair_key = tuple(sorted([ix["drug1_id"], ix["drug2_id"]])) + (ix["template"],)
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_interactions.append(ix)

    print(f"  After dedup: {len(unique_interactions):,}")

    involved_drugs = set()
    for ix in unique_interactions:
        involved_drugs.add(ix["drug1_id"])
        involved_drugs.add(ix["drug2_id"])
    print(f"  Unique drugs in interactions: {len(involved_drugs):,}")

    label_map = {}
    template_to_label = {}
    for i, (template, count) in enumerate(template_counter.most_common(), start=1):
        label_map[i] = template
        template_to_label[template] = i

    for ix in unique_interactions:
        ix["label"] = template_to_label[ix["template"]]

    return unique_interactions, label_map, involved_drugs


def pass3_drug_profiles(involved_drugs):
    """Extract pharmacological profiles for all interacting drugs."""
    print(f"\n=== Pass 3: Extracting drug profiles for {len(involved_drugs):,} drugs ===")
    profiles = {}
    drug_count = 0

    for event, elem in ET.iterparse(XML_PATH, events=("end",)):
        if elem.tag != f'{{{NS["db"]}}}drug':
            continue
        pk = elem.find('db:drugbank-id[@primary="true"]', NS)
        if pk is None:
            elem.clear()
            continue
        dbid = pk.text
        if dbid not in involved_drugs:
            elem.clear()
            continue

        drug_count += 1
        name = _get_text(elem, "db:name")
        description = _get_text(elem, "db:description")
        if description and len(description) > 500:
            description = description[:497] + "..."
        mechanism = _get_text(elem, "db:mechanism-of-action")
        if mechanism and len(mechanism) > 500:
            mechanism = mechanism[:497] + "..."
        pharmacodynamics = _get_text(elem, "db:pharmacodynamics")

        categories = []
        for cat in elem.findall("db:categories/db:category/db:category", NS):
            if cat.text:
                categories.append(cat.text.strip())

        targets = _extract_polypeptide_targets(elem, "targets")
        enzymes = _extract_polypeptide_targets(elem, "enzymes")
        transporters = _extract_polypeptide_targets(elem, "transporters")

        smiles = ""
        for prop in elem.findall("db:calculated-properties/db:property", NS):
            kind_el = prop.find("db:kind", NS)
            val_el = prop.find("db:value", NS)
            if (kind_el is not None and val_el is not None
                    and kind_el.text == "SMILES" and val_el.text):
                smiles = val_el.text.strip()
                break

        toxicity = _get_text(elem, "db:toxicity")
        metabolism = _get_text(elem, "db:metabolism")

        profiles[dbid] = {
            "drugbank_id": dbid,
            "name": name,
            "description": description,
            "mechanism_of_action": mechanism,
            "pharmacodynamics": pharmacodynamics[:300] if pharmacodynamics else "",
            "categories": categories[:5],
            "targets": targets[:10],
            "enzymes": enzymes[:10],
            "transporters": transporters[:5],
            "smiles": smiles,
            "toxicity": toxicity[:300] if toxicity else "",
            "metabolism": metabolism[:300] if metabolism else "",
        }

        if drug_count % 1000 == 0:
            print(f"  Pass 3: {drug_count:,} profiles extracted...", file=sys.stderr)
        elem.clear()

    n_desc = sum(1 for p in profiles.values() if p["description"])
    n_mech = sum(1 for p in profiles.values() if p["mechanism_of_action"])
    n_smiles = sum(1 for p in profiles.values() if p["smiles"])
    n_enz = sum(1 for p in profiles.values() if p["enzymes"])
    n_tgt = sum(1 for p in profiles.values() if p["targets"])
    n_trans = sum(1 for p in profiles.values() if p["transporters"])

    print(f"  Profiles extracted: {len(profiles):,} / {len(involved_drugs):,}")
    print(f"  Coverage: description={100*n_desc/len(profiles):.1f}%, "
          f"mechanism={100*n_mech/len(profiles):.1f}%, "
          f"smiles={100*n_smiles/len(profiles):.1f}%")
    print(f"  Enzymes={100*n_enz/len(profiles):.1f}%, "
          f"targets={100*n_tgt/len(profiles):.1f}%, "
          f"transporters={100*n_trans/len(profiles):.1f}%")
    return profiles


def load_ddinter_severity(drug_synonyms):
    """Load DDInter CSVs and match drug names to DrugBank IDs via synonyms."""
    print(f"\n=== DDInter Severity Cross-Reference ===")
    ddinter_pairs = []
    csv_files = sorted(Path(DDINTER_DIR).glob("ddinter_code_*.csv"))
    if not csv_files:
        print(f"  WARNING: No DDInter CSV files found in {DDINTER_DIR}")
        return {}

    for csv_path in csv_files:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ddinter_pairs.append({
                    "drug_a": row["Drug_A"].strip(),
                    "drug_b": row["Drug_B"].strip(),
                    "level": row["Level"].strip(),
                })

    print(f"  Total DDInter rows: {len(ddinter_pairs):,}")

    def resolve_name(name):
        name_lower = name.lower()
        ids = drug_synonyms.get(name_lower, set())
        if len(ids) == 1:
            return next(iter(ids))
        return None

    severity_map = {}
    matched = 0
    unmatched_drugs = set()

    for pair in ddinter_pairs:
        id_a = resolve_name(pair["drug_a"])
        id_b = resolve_name(pair["drug_b"])
        if id_a and id_b and id_a != id_b:
            pair_key = tuple(sorted([id_a, id_b]))
            severity_map[f"{pair_key[0]}_{pair_key[1]}"] = pair["level"]
            matched += 1
        else:
            if not id_a:
                unmatched_drugs.add(pair["drug_a"])
            if not id_b:
                unmatched_drugs.add(pair["drug_b"])

    level_counts = Counter(severity_map.values())
    print(f"  Matched pairs: {matched:,} -> {len(severity_map):,} unique")
    print(f"  Unmatched drug names: {len(unmatched_drugs):,}")
    print(f"  Severity distribution: {dict(level_counts)}")
    return severity_map


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'='*70}")
    print(f"DrugBank XML Full Extraction Pipeline")
    print(f"{'='*70}\n")

    drug_names, drug_synonyms = pass1_names_and_synonyms()
    interactions, label_map, involved_drugs = pass2_interactions(drug_names)
    profiles = pass3_drug_profiles(involved_drugs)
    severity_map = load_ddinter_severity(drug_synonyms)

    # Save all outputs
    with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nSaved label_map.json ({len(label_map)} classes)")

    with open(os.path.join(OUT_DIR, "interactions_full.jsonl"), "w") as f:
        for ix in interactions:
            f.write(json.dumps(ix) + "\n")
    print(f"Saved interactions_full.jsonl ({len(interactions):,} pairs)")

    with open(os.path.join(OUT_DIR, "drug_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved drug_profiles.json ({len(profiles):,} profiles)")

    with open(os.path.join(OUT_DIR, "severity_map.json"), "w") as f:
        json.dump(severity_map, f, indent=2)
    print(f"Saved severity_map.json ({len(severity_map):,} pairs with severity)")

    with open(os.path.join(OUT_DIR, "drug_synonyms.json"), "w") as f:
        syn_serializable = {k: sorted(v) for k, v in drug_synonyms.items()}
        json.dump(syn_serializable, f)
    print(f"Saved drug_synonyms.json ({len(drug_synonyms):,} synonym entries)")

    label_counts = Counter(ix["label"] for ix in interactions)
    sizes = sorted(label_counts.values(), reverse=True)
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total pairs:       {len(interactions):,}")
    print(f"Total classes:     {len(label_map)}")
    print(f"Drugs profiled:    {len(profiles):,}")
    print(f"Severity labels:   {len(severity_map):,}")
    print(f"Largest class:     {sizes[0]:,}")
    print(f"Smallest class:    {sizes[-1]:,}")
    print(f"Classes >= 100:    {sum(1 for s in sizes if s >= 100)}")


if __name__ == "__main__":
    main()
