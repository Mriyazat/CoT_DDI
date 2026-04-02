"""Check all available XML fields for a sample drug."""
import xml.etree.ElementTree as ET

NS = {'db': 'http://www.drugbank.ca'}
DB_NS = '{http://www.drugbank.ca}'

for event, elem in ET.iterparse('data/raw/drugbank_full.xml', events=('end',)):
    if elem.tag != f'{DB_NS}drug':
        continue
    pk = elem.find(f'db:drugbank-id[@primary="true"]', NS)
    if pk is None or pk.text != 'DB01136':
        elem.clear()
        continue

    print("=== All top-level fields for Carvedilol ===")
    for child in elem:
        tag = child.tag.replace(DB_NS, '')
        text_preview = ''
        if child.text and child.text.strip():
            text_preview = child.text.strip()[:80]
        child_count = len(list(child))
        if text_preview:
            print(f"  <{tag}> = {text_preview}")
        elif child_count > 0:
            print(f"  <{tag}> ({child_count} children)")
        else:
            print(f"  <{tag}> (empty)")

    # Toxicity
    tox = elem.find(f'db:toxicity', NS)
    print(f"\n--- TOXICITY ---")
    print(tox.text[:500] if tox is not None and tox.text else "NONE")

    # Food interactions
    food = elem.findall(f'db:food-interactions/db:food-interaction', NS)
    print(f"\n--- FOOD INTERACTIONS ({len(food)}) ---")
    for fi in food[:3]:
        print(f"  {fi.text[:150] if fi.text else '?'}")

    # Indication
    ind = elem.find(f'db:indication', NS)
    print(f"\n--- INDICATION ---")
    print(ind.text[:400] if ind is not None and ind.text else "NONE")

    # Dosages
    dosages = elem.findall(f'db:dosages/db:dosage', NS)
    print(f"\n--- DOSAGES ({len(dosages)}) ---")
    for d in dosages[:3]:
        form = d.find(f'db:form', NS)
        route = d.find(f'db:route', NS)
        strength = d.find(f'db:strength', NS)
        f_t = form.text if form is not None else '?'
        r_t = route.text if route is not None else '?'
        s_t = strength.text if strength is not None else '?'
        print(f"  {f_t} / {r_t} / {s_t}")

    # SNP adverse reactions
    snp = elem.findall(f'db:snp-adverse-drug-reactions/db:reaction', NS)
    print(f"\n--- SNP ADVERSE DRUG REACTIONS ({len(snp)}) ---")
    for s in snp[:3]:
        desc = s.find(f'db:description', NS)
        print(f"  {desc.text[:150] if desc is not None and desc.text else '?'}")

    # Interaction structure
    interactions = elem.findall(f'db:drug-interactions/db:drug-interaction', NS)
    print(f"\n--- INTERACTION FIELDS (sample of 2) ---")
    for ix in interactions[:2]:
        fields = [c.tag.replace(DB_NS, '') for c in ix]
        print(f"  Fields: {fields}")
        for c in ix:
            t = c.tag.replace(DB_NS, '')
            v = c.text[:120] if c.text else '(empty)'
            print(f"    <{t}> = {v}")
        print()

    # Check if any adverse-effects or severity field exists at drug level
    for tag_name in ['adverse-effects', 'adverse-reactions', 'severity',
                     'clinical-importance', 'contraindications']:
        found = elem.find(f'db:{tag_name}', NS)
        if found is not None:
            txt = found.text[:200] if found.text else f"({len(list(found))} children)"
            print(f"\n--- {tag_name.upper()} ---")
            print(f"  {txt}")

    break
