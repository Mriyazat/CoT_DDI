"""
Build Morgan fingerprints for all drugs with SMILES and precompute
the drug-level Tanimoto similarity matrix for dynamic few-shot retrieval.

Outputs:
  - drug_fingerprints.pkl: {drugbank_id: numpy array of 2048-bit fingerprint}
  - drug_similarity_matrix.npz: 4628x4628 sparse Tanimoto similarity matrix
  - drug_id_order.json: ordered list of drug IDs matching matrix rows/columns
"""
import json
import pickle
import sys
import numpy as np
from pathlib import Path

PROFILES_PATH = "data/processed/drug_profiles.json"
OUT_DIR = "data/processed"


def compute_fingerprints(profiles, bits=2048, radius=2):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fingerprints = {}
    failed = 0
    no_smiles = 0

    for dbid, profile in profiles.items():
        smiles = profile.get("smiles", "")
        if not smiles:
            no_smiles += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
        fingerprints[dbid] = np.array(fp, dtype=np.uint8)

    print(f"  Fingerprints computed: {len(fingerprints):,}")
    print(f"  No SMILES: {no_smiles:,}")
    print(f"  Invalid SMILES: {failed:,}")
    return fingerprints


def build_similarity_matrix(fingerprints, drug_ids):
    """Build full Tanimoto similarity matrix using vectorized numpy."""
    n = len(drug_ids)
    print(f"  Building {n}x{n} Tanimoto similarity matrix...")

    fp_matrix = np.zeros((n, len(next(iter(fingerprints.values())))), dtype=np.float32)
    for i, dbid in enumerate(drug_ids):
        fp_matrix[i] = fingerprints[dbid]

    dot = fp_matrix @ fp_matrix.T
    norms = np.sum(fp_matrix, axis=1)
    denom = norms[:, None] + norms[None, :] - dot
    denom = np.maximum(denom, 1e-10)
    sim_matrix = dot / denom

    np.fill_diagonal(sim_matrix, 1.0)
    print(f"  Matrix shape: {sim_matrix.shape}")
    print(f"  Mean similarity: {sim_matrix.mean():.4f}")
    print(f"  Mean off-diagonal: {(sim_matrix.sum() - n) / (n*n - n):.4f}")
    return sim_matrix


def main():
    print("=== Building Drug Fingerprints + Similarity Matrix ===\n")

    with open(PROFILES_PATH) as f:
        profiles = json.load(f)
    print(f"Loaded {len(profiles):,} drug profiles")

    fingerprints = compute_fingerprints(profiles)

    drug_ids = sorted(fingerprints.keys())
    sim_matrix = build_similarity_matrix(fingerprints, drug_ids)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(Path(OUT_DIR) / "drug_fingerprints.pkl", "wb") as f:
        pickle.dump(fingerprints, f)
    print(f"\nSaved drug_fingerprints.pkl ({len(fingerprints):,} drugs)")

    np.savez_compressed(
        Path(OUT_DIR) / "drug_similarity_matrix.npz",
        matrix=sim_matrix,
    )
    print(f"Saved drug_similarity_matrix.npz ({sim_matrix.shape})")

    with open(Path(OUT_DIR) / "drug_id_order.json", "w") as f:
        json.dump(drug_ids, f)
    print(f"Saved drug_id_order.json ({len(drug_ids):,} drug IDs)")

    print(f"\nDrug-level similarity stats:")
    print(f"  Drugs with fingerprints: {len(fingerprints):,} / {len(profiles):,} "
          f"({100*len(fingerprints)/len(profiles):.1f}%)")


if __name__ == "__main__":
    main()
