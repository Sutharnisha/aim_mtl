"""
Download QM9 and verify the full Uni-Mol pipeline end-to-end.
Run once before any training:

    conda activate aim_unimol
    cd Unimol/src
    python download_data.py
"""

import torch
from data import inspect_qm9, get_loaders, TASK_NAMES
from unimol_collate import UniMolCollator, get_symbol_to_idx, vocab_size


def check_unimol_weights():
    """Trigger pretrained Uni-Mol weight download and verify model loads."""
    print("\n[1/4] Loading pretrained Uni-Mol weights...")
    from unimol_tools import UniMolRepr
    repr_engine = UniMolRepr(data_type="molecule", remove_hs=False)
    model_obj   = getattr(repr_engine, "model", None) or getattr(repr_engine, "encoder", None)
    if model_obj is None:
        raise RuntimeError(
            "Cannot access nn.Module from UniMolRepr. "
            "Check your unimol_tools version."
        )
    n_params = sum(p.numel() for p in model_obj.parameters())
    print(f"   Uni-Mol backbone loaded: {n_params:,} parameters")


def check_vocabulary():
    """Verify atom vocabulary loads correctly."""
    print("\n[2/4] Checking atom vocabulary...")
    sym2idx = get_symbol_to_idx()
    vsize   = vocab_size()
    print(f"   Vocabulary size : {vsize}")
    for elem in ("H", "C", "N", "O", "F"):
        print(f"   {elem:3s} → token index {sym2idx.get(elem, '?')}")


def check_qm9():
    """Download QM9 and inspect raw structure."""
    print("\n[3/4] Downloading / verifying QM9 dataset...")
    dataset = inspect_qm9(root="../../data/qm9")
    return dataset


def check_pipeline():
    """Build loaders and verify one full batch through the Uni-Mol collator."""
    print("\n[4/4] Verifying data pipeline (500 split, batch_size=4)...")
    loaders = get_loaders(root="../../data/qm9", n_train=500, batch_size=4, seed=42)

    print(f"   Split sizes : {loaders['split_sizes']}")
    print(f"   Means (11)  : {loaders['means'].tolist()}")
    print(f"   Stds  (11)  : {loaders['stds'].tolist()}")

    batch = next(iter(loaders["primary_loader"]))
    print("\n   Batch tensor shapes:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"     {key:18s}: {tuple(val.shape)}  dtype={val.dtype}")

    # Verify shapes are consistent
    B = batch["src_tokens"].shape[0]
    T = batch["src_tokens"].shape[1]
    assert batch["src_coord"].shape     == (B, T, 3),    "src_coord shape mismatch"
    assert batch["src_distance"].shape  == (B, T, T),    "src_distance shape mismatch"
    assert batch["src_edge_type"].shape == (B, T, T),    "src_edge_type shape mismatch"
    assert batch["targets"].shape       == (B, 11),      "targets shape mismatch"
    print("\n   All shape checks passed.")

    # Task names check
    print(f"\n   11 tasks : {TASK_NAMES}")


def check_full_model_forward():
    """One forward pass through the full Uni-Mol multi-task model."""
    print("\n[BONUS] Testing full model forward pass on one batch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    from model import build_model
    from unimol_collate import batch_to_device

    model   = build_model(device=device)
    loaders = get_loaders(root="../../data/qm9", n_train=500, batch_size=4)
    batch   = next(iter(loaders["primary_loader"]))
    batch   = batch_to_device(batch, device)

    model.eval()
    with torch.no_grad():
        preds = model(batch)   # [B, 11]
    print(f"   Predictions shape : {tuple(preds.shape)}  ✓")
    print(f"   Sample preds      : {preds[0].tolist()}")


if __name__ == "__main__":
    check_unimol_weights()
    check_vocabulary()
    check_qm9()
    check_pipeline()

    run_model_check = input(
        "\nRun full model forward pass? (needs GPU for Uni-Mol) [y/N]: "
    ).strip().lower()
    if run_model_check == "y":
        check_full_model_forward()

    print("\n" + "="*55)
    print("  All checks passed. Ready to train!")
    print("  Next step:")
    print("    python train.py --method ls --n_train 10000 --n_epochs 5")
    print("="*55)
