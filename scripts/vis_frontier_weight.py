"""Sanity-check visualization for NCM labels produced by
`SemanticMapPrecomputedDataset` when `DATASET.enable_nav_label=True`.

For one precomputed sample, render a 1x4 panel:
    in_semmap (observed) | semmap (full GT) | in_floor_label | frontier_weight

where  frontier_weight = 1.0 + beta * dilate(frontier_mask, k)
(default beta=2.0, k=15  =>  values in {1.0, 3.0}).

Run from repo root, e.g.::

    python scripts/vis_frontier_weight.py --idx 0
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from poni.default import get_cfg
from poni.dataset import SemanticMapPrecomputedDataset
from poni.constants import OBJECT_CATEGORIES, d3_40_colors_rgb


DEFAULT_DATA_ROOT = "data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square"
DEFAULT_SAVE_PATH = "visualizations/frontier_weight.png"


def colorize_semmap(semmap):
    """(C, H, W) one-hot -> (H, W, 3) uint8, first active channel wins."""
    sm = semmap.float()
    occupied = sm.any(dim=0).numpy()                # (H, W)
    cls = sm.argmax(dim=0).numpy()                  # (H, W)
    colors = d3_40_colors_rgb[: semmap.shape[0]]    # (C, 3)
    img = colors[cls]                               # (H, W, 3)
    img[~occupied] = 255                            # white = unoccupied
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx", type=int, default=0,
                        help="index into the precomputed dataset")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="path to precomputed_dataset_* directory")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.defrost()
    cfg.DATASET.root = args.data_root
    cfg.DATASET.enable_nav_label = True
    cfg.DATASET.nav_loss_explored_only = True
    cfg.DATASET.nav_frontier_weight_beta = 2
    cfg.DATASET.nav_frontier_dilate_k = 15
    cfg.freeze()

    dataset = SemanticMapPrecomputedDataset(cfg.DATASET, split=args.split)
    inputs, labels = dataset[args.idx]

    in_semmap = inputs["semmap"]                # (C, H, W) observed one-hot
    semmap = labels["semmap"]                   # (C, H, W) full GT one-hot
    frontier_weight = labels["frontier_weight"] # (H, W), in {1.0, 3.0} by default
    in_floor_label = labels["in_floor_label"]   # (H, W), {0, 1}

    beta = cfg.DATASET.nav_frontier_weight_beta
    k = cfg.DATASET.nav_frontier_dilate_k
    print(f"beta={beta}, k={k}")
    print(f"frontier_weight: shape={tuple(frontier_weight.shape)}, "
          f"unique={frontier_weight.unique().tolist()}, "
          f"frontier-pixel ratio={(frontier_weight > 1).float().mean().item():.4f}")

    cats = OBJECT_CATEGORIES[cfg.DATASET.dset_name]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(colorize_semmap(in_semmap))
    axes[0].set_title("in_semmap (observed)")
    axes[0].axis("off")

    axes[1].imshow(colorize_semmap(semmap))
    axes[1].set_title("semmap (full GT)")
    axes[1].axis("off")

    axes[2].imshow(in_floor_label, cmap="Greens", vmin=0, vmax=1)
    axes[2].set_title("in_floor_label (GT floor)")
    axes[2].axis("off")

    im = axes[3].imshow(frontier_weight, cmap="hot", vmin=0.0, vmax=1.0 + beta)
    axes[3].set_title(f"frontier_weight  (beta={beta}, k={k})")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    cmap = ListedColormap(np.asarray(d3_40_colors_rgb[: len(cats)]) / 255.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(cats))]
    fig.legend(handles, cats, loc="lower center", ncol=len(cats),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.save_path}")


if __name__ == "__main__":
    main()
