from hashlib import sha1
import logging
import os
from pathlib import Path
import typing as tp
import sys

from dora import hydra_main, to_absolute_path
import flashy
import hydra
import mne
from omegaconf import OmegaConf
import torch



def get_solver(args: tp.Any, training=True):
    # Dataset and loading
    assert args.optim.batch_size % flashy.distrib.world_size() == 0
    logger.debug("Building datasets")
    args.optim.batch_size //= flashy.distrib.world_size()

    if args.num_workers is None:
        args.num_workers = min(10, 2 * args.optim.batch_size)

    assert args.dset.sample_rate is not None, "sample rate <= 1200 required"
    kwargs: tp.Dict[str, tp.Any]
    kwargs = OmegaConf.to_container(args.dset, resolve=True)  # type: ignore
    selections = [args.selections[x] for x in args.dset.selections]
    kwargs["selections"] = selections
    if args.optim.loss == "clip":
        kwargs['extra_test_features'].append("WordHash")

    dsets = dset.get_datasets(
        num_workers=args.num_workers, progress=True,
        **kwargs,
    )



def run(args: tp.Any) -> float:


    mne.set_log_level(False)
    solver = get_solver(args)



def main(args: tp.Any) -> float:


        return run(args)

if __name__ == "__main__":
    main()