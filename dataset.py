
def get_datasets(
        selections: tp.List[tp.Dict[str, tp.Any]],
        n_recordings: int,
        test_ratio: float,
        valid_ratio: float,
        sample_rate: int = studies.schoffelen2019.RAW_SAMPLE_RATE,  # FIXME
        highpass: float = 0,
        num_workers: int = 10,
        apply_baseline: bool = True,
        progress: bool = False,
        skip_recordings: int = 0,
        min_block_duration: float = 0.0,
        force_uid_assignement: bool = True,
        shuffle_recordings_seed: int = -1,
        split_assign_seed: int = 12,
        min_n_blocks_per_split: int = 20,
        features: tp.Optional[tp.List[str]] = None,
        extra_test_features: tp.Optional[tp.List[str]] = None,
        test: dict = {},
        allow_empty_split: bool = False,
        n_subjects: tp.Optional[int] = None,
        n_subjects_test: tp.Optional[int] = None,
        remove_ratio: float = 0.,
        **factory_kwargs: tp.Any) -> Datasets:
    """
    """

    num_workers = max(1, min(n_recordings, num_workers))
    # Use barrier to prevent multiple workers from computing the cache
    # in parallel.
    if not flashy.distrib.is_rank_zero():
        flashy.distrib.barrier()  # type: ignore
    # get recordings
    all_recordings = _extract_recordings(
        selections, n_recordings, skip_recordings=skip_recordings,
        shuffle_recordings_seed=shuffle_recordings_seed)
    if num_workers <= 1:
        if progress:
            all_recordings = LogProgress(logger, all_recordings,  # type: ignore
                                         name="Preparing cache", level=logging.DEBUG)
        all_recordings = [  # for debugging
            _preload(s, sample_rate=sample_rate, highpass=highpass) for s in all_recordings]
    else:
        # precompute slow metadata loading
        with futures.ProcessPoolExecutor(num_workers) as pool:
            jobs = [pool.submit(_preload, s, sample_rate=sample_rate, highpass=highpass)
                    for s in all_recordings]
            if progress:
                jobs = LogProgress(logger, jobs, name="Preparing cache",  # type: ignore
                                   level=logging.DEBUG)
            all_recordings = [j.result() for j in jobs]  # check for exceptions
    if flashy.distrib.is_rank_zero():
        flashy.distrib.barrier()  # type: ignore