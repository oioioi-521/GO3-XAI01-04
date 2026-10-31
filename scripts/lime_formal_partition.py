"""Explicit, hash-bound image partitions for the immutable LIME overlay."""
from contextlib import contextmanager
import hashlib
import json


def partition_ids(ids, size):
    ids = list(map(str, ids))
    if size < 1 or not ids or len(ids) != len(set(ids)):
        raise ValueError('partition requires unique nonempty IDs and positive size')
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def select_rows(frame, image_ids):
    ids = list(map(str, image_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate partition IDs')
    indexed = frame.copy()
    indexed['image_id'] = indexed.image_id.astype(str)
    if indexed.image_id.duplicated().any():
        raise ValueError('duplicate metadata IDs')
    indexed = indexed.set_index('image_id', drop=False)
    if not set(ids).issubset(indexed.index):
        raise ValueError('partition IDs outside the frozen split')
    return indexed.loc[ids].reset_index(drop=True)


@contextmanager
def execution_partition(config, implementation):
    """Change process-local routing only; leave frozen source and metadata intact."""
    from preprocessing import dataset
    from experiments import run_unit
    original_split = dataset.get_split
    original_lime = run_unit.SUPPORTED_METHODS['lime']
    selected = select_rows(original_split(config['dataset']['name'], config['dataset']['split']),
                           config['dataset']['image_ids'])
    if config['runtime'].get('max_images') is not None:
        raise ValueError('partition cannot be truncated by max_images')

    def split(name, part):
        if (name, part) != (config['dataset']['name'], config['dataset']['split']):
            raise ValueError('worker requested a different dataset/split')
        return selected.copy()

    dataset.get_split = split
    run_unit.SUPPORTED_METHODS['lime'] = implementation
    try:
        yield selected
    finally:
        dataset.get_split = original_split
        run_unit.SUPPORTED_METHODS['lime'] = original_lime


def summarize(frame, config_hash):
    """Recompute across images; never average shard standard deviations."""
    keys = ['method', 'model', 'dataset', 'metric']
    summary = frame.groupby(keys, sort=True)['value'].agg(mean='mean', std='std', n='count').reset_index()
    summary['std'] = summary['std'].fillna(0.)
    summary['config_hash'] = config_hash
    return summary


def aggregate_hash(config_hashes):
    return hashlib.sha256(json.dumps(sorted(config_hashes), separators=(',', ':')).encode()).hexdigest()[:12]
