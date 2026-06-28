"""Unified training entry point for segmentation baselines."""
'''
python train_unified.py
python train_unified.py --force-refit-prototypes
python test_unified.py
'''
import argparse
import os
import runpy
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('XFORMERS_DISABLED', '1')

if sys.platform == 'win32':
    _torch_lib = os.path.join(os.path.dirname(sys.executable),
                              'Lib', 'site-packages', 'torch', 'lib')
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
    _cuda_bin = os.path.join(
        os.environ.get('CUDA_PATH',
                       r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8'),
        'bin')
    if os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)

import torch  # noqa: E402,F401


# User configuration.
dataset_name = 'glas'   # 'ccg' | 'glas' | 'pglandseg' | 'all'
methods = 'uagland'         # 'uagland' | 'unet' | 'medt' | 'sam' | 'tanet' | 'all'


ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='Unified training entry point for segmentation baselines.')
    parser.add_argument('--force-refit-prototypes', action='store_true',
                        help='For UAGlandSeg, ignore an existing prototype_model.joblib and fit prototypes again.')
    return parser.parse_args()


ARGS = parse_args()

_METHOD_PREFIXES = (
    'unet', 'lib', 'networks', 'Constants', 'segment_anything',
    'baselines_unified',
    'uagland', 'uagland_unified',
)


def _clear_root_handlers():
    """Remove and close all root logger handlers before runpy handoff."""
    import logging as _logging
    root = _logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def run_main(method_dir, script_file, argv):
    """Run a method script as __main__, equivalent to ``python script.py ...``."""
    saved_path = sys.path[:]
    saved_argv = sys.argv[:]
    saved_cwd = os.getcwd()
    _clear_root_handlers()

    try:
        sys.path = [method_dir] + [p for p in sys.path if p != method_dir]
        sys.argv = argv
        os.chdir(method_dir)
        runpy.run_path(os.path.join(method_dir, script_file), run_name='__main__')
    finally:
        sys.path = saved_path
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        for key in list(sys.modules.keys()):
            if any(key == prefix or key.startswith(prefix + '.')
                   for prefix in _METHOD_PREFIXES):
                del sys.modules[key]
        _clear_root_handlers()


def uagland_argv(dataset):
    argv = ['uagland_unified.py',
            '--mode', 'train', '--method', 'uagland',
            '--dataset', dataset, '--epochs', '60',
            '--batch_size', '4', '--crop_size', '512',
            '--rounds', '1', '--workers', '0',
            '--encoder', 'dinov2',
            '--dinov2-model', 'dinov2_vits14',
            '--dinov2-local-repo', os.path.join(ROOT, 'Gland_Segmentation', 'dinov2-main'),
            '--dinov2-weights', os.path.join(ROOT, 'pretrained', 'dinov2_vits14_pretrain.pth')]
    if ARGS.force_refit_prototypes:
        argv.append('--force-refit-prototypes')
    return argv


_ALL_DATASETS = ['glas', 'ccg', 'pglandseg']
_ALL_METHODS = [
    'uagland',
    'unet',
    'medt',
    'tanet',
    'sam',
]
_datasets = _ALL_DATASETS if dataset_name == 'all' else [dataset_name]
_methods = _ALL_METHODS if methods == 'all' else [methods]

_METHOD_CONFIGS = {
    'uagland': {
        'title': 'UAGlandSeg',
        'method_dir': ROOT,
        'script_file': 'uagland_unified.py',
        'argv': uagland_argv,
    },
    'unet': {
        'title': 'Pytorch-UNet',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'train', '--method', 'unet',
                                 '--dataset', dataset, '--epochs', '200',
                                 '--batch_size', '4', '--input_size', '256',
                                 '--val_interval', '5'],
    },
    'medt': {
        'title': 'MedT',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'train', '--method', 'medt',
                                 '--dataset', dataset, '--epochs', '200',
                                 '--batch_size', '4', '--input_size', '128',
                                 '--val_interval', '5'],
    },
    'tanet': {
        'title': 'TA-Net',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'train', '--method', 'tanet',
                                 '--dataset', dataset, '--epochs', '200',
                                 '--batch_size', '4', '--input_size', '256',
                                 '--val_interval', '5'],
    },
    'sam': {
        'title': 'Segment-Anything',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'train', '--method', 'sam',
                                 '--dataset', dataset, '--epochs', '30',
                                 '--batch_size', '1', '--input_size', '256',
                                 '--val_interval', '2'],
    },
}

_tasks = [(method, dataset) for method in _methods for dataset in _datasets]
print(f'\n[train_unified] {len(_tasks)} task(s): {_methods} on {_datasets}\n')

for idx, (_method, _dataset) in enumerate(_tasks, start=1):
    cfg = _METHOD_CONFIGS[_method]
    print(f'\n{"=" * 55}')
    print(f'  [{idx}/{len(_tasks)}] {cfg["title"]} on {_dataset.upper()}')
    print(f'{"=" * 55}\n')
    run_main(
        method_dir=cfg['method_dir'],
        script_file=cfg['script_file'],
        argv=cfg['argv'](_dataset),
    )

print('\n[train_unified] All tasks finished.')
