"""Unified evaluation entry point for segmentation baselines."""
'''
python test_unified.py --all
'''
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
RESULT_DIR = os.path.join(ROOT, 'test_results')
VIS_ROOT = os.path.join(RESULT_DIR, 'visuals')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(VIS_ROOT, exist_ok=True)

_METHOD_PREFIXES = (
    'unet', 'lib', 'networks', 'Constants', 'segment_anything',
    'baselines_unified',
    'uagland', 'uagland_unified',
)


def _clear_root_handlers():
    import logging as _logging
    root = _logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def run_main(method_dir, script_file, argv):
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


_ALL_DATASETS = ['ccg', 'glas', 'pglandseg']
_ALL_METHODS = [
    'uagland',
    'unet',
    'medt',
    'sam',
    'tanet',
]
_datasets = _ALL_DATASETS if dataset_name == 'all' else [dataset_name]
_methods = _ALL_METHODS if methods == 'all' else [methods]

_METHOD_CONFIGS = {
    'uagland': {
        'title': 'UAGlandSeg',
        'table_name': 'UAGlandSeg',
        'method_dir': ROOT,
        'script_file': 'uagland_unified.py',
        'argv': lambda dataset: ['uagland_unified.py',
                                 '--mode', 'test', '--method', 'uagland',
                                 '--dataset', dataset],
    },
    'unet': {
        'title': 'Pytorch-UNet',
        'table_name': 'Pytorch-UNet',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'test', '--method', 'unet',
                                 '--dataset', dataset, '--batch_size', '4',
                                 '--input_size', '256'],
    },
    'medt': {
        'title': 'MedT',
        'table_name': 'MedT',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'test', '--method', 'medt',
                                 '--dataset', dataset, '--batch_size', '4',
                                 '--input_size', '128'],
    },
    'sam': {
        'title': 'Segment-Anything',
        'table_name': 'Segment-Anything',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'test', '--method', 'sam',
                                 '--dataset', dataset, '--batch_size', '1',
                                 '--input_size', '256'],
    },
    'tanet': {
        'title': 'TA-Net',
        'table_name': 'TA-Net',
        'method_dir': ROOT,
        'script_file': 'baselines_unified.py',
        'argv': lambda dataset: ['baselines_unified.py',
                                 '--mode', 'test', '--method', 'tanet',
                                 '--dataset', dataset, '--batch_size', '4',
                                 '--input_size', '256'],
    },
}


def checkpoint_path(method, dataset):
    if method == 'uagland':
        return os.path.join(ROOT, 'Gland_Segmentation', 'UAGlandSeg', 'train_out', dataset, 'uagland', 'checkpoints', 'best.pt')
    return os.path.join(ROOT, {
        'unet': 'Pytorch-UNet-master',
        'medt': 'MedT-main',
        'sam': 'segment-anything-main',
        'tanet': 'TA-Net-master',
    }[method], 'train_out', dataset, 'supervised', 'checkpoint_best.pth')


def parse_metrics(log_text):
    metrics = {}
    patterns = {
        'mDice': r'mDice is ([\d.]+)',
        'mIoU': r'mIoU is ([\d.]+)',
        'mRecall': r'mRecall is ([\d.]+)',
        'Accuracy': r'Accuracy is ([\d.]+)',
        'Gland Dice': r'Dice class gland is ([\d.]+)',
        'Gland IoU': r'IoU class gland is ([\d.]+)',
        'Gland Recall': r'Recall class gland is ([\d.]+)',
        'Background Dice': r'Dice class background is ([\d.]+)',
        'Background IoU': r'IoU class background is ([\d.]+)',
        'Background Recall': r'Recall class background is ([\d.]+)',
    }
    for key, pattern in patterns.items():
        import re
        matches = re.findall(pattern, log_text, flags=re.IGNORECASE)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def format_report(model_name, dataset, metrics):
    lines = []
    width = 78
    lines.append('=' * width)
    lines.append(f'  {model_name.upper()} on {dataset.upper()}')
    lines.append('=' * width)

    overall_keys = ['mDice', 'mIoU', 'mRecall', 'Accuracy']
    if any(key in metrics for key in overall_keys):
        lines.append('  Overall metrics')
        for key in overall_keys:
            if key in metrics:
                lines.append(f'    {key:<12}{metrics[key]:>10.2f}%')
        lines.append('')

    class_rows = [
        ('Background', 'Background Dice', 'Background IoU', 'Background Recall'),
        ('Gland', 'Gland Dice', 'Gland IoU', 'Gland Recall'),
    ]
    if any(label in metrics for _, *labels in class_rows for label in labels):
        lines.append('  Class metrics')
        for title, dice_key, iou_key, recall_key in class_rows:
            if any(key in metrics for key in (dice_key, iou_key, recall_key)):
                dice = metrics.get(dice_key, float('nan'))
                iou = metrics.get(iou_key, float('nan'))
                recall = metrics.get(recall_key, float('nan'))
                lines.append(
                    f'    {title:<11}'
                    f'Dice {dice:>8.2f}%   IoU {iou:>8.2f}%   Recall {recall:>8.2f}%'
                )
        lines.append('')

    lines.append('=' * width)
    return '\n'.join(lines)


_tasks = [(method, dataset) for method in _methods for dataset in _datasets]
print(f'\n[test_unified] {len(_tasks)} task(s): {_methods} on {_datasets}\n')

reports = []
for idx, (_method, _dataset) in enumerate(_tasks, start=1):
    cfg = _METHOD_CONFIGS[_method]
    print(f'\n{"=" * 55}')
    print(f'  [{idx}/{len(_tasks)}] {cfg["title"]} on {_dataset.upper()}')
    print(f'{"=" * 55}\n')

    vis_dir = os.path.join(VIS_ROOT, f'{_method}_{_dataset}')
    ckpt = checkpoint_path(_method, _dataset)
    if not os.path.exists(ckpt):
        print(f'[SKIP] Missing checkpoint: {ckpt}')
        print('       Run train_unified.py first.')
        continue

    saved = []
    class _Capture:
        def write(self, text):
            saved.append(text)
        def flush(self):
            pass

    old_stdout = sys.stdout
    sys.stdout = _Capture()
    try:
        run_main(
            cfg['method_dir'],
            cfg['script_file'],
            cfg['argv'](_dataset) + ['--checkpoint', ckpt, '--vis_dir', vis_dir],
        )
    finally:
        sys.stdout = old_stdout

    log_text = ''.join(saved)
    metrics = parse_metrics(log_text)
    report = format_report(cfg['table_name'], _dataset, metrics)
    reports.append(report)
    print('\n' + report + '\n')
    print(f'[Visualizations] {vis_dir}')

    out_file = os.path.join(RESULT_DIR, f'{_method}_{_dataset}.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(report + '\n\n--- Raw Log ---\n\n')
        f.write(log_text)
    print(f'[Saved] {out_file}')

summary_file = os.path.join(RESULT_DIR, 'summary.txt')
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write('Unified Segmentation Baseline Test Report\n\n')
    for report in reports:
        f.write(report + '\n\n')

print(f'\n[Summary saved] {summary_file}')
for report in reports:
    print(report)
    print()

print('[test_unified] All evaluations finished.')
