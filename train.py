"""Train Sketchy retrieval with per-epoch SwanLab metrics and automatic resume."""
import argparse
from datetime import datetime
import json
import os
import sys
from pathlib import Path
import uuid

from experiments import sketchy

ROOT = Path(__file__).resolve().parent


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected True or False')


def task_name(value):
    values = {'cl': 'cl', 'cl-zs-sbir': 'cl', 'fg': 'fg', 'fg-zs-sbir': 'fg'}
    try:
        return values[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('Use cl / CL-ZS-SBIR or fg / FG-ZS-SBIR')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--epoch', type=int, default=20, help='Total target epochs, including completed epochs')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--swanlab_api_key', default='IV17QULap1Ekc1OF7mN88', help='Alternatively use SWANLAB_API_KEY or saved login')
    p.add_argument('--swanlab_project', default='Sketch_LVM')
    p.add_argument('--dataset', type=str.lower, choices=['sketchy'], default='sketchy')
    p.add_argument('--resume', type=boolean, nargs='?', const=True, default=False)
    p.add_argument('--task', type=task_name, default='cl')
    p.add_argument('--data_dir', default=r'/opt/data/private/data_set/sketch_retrieval/sketchy')
    p.add_argument('--run_dir', type=Path, default=ROOT/'runs'/'train', help='Directory searched for resumable runs')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--eval_batch_size', type=int, default=64)
    p.add_argument('--steps_per_epoch', type=int, default=0, help='0 uses a full epoch sampling budget')
    p.add_argument('--eval_limit', type=int, default=0, help='0 evaluates all test sketches; positive values are for debugging')
    p.add_argument('--samples_per_class', type=int, default=4)
    p.add_argument('--weights', default='ViT-B/32')
    p.add_argument('--swanlab_mode', choices=['online','offline','disabled'], default='online')
    return p


def identity(args):
    return {'dataset':args.dataset, 'data_dir':os.path.normcase(str(Path(args.data_dir).resolve())),
            'task':args.task, 'seed':args.seed, 'project':args.swanlab_project, 'mode':args.swanlab_mode}


def find_checkpoint(run_dir, expected):
    candidates = []
    for path in Path(run_dir).glob('*/session.json'):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        checkpoint = path.parent/'last.pt'
        if metadata['identity'] == expected and checkpoint.is_file():
            candidates.append((checkpoint.stat().st_mtime_ns, checkpoint, metadata))
    if not candidates:
        raise FileNotFoundError(f'No matching checkpoint under {run_dir}.')
    _, checkpoint, metadata = max(candidates,key=lambda item:item[0])
    return checkpoint, metadata


def metric_payload(record, metrics):
    payload = {'epoch':record['epoch'], **{'train/'+k:v for k,v in record.items()
                                        if k not in ('epoch','seconds','steps')}}
    if metrics['task'] == 'cl':
        payload.update({'test/mAP@200':metrics['mAP_200'], 'test/precision@200':metrics['P_200']})
    else:
        payload.update({'test/Acc@1':metrics['top1'], 'test/Acc@5':metrics['top5']})
    return payload


def main(argv=None, tracker=None):
    args = parser().parse_args(argv)
    # SwanLab prints Unicode status icons even when Windows output is redirected.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    # The Windows SDK captures sys.argv as experiment metadata.
    for i, value in enumerate(sys.argv):
        if value.startswith('--swanlab_api_key='):
            sys.argv[i] = '--swanlab_api_key=[REDACTED]'
        elif value == '--swanlab_api_key' and i + 1 < len(sys.argv):
            sys.argv[i + 1] = '[REDACTED]'
    if args.epoch < 1 or args.steps_per_epoch < 0 or args.eval_limit < 0:
        raise ValueError('epoch must be positive; step/evaluation limits must be nonnegative')
    if args.samples_per_class < 2 or args.batch_size % args.samples_per_class or args.batch_size // args.samples_per_class < 2:
        raise ValueError('batch_size must contain >=2 classes with >=2 samples/class and be divisible by samples_per_class')
    if args.workers < 0 or args.eval_batch_size < 1:
        raise ValueError('workers must be nonnegative and eval_batch_size positive')
    if not Path(args.data_dir).is_dir():
        raise FileNotFoundError(args.data_dir)
    expected = identity(args)
    checkpoint = None
    # Do not persist credentials in checkpoints, manifests or SwanLab config.
    safe_config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k != 'swanlab_api_key'}
    if args.resume:
        try:
            checkpoint, metadata = find_checkpoint(args.run_dir, expected)
        except FileNotFoundError as exc:
            print(f'WARNING: {exc} Starting training from scratch.', file=sys.stderr, flush=True)
    if checkpoint is not None:
        state = sketchy.torch.load(checkpoint,map_location='cpu',weights_only=False)
        incompatibilities = []
        if state.get('tracking') != metadata['tracking']:
            incompatibilities.append('Checkpoint and session SwanLab identity mismatch')
        for key in ('batch_size','samples_per_class','weights','steps_per_epoch','eval_limit','workers'):
            if state['args'][key] != getattr(args,key):
                incompatibilities.append(f'{key}: checkpoint={state["args"][key]}, requested={getattr(args,key)}')
        if incompatibilities:
            print(f'WARNING: Incompatible checkpoint {checkpoint}: {"; ".join(incompatibilities)}. '
                  'Starting training from scratch.', file=sys.stderr, flush=True)
            checkpoint = None
            del state
    if checkpoint is not None:
        if args.epoch <= state['epoch']:
            print(f'Checkpoint has completed {state["epoch"]} epochs; set --epoch greater than that to continue.')
            if not state.get('pending_log'):
                return
        output = checkpoint.parent
        print(f'Resuming {checkpoint} after epoch {state["epoch"]}; SwanLab run {metadata["tracking"]["id"]}',flush=True)
        del state
    else:
        run_id = uuid.uuid4().hex[:21]
        output = args.run_dir/f'{args.dataset}_{args.task}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}_{run_id[:8]}'
        output.mkdir(parents=True,exist_ok=False)
        metadata = {'identity':expected, 'tracking':{'id':run_id,'project':args.swanlab_project,'mode':args.swanlab_mode},
                    'config':safe_config}
        (output/'session.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    engine = sketchy.build_parser().parse_args(['--task',args.task,'--output',str(output)])
    for key in ('data_dir','batch_size','seed','workers','eval_batch_size','steps_per_epoch','eval_limit','samples_per_class','weights'):
        setattr(engine,key,getattr(args,key))
    engine.epochs = args.epoch
    engine.checkpoint = str(checkpoint) if checkpoint else None
    if tracker is None:
        import swanlab as tracker
    key = args.swanlab_api_key or os.environ.get('SWANLAB_API_KEY')
    if key and args.swanlab_mode == 'online':
        tracker.login(api_key=key,save=False)
    tracking = metadata['tracking']
    tracker.init(project=tracking['project'],name=output.name,id=tracking['id'],
                 resume='must' if checkpoint is not None else 'allow',mode=args.swanlab_mode,
                 config=safe_config,log_dir=str(output/'swanlog'))
    try:
        return sketchy.run(engine,
                          on_epoch=lambda record,metrics:tracker.log(metric_payload(record,metrics),step=record['epoch']),
                          tracking=tracking)
    finally:
        tracker.finish()


if __name__ == '__main__':
    main()
