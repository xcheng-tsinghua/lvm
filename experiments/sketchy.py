"""Reproducible native-PyTorch Sketchy training and exhaustive retrieval evaluation.
FG additions are a paper-guided reimplementation, not released author code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from src.clip import clip
from src.dataset_retrieval import unseen_classes


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inventory(root):
    root = Path(root)
    sketch_root = root / ('sketch_png' if (root / 'sketch_png').exists() else 'sketch')
    cats = sorted(p.name for p in sketch_root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not set(unseen_classes).issubset(cats):
        raise ValueError('Missing unseen categories')
    photos = {c: sorted((root / 'photo' / c).glob('*.jpg')) for c in cats}
    sketches = {c: sorted((sketch_root / c).glob('*.png')) for c in cats}
    lookup = {c: {p.stem: p for p in photos[c]} for c in cats}
    missing = [str(s) for c in cats for s in sketches[c] if s.stem.rsplit('-', 1)[0] not in lookup[c]]
    if missing:
        raise ValueError(f'{len(missing)} sketches lack paired photos: {missing[:5]}')
    if any(not photos[c] or not sketches[c] for c in cats):
        raise ValueError('Empty category')
    return cats, photos, sketches, lookup


TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def read_image(path):
    with Image.open(path) as im:
        return TRANSFORM(ImageOps.pad(im.convert('RGB'), (224, 224)))


class Triplets(Dataset):
    def __init__(self, cats, photos, sketches, lookup, task):
        self.cats, self.photos, self.lookup, self.task = cats, photos, lookup, task
        self.rows = [(s, c) for c in cats for s in sketches[c]]
        self.groups = [[i for i, (_, label) in enumerate(self.rows) if label == c] for c in cats]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        s, c = self.rows[index]
        label = self.cats.index(c)
        if self.task == 'fg':
            p = self.lookup[c][s.stem.rsplit('-', 1)[0]]
            n = random.choice(self.photos[c])
            while n == p:
                n = random.choice(self.photos[c])
        else:
            p = random.choice(self.photos[c])
            other = random.choice([x for x in self.cats if x != c])
            n = random.choice(self.photos[other])
        return read_image(s), read_image(p), read_image(n), label


class BalancedBatches(Sampler):
    """Equal samples/category make the paper's batchwise KL distributions compatible."""
    def __init__(self, dataset, batch_size, samples_per_class, steps):
        if batch_size % samples_per_class or samples_per_class < 2 or batch_size // samples_per_class < 2:
            raise ValueError('Require >=2 classes, >=2 samples/class and divisible batch size')
        self.groups = dataset.groups
        self.k, self.n, self.steps = samples_per_class, batch_size // samples_per_class, steps
        if self.n > len(self.groups):
            raise ValueError('Too many classes per batch')

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            yield [i for g in random.sample(self.groups, self.n) for i in random.sample(g, self.k)]


class RetrievalModel(nn.Module):
    def __init__(self, task, weights, prompts=3):
        super().__init__()
        self.clip, _ = clip.load(weights, device='cpu')
        self.clip.requires_grad_(False)
        self.task = task
        # Separate visual LayerNorm parameters for CL; frozen backbone weights stay shared.
        self.sk_lns = nn.ModuleDict()
        for name, layer in self.clip.visual.named_modules():
            if isinstance(layer, nn.LayerNorm):
                layer.requires_grad_(True)
                if task == 'cl':
                    self.sk_lns[name.replace('.', '__')] = copy.deepcopy(layer)
        self.img_prompt = nn.Parameter(torch.randn(prompts, 768))
        self.sk_prompt = nn.Parameter(torch.randn(prompts, 768)) if task == 'cl' else None

    def forward(self, x, sketch=False):
        prompt = self.sk_prompt if sketch and self.task == 'cl' else self.img_prompt
        if sketch and self.task == 'cl':
            params = {name.replace('__', '.') + '.' + pn: p
                      for name, layer in self.sk_lns.items() for pn, p in layer.named_parameters()}
            feat = torch.func.functional_call(self.clip.visual, params, (x, prompt.expand(len(x), -1, -1)), strict=False)
        else:
            feat = self.clip.encode_image(x, prompt.expand(len(x), -1, -1))
        return F.normalize(feat.float(), dim=-1)

    def adapter(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}


def triplet(s, p, n, margin):
    return F.relu(margin + (s*n).sum(-1) - (s*p).sum(-1)).mean()


def relative_kl(s, p, n, labels):
    delta = (s*n).sum(-1) - (s*p).sum(-1)
    distributions = [F.log_softmax(delta[labels == c], dim=0) for c in labels.unique()]
    if len(distributions) < 2 or len({len(x) for x in distributions}) != 1:
        raise ValueError('KL requires balanced multiple-category batches')
    return torch.stack([(a.exp() * (a-b)).sum() for i, a in enumerate(distributions)
                        for j, b in enumerate(distributions) if i != j]).mean()


def shuffle_patches(x, order):
    b, c, h, w = x.shape
    patches = x.reshape(b, c, 2, h//2, 2, w//2).permute(0, 2, 4, 1, 3, 5).reshape(b, 4, c, h//2, w//2)
    patches = patches[torch.arange(b, device=x.device)[:, None], order]
    return patches.reshape(b, 2, 2, c, h//2, w//2).permute(0, 3, 1, 4, 2, 5).reshape_as(x)


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return read_image(self.paths[i])


@torch.inference_mode()
def encode(model, paths, sketch, args):
    result = []
    loader = DataLoader(Images(paths), batch_size=args.eval_batch_size, num_workers=args.workers)
    with tqdm(loader, desc=f'Encode {"sketch" if sketch else "photo"}',
              unit='batch', dynamic_ncols=True) as progress:
        for x in progress:
            with torch.autocast('cuda', enabled=args.device.startswith('cuda')):
                result.append(model(x.to(args.device), sketch).cpu())
    return torch.cat(result)


@torch.inference_mode()
def evaluate(model, cats, photos, sketches, args, writer=None):
    model.eval()
    gallery = [p for c in cats for p in photos[c]]
    queries = [s for c in cats for s in sketches[c]]
    if args.eval_limit:
        queries = [s for c in cats for s in sketches[c][:args.eval_limit]]
    gf = encode(model, gallery, False, args)
    qf = encode(model, queries, True, args)
    gc = np.array([p.parent.name for p in gallery])
    qc = np.array([p.parent.name for p in queries])
    metrics = {'queries': len(queries), 'gallery': len(gallery), 'test_classes': cats,
               'limited_queries_per_class': args.eval_limit, 'task': args.task}
    per_class = {}
    if args.task == 'cl':
        ap, ap200, precision = [], [], []
        for start in range(0, len(queries), 128):
            order = torch.argsort(qf[start:start+128] @ gf.T, descending=True, stable=True).numpy()
            rel = torch.from_numpy(gc[order] == qc[start:start+len(order), None]).float()
            pr = rel.cumsum(1) / torch.arange(1, len(gallery)+1)
            total = rel.sum(1)
            k = min(200, len(gallery))
            ap.extend(((pr*rel).sum(1)/total).tolist())
            # Truncated AP uses min(number of relevant gallery items, K).
            ap200.extend(((pr[:, :k]*rel[:, :k]).sum(1)/total.clamp(max=k)).tolist())
            precision.extend(rel[:, :k].mean(1).tolist())
        metrics.update(mAP_all=float(np.mean(ap)), mAP_200=float(np.mean(ap200)), P_200=float(np.mean(precision)))
    else:
        all_ranks = []
        for c in cats:
            qi, gi = np.where(qc == c)[0], np.where(gc == c)[0]
            order = torch.argsort(qf[qi] @ gf[gi].T, descending=True, stable=True).numpy()
            ids = np.array([gallery[i].stem for i in gi])
            target = np.array([queries[i].stem.rsplit('-', 1)[0] for i in qi])
            ranks = np.argmax(ids[order] == target[:, None], axis=1) + 1
            all_ranks.extend(ranks.tolist())
            per_class[c] = {'queries':len(qi), 'gallery':len(gi), 'top1':float(np.mean(ranks<=1)), 'top5':float(np.mean(ranks<=5))}
        ranks = np.array(all_ranks)
        metrics.update(top1=float(np.mean(ranks<=1)), top5=float(np.mean(ranks<=5)),
                       macro_top1=float(np.mean([v['top1'] for v in per_class.values()])),
                       macro_top5=float(np.mean([v['top5'] for v in per_class.values()])), per_class=per_class)
    (writer.save if writer else safe_checkpoint)({'query_features':qf, 'gallery_features':gf, 'queries':[str(p) for p in queries],
                'gallery':[str(p) for p in gallery]}, Path(args.output)/'features.pt')
    return metrics


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=r'D:\document\DataSet\sketch_retrieval\sketchy')
    p.add_argument('--task', choices=['cl','fg'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--weights', default='ViT-B/32')
    p.add_argument('--checkpoint')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps-per-epoch', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--samples-per-class', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--prompt-lr', type=float, default=1e-5)
    p.add_argument('--ln-lr', type=float, default=1e-6)
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--eval-limit', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p


def atomic_checkpoint(state, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def save_warning(path, exc):
    tqdm.write(f'WARNING: Could not save {path}: {exc}. Skipping save and continuing.')


def safe_checkpoint(state, path):
    try:
        atomic_checkpoint(state, path)
        return True
    except Exception as exc:
        save_warning(path, exc)
        try:
            Path(path).with_suffix('.tmp').unlink(missing_ok=True)
        except OSError:
            pass
        return False


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


class AsyncWriter:
    """Ordered disk writes with independent CPU snapshots and a bounded queue."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint')
        self.slots = BoundedSemaphore(4)

    def save(self, state, path):
        if not self.slots.acquire(blocking=False):
            save_warning(path, 'background save queue is full')
            return
        try:
            snapshot = cpu_snapshot(state)
            future = self.executor.submit(safe_checkpoint, snapshot, path)
        except Exception as exc:
            self.slots.release()
            save_warning(path, exc)
            return
        future.add_done_callback(lambda _: self.slots.release())

    def close(self):
        self.executor.shutdown(wait=True)


def write_text_safely(path, text, append=False):
    try:
        with Path(path).open('a' if append else 'w', encoding='utf-8') as stream:
            stream.write(text)
    except OSError as exc:
        save_warning(path, exc)


def run(args, on_epoch=None, tracking=None):
    writer = AsyncWriter()
    try:
        return _run(args, on_epoch, tracking, writer)
    finally:
        writer.close()


def _run(args, on_epoch, tracking, writer):
    seed_all(args.seed)
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cats, photos, sketches, lookup = inventory(args.data_dir)
    train_cats = sorted(set(cats)-set(unseen_classes))
    test_cats = sorted(unseen_classes)
    manifest = {'train_classes':train_cats,'test_classes':test_cats,'photos':sum(map(len,photos.values())),
                'sketches':sum(map(len,sketches.values())), 'args':vars(args),
                'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write_text_safely(out/'manifest.json', json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['args','train_classes','test_classes']}),flush=True)
    model = RetrievalModel(args.task, args.weights).to(args.device)
    optimizer = torch.optim.Adam([{'params':[v for n,v in model.named_parameters() if v.requires_grad and 'prompt' in n], 'lr':args.prompt_lr},
                                  {'params':[v for n,v in model.named_parameters() if v.requires_grad and 'prompt' not in n], 'lr':args.ln_lr}])
    start_epoch = 0
    scaler = torch.amp.GradScaler('cuda', enabled=args.device.startswith('cuda'))
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if ckpt['task'] != args.task:
            raise ValueError('Checkpoint task mismatch')
        if ckpt['args']['weights'] != args.weights:
            raise ValueError('Use the same --weights as the checkpoint')
        if set(ckpt['adapter']) != set(model.adapter()):
            raise ValueError('Checkpoint adapter parameters do not match the model')
        model.load_state_dict(ckpt['adapter'],strict=False)
        if not args.eval_only:
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch']
            if 'rng' in ckpt:
                random.setstate(ckpt['rng']['python'])
                np.random.set_state(ckpt['rng']['numpy'])
                torch.set_rng_state(ckpt['rng']['torch'])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
            if on_epoch and ckpt.get('pending_log'):
                on_epoch(ckpt['record'], ckpt['metrics'])
                ckpt['pending_log'] = False
                writer.save(ckpt, out/'last.pt')
    elif args.eval_only:
        raise ValueError('--eval-only requires a trained --checkpoint')
    if not args.eval_only:
        ds = Triplets(train_cats, photos, sketches, lookup, args.task)
        steps = args.steps_per_epoch or (len(ds)+args.batch_size-1)//args.batch_size
        sampler = BalancedBatches(ds,args.batch_size,args.samples_per_class,steps)
        loader = DataLoader(ds,batch_sampler=sampler,num_workers=args.workers)
        with torch.no_grad():
            tokens = clip.tokenize(['a photo of a '+c.replace('_',' ') for c in train_cats]).to(args.device)
            texts = F.normalize(model.clip.encode_text(tokens).float(),dim=-1)
        for epoch in range(start_epoch,args.epochs):
            model.train()
            sums = {}
            started = time.time()
            with tqdm(loader, total=steps, desc=f'{args.task.upper()} epoch {epoch+1}/{args.epochs}',
                      unit='batch', dynamic_ncols=True) as progress:
                for step,(s,im,n,labels) in enumerate(progress):
                    s,im,n,labels = [x.to(args.device) for x in (s,im,n,labels)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',enabled=args.device.startswith('cuda')):
                        sf,pf,nf = model(s,True),model(im),model(n)
                        losses = {'triplet':triplet(sf,pf,nf,args.margin)}
                        scale = model.clip.logit_scale.exp().detach()
                        losses['classification'] = F.cross_entropy(scale*sf@texts.T,labels)+F.cross_entropy(scale*pf@texts.T,labels)
                        loss = losses['triplet']+0.5*losses['classification']
                        if args.task == 'fg':
                            losses['relative_kl'] = relative_kl(sf,pf,nf,labels)
                            order = torch.stack([torch.randperm(4,device=args.device) for _ in range(len(s))])
                            other = order.roll(1,dims=1)
                            ss, pp, pn = shuffle_patches(s,order),shuffle_patches(im,order),shuffle_patches(im,other)
                            losses['patch_triplet'] = triplet(model(ss,True),model(pp),model(pn),args.margin)
                            loss = loss+0.1*losses['relative_kl']+losses['patch_triplet']
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    losses['total'] = loss
                    for k,v in losses.items():
                        sums[k] = sums.get(k,0)+float(v.detach())
                    progress.set_postfix(loss=f'{loss.item():.4f}',
                                         avg_loss=f'{sums["total"]/(step+1):.4f}', refresh=False)
            record = {'epoch':epoch+1,'steps':steps,'seconds':time.time()-started, **{k:v/steps for k,v in sums.items()}}
            write_text_safely(out/'train.jsonl', json.dumps(record)+'\n', append=True)
            epoch_metrics = evaluate(model,test_cats,photos,sketches,args,writer=writer) if on_epoch else None
            state = {'task':args.task,'adapter':model.adapter(),'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict(),'epoch':epoch+1,'args':vars(args),
                        'tracking':tracking, 'record':record, 'metrics':epoch_metrics,
                        'pending_log':bool(on_epoch),
                        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
                               'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
            writer.save(state,out/'last.pt')
            if on_epoch:
                on_epoch(record,epoch_metrics)
                state['pending_log'] = False
                writer.save(state,out/'last.pt')
                write_text_safely(out/'metrics.json', json.dumps(epoch_metrics,indent=2))
    if on_epoch and not args.eval_only:
        metrics = epoch_metrics if start_epoch < args.epochs else ckpt['metrics']
    else:
        metrics = evaluate(model,test_cats,photos,sketches,args,writer=writer)
    metrics['checkpoint'] = str(Path(args.checkpoint) if args.eval_only else out/'last.pt')
    writer.close()
    try:
        metrics['checkpoint_sha256'] = hashlib.sha256(Path(metrics['checkpoint']).read_bytes()).hexdigest()
    except OSError as exc:
        metrics['checkpoint_sha256'] = None
        tqdm.write(f'WARNING: Could not read checkpoint for checksum: {exc}')
    write_text_safely(out/'metrics.json', json.dumps(metrics,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['per_class','test_classes']}),flush=True)
    return metrics


def main():
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
