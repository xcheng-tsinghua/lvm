import unittest
from threading import Event
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.sketchy import AsyncWriter, atomic_checkpoint, evaluate, relative_kl, shuffle_patches, triplet

class RetrievalTests(unittest.TestCase):
    def test_async_snapshot_and_order(self):
        entered, release = Event(), Event()
        saved = []
        def slow_save(state, path):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test writer did not release')
            saved.append(state)
        writer = AsyncWriter()
        with patch('experiments.sketchy.atomic_checkpoint', side_effect=slow_save):
            try:
                state = {'tensor': torch.tensor([1.]), 'pending_log': True}
                writer.save(state, 'unused.pt')
                self.assertTrue(entered.wait(5))
                state['tensor'].add_(1)
                state['pending_log'] = False
                writer.save(state, 'unused.pt')
            finally:
                release.set()
                writer.close()
        self.assertEqual([x['tensor'].item() for x in saved], [1., 2.])
        self.assertEqual([x['pending_log'] for x in saved], [True, False])

    def test_failed_save_preserves_checkpoint_and_next_save_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            atomic_checkpoint({'epoch': 1}, path)
            writer = AsyncWriter()
            with patch('experiments.sketchy.torch.save', side_effect=OSError('disk full')):
                with patch('experiments.sketchy.save_warning') as warning:
                    writer.save({'epoch': 2}, path)
                    writer.close()
                    warning.assert_called_once()
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 1)
            writer = AsyncWriter()
            writer.save({'epoch': 3}, path)
            writer.close()
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 3)

    def test_patch_shuffle_inverse(self):
        x = torch.arange(2*3*8*8).reshape(2,3,8,8)
        order = torch.tensor([[2,0,3,1],[3,2,1,0]])
        self.assertTrue(torch.equal(x,shuffle_patches(shuffle_patches(x,order),order.argsort(1))))

    def test_triplet_prefers_matching_instance(self):
        s = torch.tensor([[1.,0.]])
        p = s.clone()
        n = torch.tensor([[0.,1.]])
        self.assertEqual(triplet(s,p,n,.3).item(),0)
        self.assertGreater(triplet(s,n,p,.3).item(),1)

    def test_kl_identical_and_gradients(self):
        s = torch.tensor([[1.,0.]]*4)
        p = torch.tensor([[1.,0.],[.5,.5],[1.,0.],[.5,.5]],requires_grad=True)
        n = torch.zeros_like(s)
        labels = torch.tensor([0,0,1,1])
        loss = relative_kl(s,p,n,labels)
        self.assertAlmostEqual(loss.item(),0,places=6)
        p2 = torch.tensor([[1.,0.],[.5,.5],[.1,0.],[.9,.5]],requires_grad=True)
        loss = relative_kl(s,p2,n,labels)
        self.assertGreater(loss.item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(p2.grad).all())
        self.assertGreater(p2.grad.abs().sum().item(),0)

    def test_exhaustive_retrieval_protocols(self):
        photos = {'a':[Path('a/p1.jpg'),Path('a/p2.jpg')],
                  'b':[Path('b/p3.jpg'),Path('b/p4.jpg')]}
        sketches = {'a':[Path('a/p1-1.png')], 'b':[Path('b/p3-1.png')]}
        gallery = torch.eye(4)
        # The category-a query ranks its wrong instance first, true instance second.
        queries = torch.tensor([[.6,.8,0.,0.],[0.,0.,1.,0.]])
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(task='fg',eval_limit=0,output=folder)
            with patch('experiments.sketchy.encode',side_effect=[gallery,queries]):
                result = evaluate(torch.nn.Identity(),['a','b'],photos,sketches,args)
            self.assertEqual(result['top1'],.5)
            self.assertEqual(result['top5'],1.)
            self.assertEqual(result['gallery'],4)
            args.task = 'cl'
            with patch('experiments.sketchy.encode',side_effect=[gallery,queries]):
                result = evaluate(torch.nn.Identity(),['a','b'],photos,sketches,args)
            # a: AP=1; b: stable tie places the second relevant at rank 4, AP=.75.
            self.assertAlmostEqual(result['mAP_all'],.875)
            self.assertAlmostEqual(result['mAP_200'],.875)
            self.assertEqual(result['P_200'],.5)

if __name__ == '__main__':
    unittest.main()
