import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import train
from experiments import sketchy


class LauncherTests(unittest.TestCase):
    def test_resume_flag_and_metrics(self):
        self.assertFalse(train.parser().parse_args(['--resume','False']).resume)
        self.assertTrue(train.parser().parse_args(['--resume','True']).resume)
        self.assertTrue(train.parser().parse_args(['--resume']).resume)
        record = {'epoch':3,'steps':10,'seconds':1,'total':.2}
        cl = train.metric_payload(record,{'task':'cl','mAP_200':.5,'P_200':.6})
        fg = train.metric_payload(record,{'task':'fg','top1':.1,'top5':.2})
        self.assertEqual(cl['mAP@200'],.5)
        self.assertEqual(cl['precision@200'],.6)
        self.assertEqual(fg['Acc@1'],.1)
        self.assertEqual(fg['Acc@5'],.2)

    def test_new_and_resume_identity_and_secret_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            tracker = MagicMock()
            common = ['--data_dir',folder,'--run_dir',folder,'--epoch','1','--swanlab_api_key','test-secret']
            def fake_engine(args,on_epoch,tracking):
                metrics = {'task':args.task,'mAP_200':.4,'P_200':.3}
                record = {'epoch':args.epochs,'steps':1,'total':.2}
                sketchy.atomic_checkpoint({'args':vars(args),'epoch':args.epochs,'tracking':tracking},Path(args.output)/'last.pt')
                on_epoch(record,metrics)
            with patch.object(sketchy,'run',side_effect=fake_engine):
                train.main(common,tracker=tracker)
                first = tracker.init.call_args.kwargs
                train.main(common+['--resume','True','--epoch','2'],tracker=tracker)
                second = tracker.init.call_args.kwargs
                self.assertEqual(first['id'],second['id'])
                self.assertEqual(second['resume'],'must')
                self.assertEqual([x.kwargs['step'] for x in tracker.log.call_args_list],[1,2])
                self.assertNotIn('swanlab_api_key',first['config'])
                for path in Path(folder).glob('*/session.json'):
                    self.assertNotIn('test-secret',path.read_text())
                with self.assertRaises(FileNotFoundError):
                    train.main(common+['--resume','True','--seed','99'],tracker=tracker)
                with self.assertRaises(ValueError):
                    train.main(common+['--resume','True','--epoch','3','--batch_size','32'],tracker=tracker)
                train.main(common,tracker=tracker)
                self.assertNotEqual(first['id'],tracker.init.call_args.kwargs['id'])
                self.assertEqual(len(list(Path(folder).glob('*/session.json'))),2)


if __name__ == '__main__':
    unittest.main()
