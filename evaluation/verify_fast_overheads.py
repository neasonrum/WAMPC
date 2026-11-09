"""Verify 100-step numerical equivalence, metrics and bounded batch recovery."""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch
import ujson
from evaluation.cached_100_ddim import Cached100DDIM
from evaluation.fast_carla_io import MetricJournal, replace_metric_write, replace_startup_sleep, wait_for_server, cooperative_pause
from method.full_diffusion.diffusion import VPFullDiffusion
from evaluation.run_full_diffusion_carla_fast import publish_completed

torch.set_num_threads(1)
REPORT = {}
DEVICE = 'cpu'
CHECKPOINT = ROOT / 'results/full_diffusion_carla_20261031_epoch10/checkpoint/epoch_10.pt'
TEST_ROOT = ROOT / 'results/full_diffusion_carla_20261031_fast_validation'


class OverheadChecks(unittest.TestCase):
    def test_cooperative_pause_requires_explicit_release(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            folder=Path(folder);request=folder/'pause_request.json'
            request.write_text(json.dumps({'job_id':'13046830','token':'test-owned-request'}))
            finished=threading.Event()
            def worker():
                cooperative_pause(folder,'13046830',poll_seconds=.01)
                finished.set()
            thread=threading.Thread(target=worker);thread.start()
            deadline=time.monotonic()+2
            while not (folder/'pause_ack.json').exists() and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(json.loads((folder/'pause_ack.json').read_text())['state'],'paused')
            self.assertFalse(finished.is_set());self.assertTrue(thread.is_alive())
            request.unlink();thread.join(timeout=2)
            self.assertTrue(finished.is_set())
            self.assertEqual(json.loads((folder/'pause_ack.json').read_text())['state'],'resumed')
            request.write_text(json.dumps({'job_id':'another-job','token':'foreign-request'}))
            with self.assertRaises(RuntimeError):cooperative_pause(folder,'13046830')

    def test_100_calls_and_rng_sequence(self):
        for batch in (1, 3):
            original = VPFullDiffusion()
            fast = Cached100DDIM(original)
            left = torch.Generator(device=DEVICE).manual_seed(901)
            right = torch.Generator(device=DEVICE).manual_seed(901)
            for frame in range(2):
                calls = [0, 0]
                def predict(index):
                    def fn(sample, timestep):
                        calls[index] += 1
                        return sample.tanh() * 0.4 + timestep[:, None, None] * 0.01
                    return fn
                a = original.sample(predict(0), batch, DEVICE, left, 100)
                b = fast.sample(predict(1), batch, DEVICE, right, 100)
                self.assertEqual(calls, [100, 100])
                self.assertEqual(fast.last_denoiser_calls, 100)
                self.assertTrue(torch.equal(left.get_state(), right.get_state()))
                self.assertTrue(torch.equal(a, b), f'{DEVICE}, batch={batch}, frame={frame}, maxdiff={(a-b).abs().max()}')
                self.assertTrue(torch.equal(b[:, -1, 1], torch.zeros_like(b[:, -1, 1])))
        REPORT['synthetic_100_step_bitwise_equal'] = True

    def test_reject_fewer_steps(self):
        fast = Cached100DDIM(VPFullDiffusion())
        with self.assertRaises(ValueError):
            fast.sample(lambda x,t:x, 1, DEVICE, inference_steps=20)

    def test_real_checkpoint_head(self):
        from method.full_diffusion.model import TrajectoryFullDiffuGeoHead
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        self.assertEqual(state['config']['diffusion']['inference_steps'], 100)
        head = TrajectoryFullDiffuGeoHead(**state['config']['model']).eval().to(DEVICE)
        head.load_state_dict(state['model'], strict=True)
        original = VPFullDiffusion(head.codec, **{
            k:v for k,v in state['config']['diffusion'].items() if k!='inference_steps'})
        fast = Cached100DDIM(original)
        generator = torch.Generator(device=DEVICE).manual_seed(19)
        bev = torch.randn((1,64,8,8),device=DEVICE,generator=generator)
        memory = torch.randn((1,7,256),device=DEVICE,generator=generator)
        ego = torch.randn((1,3,256),device=DEVICE,generator=generator)
        target = torch.tensor([[12.,2.]],device=DEVICE)
        left = torch.Generator(device=DEVICE).manual_seed(901)
        right = torch.Generator(device=DEVICE).manual_seed(901)
        from contextlib import nullcontext
        context = torch.autocast('cuda', dtype=torch.bfloat16) if DEVICE.startswith('cuda') else nullcontext()
        with torch.inference_mode(), context:
            condition = head.prepare_condition(bev,memory,ego,target)
            predict = lambda noisy, timestep:head(noisy,timestep,condition)[-1]
            a = original.sample(predict,1,DEVICE,left,100)
            b = fast.sample(predict,1,DEVICE,right,100)
        max_difference=float((a-b).abs().max())
        self.assertTrue(torch.equal(a,b),f'Real head changed: maxdiff={max_difference}')
        self.assertEqual(fast.last_denoiser_calls,100)
        REPORT['real_checkpoint_head']={'bitwise_equal':True,'max_abs_difference':max_difference,'device':DEVICE,'steps':100}
        if DEVICE.startswith('cuda'):
            durations={'original':[],'cached100':[]}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for repetition in range(3):
                    for name,sampler in (('original',original),('cached100',fast)):
                        torch.cuda.synchronize()
                        started=time.perf_counter()
                        sampler.sample(predict,1,DEVICE,torch.Generator(device=DEVICE).manual_seed(901),100)
                        torch.cuda.synchronize()
                        durations[name].append(time.perf_counter()-started)
            REPORT['sampler_microbenchmark_seconds']={**durations,'scope':'Fixed synthetic condition with real checkpoint head, 100 steps; excludes perception and CARLA'}

    def test_metric_fields_and_recovery(self):
        path=next((ROOT/'results/full_diffusion_carla_20261031_epoch07/full').glob('worker*/route_1852/attempt_*/agent_data/*/metric_info.json'))
        data=json.loads(path.read_text())
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            journal=MetricJournal(folder,ujson)
            for step,metric in data.items():journal.append(int(step),metric)
            journal.finalize({int(k):v for k,v in data.items()})
            final=json.loads((Path(folder)/'metric_info.json').read_text())
            recovered=MetricJournal.recover(Path(folder)/'metric_info.jsonl')
            self.assertEqual(final,data)
            self.assertEqual(recovered,data)
            # A torn last append is recoverable; corruption in the middle is not hidden.
            with (Path(folder)/'metric_info.jsonl').open('a') as handle:handle.write('[9999,')
            self.assertEqual(MetricJournal.recover(Path(folder)/'metric_info.jsonl'),data)
            self.assertFalse((Path(folder)/'metric_info.json.tmp').exists())
            REPORT['metric_log_exact_fields']={'ticks':len(data),'all_fields_equal':True,'journal_recovery_equal':True}

    def test_logging_rewrite_preserves_control_and_order(self):
        source='''def run_step(self):
    control = self.predict()
    metric_info = self.get_metric_info()
    self.metric_info[self.step] = metric_info
    if self.save_path is not None and self.step % 1 == 0:
        with open(self.save_path / 'metric_info.json', 'w') as outfile:
            ujson.dump(self.metric_info, outfile, indent=4)
    self.finished()
    return control
'''
        base={"ujson":ujson}; altered={"ujson":ujson}
        exec(source,base);exec(compile(replace_metric_write(source),'metric_adapter_check','exec'),altered)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            trace=[]
            agent=types.SimpleNamespace(step=0,save_path=Path(folder),metric_info={})
            agent.predict=lambda:(trace.append('predict') or {'steer':0.2,'throttle':0.7,'brake':0})
            agent.get_metric_info=lambda:(trace.append('metric') or {'velocity':[1.,2.,3.]})
            agent.finished=lambda:trace.append('finished')
            original=base['run_step'](agent); original_trace=list(trace);trace.clear()
            writer=MetricJournal(folder,ujson)
            agent._store_metric=lambda metric:writer.append(agent.step,metric)
            optimized=altered['run_step'](agent);writer.finalize(agent.metric_info)
            self.assertEqual(original,optimized);self.assertEqual(original_trace,trace)
            with self.assertRaises(RuntimeError):replace_metric_write('def run_step(self): return 0')

    def test_readiness_and_no_fixed_sleep(self):
        class Client:
            def __init__(self,host,port):pass
            def set_timeout(self,timeout):pass
            def get_server_version(self):return '0.9.15'
            def get_world(self):return self
            def get_snapshot(self):return types.SimpleNamespace(frame=123)
        record=wait_for_server(None,'localhost',1,None,timeout=1,client_factory=Client)
        self.assertTrue(record['ready']);self.assertLess(record['elapsed_seconds'],.2)
        source=(ROOT/'leaderboard/leaderboard/leaderboard_evaluator.py').read_text()
        changed=replace_startup_sleep(source)
        self.assertNotIn('        time.sleep(30)',changed)
        self.assertEqual(changed.replace('        _wmmpc_wait_for_server(self.server, args.host, args.port)',
                                         '        time.sleep(30)'),source)
        ast.parse(changed)

    def test_completed_route_survives_later_batch_failure(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as folder:
            folder=Path(folder);attempt=folder/'attempt_01';metric=attempt/'agent_data/save_name/metric_info.json'
            metric.parent.mkdir(parents=True);metric.write_text('{}')
            record={'route_id':'RouteScenario_1711_rep0','status':'Completed','save_name':'save_name',
                    'meta':{'duration_game':10},'scores':{}}
            result={'entry_status':'Rejected','_checkpoint':{'records':[record]}}
            published,_=publish_completed(result,['1711'],attempt,folder,time.time()-10)
            self.assertEqual(published,['1711'])
            self.assertEqual(json.loads((folder/'route_1711/summary.json').read_text())['classification'],'driving_result')
            published,_=publish_completed(result,['1711'],attempt,folder,time.time())
            self.assertEqual(published,[])


def main():
    global DEVICE
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    DEVICE=args.device
    TEST_ROOT.mkdir(parents=True,exist_ok=True)
    started=time.time()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OverheadChecks))
    REPORT.update({'elapsed_seconds':time.time()-started,'device':DEVICE,'tests':result.testsRun,
                   'successful':result.wasSuccessful(),'errors':len(result.errors),'failures':len(result.failures),
                   'inference_steps':100,'model_weights_or_simulator_settings_changed':False})
    (TEST_ROOT/f'verification_{DEVICE.replace(":","_")}.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps(REPORT),flush=True)
    return int(not result.wasSuccessful())


if __name__=='__main__':raise SystemExit(main())
