"""Fresh paired PPO comparison: only the 300m lane-preparation shaping differs.

Arms: baseline, prepare300 (one-time progress credit),
prepare300_pressure (credit + per-step penalty while changes are still owed),
pressure_lcrisk (pressure arm + penalty for executed lane changes into a
target lane whose leader/follower gap or TTC is short),
pressure_ttc (pressure arm + per-step penalty while closing in on the visible
same-lane leader with a short time to contact).

Does not resume historical checkpoints or invoke legacy publication routines.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

os.environ.setdefault('IC_LIBSUMO', '0')  # Installed SUMO binary, no libsumo required.
import numpy as np
from env.prepare_reward import PrepareCredit, PreparePressure, lane_change_risk, approach_risk
from utils import train_navigation_clarity as c

f, b = c.f, c.f.base
OUT = f.ROOT / 'results/pmh_prepare300_fresh'
MODES = ('baseline', 'prepare300', 'prepare300_pressure', 'pressure_lcrisk', 'pressure_ttc')
PRESSURE_WEIGHT = 0.01
RISK_WEIGHT = 0.5
RISK_MIN_GAP = 5.0
RISK_HORIZON = 3.0
TTC_WEIGHT = 0.05
TTC_HORIZON = 3.0


class TrialEnv(c.ClarityEnv):
    def __init__(self, *args, prepare=False, pressure=False, risk=False, ttc=False, **kwargs):
        self.prepare = prepare
        self.ttc = ttc
        self.ttc_total = 0.0
        self.ttc_events = 0
        self.risk = risk
        self.risk_total = 0.0
        self.risk_events = 0
        self.pressure = PreparePressure(weight=PRESSURE_WEIGHT) if pressure else None
        self.extra_total = 0.0
        self.extra_events = 0
        self.pressure_total = 0.0
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.credit = PrepareCredit()
        return super().reset(*args, **kwargs)

    def target_lane_risk(self, action):
        """Risk of the lane change this action requests, measured before it runs."""
        raw = float(np.clip(np.asarray(action)[self.lane_change_index], -1.0, 1.0))
        direction = self._quantize_lane_change(raw)
        v = b.traci.vehicle
        if direction == 0 or v.getRoadID(self.ego_id).startswith(':'):
            return 0.0
        scan_front, scan_rear = self._scan_lane_neighbors(direction)

        def closer(a, c):
            return c if a is None else a if c is None or a[0] <= c[0] else c

        front = closer(self._neighbor(direction, leader=True), scan_front)
        rear = closer(self._neighbor(direction, leader=False), scan_rear)
        return lane_change_risk(front, rear, v.getSpeed(self.ego_id),
                                RISK_MIN_GAP, RISK_HORIZON)

    def step(self, action):
        v = b.traci.vehicle
        edge, lane = v.getRoadID(self.ego_id), v.getLaneIndex(self.ego_id)
        nav = self.navigation_context()
        risk = self.target_lane_risk(action) if self.risk else 0.0
        obs, reward, done, cut, info = super().step(action)
        extra = 0.0
        if (self.prepare and not done and not cut and not edge.startswith(':')
                and self.ego_id in v.getIDList() and v.getRoadID(self.ego_id) == edge):
            extra = self.credit.award(edge, nav['decision_edge'], nav['prepare_targets'],
                                      lane, v.getLaneIndex(self.ego_id), nav['distance'])
        self.extra_total += extra
        self.extra_events += int(extra > 0)
        info['prepare_reward'] = extra
        penalty = 0.0
        if self.pressure and not done and not cut and self.ego_id in v.getIDList():
            after = self.navigation_context()  # internal lanes return no targets
            penalty = self.pressure.penalty(after['prepare_targets'],
                                            v.getLaneIndex(self.ego_id), after['distance'])
        self.pressure_total += penalty
        info['prepare_pressure'] = -penalty
        # Only executed changes pay; collisions from the change still pay it too.
        risk_penalty = 0.0
        if self.risk and info.get('lane_change', 0) != 0 and info.get('lane_change_applied'):
            risk_penalty = RISK_WEIGHT * risk
        self.risk_total += risk_penalty
        self.risk_events += int(risk_penalty > 0)
        info['lane_change_risk'] = -risk_penalty
        # Rear-end risk of the state this action led to; the collision step
        # already pays collision_penalty, so terminal steps are skipped.
        ttc_penalty = 0.0
        if self.ttc and not done and not cut and self.ego_id in v.getIDList():
            ttc_penalty = TTC_WEIGHT * approach_risk(
                self._neighbor(0, leader=True), v.getSpeed(self.ego_id), TTC_HORIZON)
        self.ttc_total += ttc_penalty
        self.ttc_events += int(ttc_penalty > 0)
        info['approach_risk'] = -ttc_penalty
        return obs, reward + extra - penalty - risk_penalty - ttc_penalty, done, cut, info


def environment(seed, load='low', prepare=False, variants=(), pressure=False, risk=False,
                ttc=False):
    return TrialEnv(str(f.prepare(load)), copy.deepcopy(b.ROAD_NEW),
                    dict(b.EGO, depart_lane='best', depart_speed='max'),
                    dict(b.SIMULATION, warmup_steps=300), b.OBSERVATION,
                    copy.deepcopy(b.ACTION), copy.deepcopy(b.REWARD),
                    dict(b.TRAFFIC, prefill={'enabled': False}), variants, seed,
                    navigation=b.NAV, clarity='baseline', prepare=prepare,
                    pressure=pressure, risk=risk, ttc=ttc)


def summary(rows):
    def group(xs):
        n = len(xs)
        a, col = sum(x['arrival'] for x in xs), sum(x['collision'] for x in xs)
        return dict(cases=n, arrivals=a, collisions=col, other=n-a-col,
                    success_rate=a/n, collision_rate=col/n,
                    failures=dict(Counter(x['failure'] for x in xs if not x['arrival'])))
    return {load: dict(group([x for x in rows if x['load'] == load]),
                       per_route={route: group([x for x in rows if x['load'] == load and x['route'] == route])
                                  for route in f.ROUTES}) for load in ('none', 'low')}


def evaluate(directory, model, count):
    meta = f.read(directory / 'manifest.json')
    b.torch.set_num_threads(1)
    agent = b.PPO(obs_dim=47, act_dim=2, **meta['hp'])
    agent.load(str(model))
    f.environment = lambda mode, seed, load='none', routes=None, variants=(): environment(seed, load, False, variants)
    rows = []
    for load in ('none', 'low'):
        for route in f.ROUTES:
            b.save(directory/'status.json', dict(stage='evaluating', load=load,
                   route=route, completed_cases=len(rows), expected_cases=32*count))
            rows.extend(f.evaluate(agent, 'N1', meta['seed'], load, [route], count, ()))
            b.save(directory / (model.stem + '.evaluation.json'), rows)
    b.save(directory / (model.stem + '.evaluation.json'), rows)
    b.save(directory / (model.stem + '.summary.json'), summary(rows))


class Log(b.Log):
    def log_train(self, step, metrics, verbose=False):
        super().log_train(step, dict(metrics, prepare_total=self.env.extra_total,
                                    prepare_events=self.env.extra_events,
                                    pressure_total=self.env.pressure_total,
                                    risk_total=self.env.risk_total,
                                    risk_events=self.env.risk_events,
                                    ttc_total=self.env.ttc_total,
                                    ttc_events=self.env.ttc_events), False)
        b.save(self.directory / 'status.json', dict(stage='training', step=step,
               budget=self.budget, prepare_total=self.env.extra_total,
               prepare_events=self.env.extra_events, pressure_total=self.env.pressure_total,
               risk_total=self.env.risk_total, risk_events=self.env.risk_events,
               ttc_total=self.env.ttc_total, ttc_events=self.env.ttc_events,
               updated=time.time()))
        if step >= getattr(self, 'next_save', 50000) or step == self.budget:
            self.agent.save(str(self.directory / f'policy-{step}.pt'))
            self.next_save = step + 50000


def train(spec):
    mode, seed, budget, count = spec
    b.torch.set_num_threads(1)
    f.OUT = OUT
    directory = OUT / 'runs' / f'{mode}-{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    hp = dict(b.PLAN['hparams'], seed=seed)
    env = environment(seed, prepare=mode != 'baseline',
                      pressure=mode in ('prepare300_pressure', 'pressure_lcrisk', 'pressure_ttc'),
                      risk=mode == 'pressure_lcrisk', ttc=mode == 'pressure_ttc')
    agent = b.PPO(obs_dim=47, act_dim=2, buffer=b.RolloutBuffer(), **hp)
    # Identical per-seed state tensors are verified by the parent before training.
    initial = OUT / 'initial' / f'{seed}.pt'
    agent.load(str(initial))
    b.save(directory / 'manifest.json', dict(mode=mode, seed=seed, budget=budget, hp=hp,
           initialization='fresh, identical per-seed state_dict in both arms',
           initial_sha=b.digest(initial), map_sha=b.digest(b.MAP/'network.net.xml'),
           base_commit=b.COMMIT, action=b.ACTION, reward=b.REWARD,
           simulation=dict(b.SIMULATION, warmup_steps=300), observation=b.OBSERVATION,
           navigation_horizon=b.NAV.horizon, variants=[], train_load='low',
           traffic=f.read(f.prepare('low').parent/'manifest.json'),
           prepare_reward=dict(enabled=mode != 'baseline', horizon=300, weight=.1,
                               duplicate_credit=False),
           lane_change_risk=dict(enabled=mode == 'pressure_lcrisk', weight=RISK_WEIGHT,
                                 min_gap=RISK_MIN_GAP, ttc_horizon=RISK_HORIZON,
                                 applies='executed lane changes only; safety gate stays off'),
           approach_risk=dict(enabled=mode == 'pressure_ttc', weight=TTC_WEIGHT,
                              ttc_horizon=TTC_HORIZON,
                              leader='same lane, within observation visibility'),
           prepare_pressure=dict(enabled=mode in ('prepare300_pressure', 'pressure_lcrisk',
                                                  'pressure_ttc'), horizon=300,
                                 weight=PRESSURE_WEIGHT,
                                 formula='weight*remaining_changes*(1-distance/horizon)'),
           evaluation_count=count))
    logger = Log(directory, agent, budget, env)
    started = time.monotonic()
    try:
        agent.learn(env, budget, logger=logger, eval_interval=0, exact_budget=True)
        assert agent.num_timesteps == budget and env.route_override_count == 0
        agent.save(str(directory/'model.pt'))
    except Exception as exc:
        b.save(directory/'status.json', dict(stage='error', error=repr(exc)))
        raise
    finally:
        env.close()
        logger.close()
    b.save(directory/'status.json', dict(stage='evaluating', step=budget))
    evaluate(directory, directory/'model.pt', count)
    result = dict(mode=mode, seed=seed, steps=budget, seconds=time.monotonic()-started,
                  model_sha=b.digest(directory/'model.pt'), results=f.read(directory/'model.summary.json'))
    b.save(directory/'complete.json', result)
    b.save(directory/'status.json', dict(stage='complete', step=budget))
    return result


def main():
    global OUT, PRESSURE_WEIGHT, RISK_WEIGHT, TTC_WEIGHT
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--seeds', type=int, nargs='+', default=[11,22,33])
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--pressure-weight', type=float, default=PRESSURE_WEIGHT)
    p.add_argument('--risk-weight', type=float, default=RISK_WEIGHT)
    p.add_argument('--ttc-weight', type=float, default=TTC_WEIGHT)
    args = p.parse_args()
    PRESSURE_WEIGHT = args.pressure_weight
    RISK_WEIGHT = args.risk_weight
    TTC_WEIGHT = args.ttc_weight
    OUT = args.output.resolve()
    f.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'initial').mkdir()
    b.torch.set_num_threads(1)
    for load in ('none', 'low'):
        f.prepare(load)
    b.save(OUT/'plan.json', dict(steps=args.steps, seeds=args.seeds, count=args.count,
           modes=args.modes, pressure_weight=PRESSURE_WEIGHT, risk_weight=RISK_WEIGHT,
           ttc_weight=TTC_WEIGHT,
           baseline='fresh; NOT meeting checkpoint continuation',
           backend='libsumo' if os.environ.get('IC_LIBSUMO') == '1' else 'traci',
           sumo_version=subprocess.check_output(['sumo', '--version'], text=True),
           torch_version=b.torch.__version__))
    paths = list(f.ROOT.glob('env/*.py')) + list(f.ROOT.glob('utils/*.py'))
    paths += list(f.ROOT.glob('algorithms/*.py')) + [f.ROOT/'train.py', b.MAP/'network.net.xml',
              b.MAP/'manifest.json', f.ROOT/'docs/experiments/ic-navigation-plan-v1.json']
    hashes = {}
    for path in paths:
        rel = path.relative_to(f.ROOT)
        target = OUT/'snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(rel)] = b.digest(target)
    b.save(OUT/'source-hashes.json', hashes)
    for seed in args.seeds:
        hp = dict(b.PLAN['hparams'], seed=seed)
        first = b.PPO(obs_dim=47, act_dim=2, **hp)
        second = b.PPO(obs_dim=47, act_dim=2, **hp)
        assert all(b.torch.equal(value, second.policy.state_dict()[name])
                   for name, value in first.policy.state_dict().items())
        first.save(str(OUT/'initial'/f'{seed}.pt'))
    specs = [(mode, seed, args.steps, args.count) for seed in args.seeds
             for mode in args.modes]
    # macOS spawn workers need the selected output, not a module default.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure,
                             initargs=(str(OUT), PRESSURE_WEIGHT, RISK_WEIGHT, TTC_WEIGHT)) as pool:
        results = list(pool.map(train, specs))
    b.save(OUT/'comparison.json', results)
    write_report(OUT, results, args.modes)
    print(f'Completed {len(results)} runs. Report: {OUT / "report.md"}', flush=True)


def configure(output, pressure_weight=0.01, risk_weight=0.5, ttc_weight=0.05):
    global OUT, PRESSURE_WEIGHT, RISK_WEIGHT, TTC_WEIGHT
    TTC_WEIGHT = ttc_weight
    PRESSURE_WEIGHT = pressure_weight
    RISK_WEIGHT = risk_weight
    OUT = Path(output)
    f.OUT = OUT


def write_report(output, results, modes=MODES[:2]):
    """Persist a readable report even if the chat disconnects after training."""
    records = []
    for run in results:
        for load, metrics in run['results'].items():
            records.append(dict(mode=run['mode'], seed=run['seed'], load=load,
                **{key: metrics[key] for key in ('cases','arrivals','collisions','other',
                                                'success_rate','collision_rate')}))
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# pmh 분기 준비 보상 단독 비교 결과', '',
        '기존 모델 없이 동일 seed의 동일 초기 가중치에서 새로 학습한 비교다. 회의 모델 개선율이 아니다.', '',
        '| 교통 | 조건 | 평가 횟수 | 완주 | 충돌 | 기타 실패 | 완주율 | 충돌률 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for load in ('none','low'):
        for mode in modes:
            rows = [r for r in records if r['mode']==mode and r['load']==load]
            n, a, c, o = [sum(r[k] for r in rows) for k in ('cases','arrivals','collisions','other')]
            assert n == a+c+o
            lines.append(f'| {load} | {mode} | {n} | {a} | {c} | {o} | {a/n:.2%} | {c/n:.2%} |')
    lines += ['', '## 경로별 완주 / 충돌 / 차로 선택 실패 (전체 seed 합계)', '',
        '| 교통 | 경로 | ' + ' | '.join(modes) + ' |', '|---|---|' + '---|'*len(modes)]
    for load in ('none','low'):
        for route in f.ROUTES:
            cells = []
            for mode in modes:
                ms = [r['results'][load]['per_route'][route] for r in results if r['mode']==mode]
                a, col = sum(m['arrivals'] for m in ms), sum(m['collisions'] for m in ms)
                blk = sum(m['failures'].get('route_lane_blocked', 0) for m in ms)
                cells.append(f'{a}/{col}/{blk}')
            lines.append(f'| {load} | {route} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 저교통량 seed별 결과', '',
        '| seed | 조건 | 완주 | 충돌 | 기타 실패 |', '|---:|---|---:|---:|---:|']
    for r in records:
        if r['load']=='low':
            lines.append(f"| {r['seed']} | {r['mode']} | {r['arrivals']} | {r['collisions']} | {r['other']} |")
    lines += ['', '전체 16개 경로를 동일 평가 seed로 비교했다. 무교통의 deterministic 평가는 반복이 동일할 수 있으므로 독립 표본으로 해석하지 않는다.',
              '3개 학습 seed의 제한된 결과이며, 보상 외 교통량·지도·관측·행동·기존 안전 설정은 두 군에서 동일하다.',
              '완주율과 충돌률을 함께 판단해야 하며, 정지·시간 초과가 늘어 충돌이 감소한 경우는 개선으로 단정하지 않는다.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
