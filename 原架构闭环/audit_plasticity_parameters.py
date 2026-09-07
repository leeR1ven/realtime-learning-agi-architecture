"""Independent scalar math and original-PFC fixtures for three plasticity arms.

No production file or saved brain is modified. Proposed nonlinear tables are
identified as a changed plasticity law; original PFC current/gating equations
are retained in the artificial-cluster checks. Partial-cue recall is an ability,
not a failure against a permanent AND requirement.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from 前额叶区 import 前额叶联想区
from 权重连接管理 import 连接表
from weight_dependent_plasticity import SoftBoundConnectionTable


CAP, RATE, INTERVAL, FLOOR = .025, .9, 25, .00005
MILESTONES = (1, 3, 10, 30, 100, 1000)
MODES = ('additive', 'soft_multiplicative', 'soft_weight_dependent')


def eta(mode):
    return .0001 if mode == 'additive' else .0005


def potentiation(weight, mode):
    return weight + eta(mode)*(1. if mode == 'additive' else max(0., 1.-weight/CAP))


def depression(weight, mode):
    return weight*RATE if mode != 'soft_weight_dependent' else weight/(1.+(1./RATE-1.)*(weight/CAP))


class Scalar:
    """Independent implementation of the original maintenance call ordering."""
    def __init__(self, mode):
        self.mode, self.weight, self.calls = mode, 0., 0
        self.last_decay, self.decays = -INTERVAL, 0

    def advance(self, learn):
        self.calls += 1
        if self.weight > 0 and self.calls-self.last_decay >= INTERVAL:
            self.weight = depression(self.weight, self.mode)
            if self.weight < FLOOR:
                self.weight = 0.
            self.decays += 1
            self.last_decay = self.calls
        if learn:
            self.weight = potentiation(self.weight, self.mode)
        return self.weight


def table(mode):
    kwargs = dict(强化量=eta(mode), 衰减率=RATE, 消失下限=FLOOR, 警戒线=1, 衰减间隔=INTERVAL)
    if mode == 'additive':
        return 连接表(**kwargs)
    return SoftBoundConnectionTable(**kwargs, ceiling=CAP,
        depression='weight_dependent' if mode=='soft_weight_dependent' else 'multiplicative')


def equilibrium(mode, exposures):
    """Exact cycle peak: one decay, then m positive exposure updates."""
    q = (1.-eta(mode)/CAP)**exposures
    if mode == 'additive':
        analytic = exposures*eta(mode)/(1.-RATE)
    elif mode == 'soft_multiplicative':
        analytic = CAP*(1.-q)/(1.-RATE*q)
    else:
        a = 1./RATE-1.
        b = (1.-a)*(1.-q)
        x = 2.*(1.-q)/(b+math.sqrt(b*b+4.*a*(1.-q)))
        analytic = CAP*x
    value = 0.
    for iteration in range(100000):
        following = depression(value, mode)
        for _ in range(exposures):
            following = potentiation(following, mode)
        if abs(value-following) < 1e-15:
            break
        value = following
    assert np.isclose(value, analytic, rtol=1e-10, atol=1e-12)
    return {'mode': mode, 'positive_exposures_per_decay_cycle': exposures,
        'peak_weight_analytic': analytic, 'peak_weight_numeric': value,
        'after_decay_weight': depression(analytic, mode),
        'coactive_250_sources_E_at_peak': 250*analytic, 'numeric_iterations': iteration+1,
        'timing': 'A cycle applies one depression then m consecutive positive exposures; peak is after those exposures. No floor deletion occurs at these positive equilibria.'}


def scalar_checks(mode):
    reference = Scalar(mode)
    actual = table(mode)
    milestones = []
    for exposure in range(1, max(MILESTONES)+1):
        expected = reference.advance(True)
        actual.维护()
        actual.学习(0, 1)
        observed = actual.查(0)[1]
        assert np.isclose(expected, observed, atol=1e-14, rtol=1e-12)
        assert actual.维护次数 == reference.calls
        assert actual.上次衰减维护 == reference.last_decay
        if exposure in MILESTONES:
            milestones.append({'exposures': exposure, 'weight': observed,
                'maintenance_calls': reference.calls, 'decays_so_far': reference.decays,
                'last_decay_call': reference.last_decay,
                'next_increment': potentiation(observed, mode)-observed})
    one = Scalar(mode)
    actual_one = table(mode)
    one.advance(True)
    actual_one.维护()
    actual_one.学习(0, 1)
    initial = one.weight
    retention = [{'blank_maintenance_calls': 0, 'weight': initial, 'present': True, 'decays_so_far': 0}]
    deleted = None
    checkpoints = {1,3,10,30,100,1000,10000,100000}
    for blank in range(1, 125001):
        expected = one.advance(False)
        actual_one.维护()
        observed = actual_one.查(0).get(1, 0.)
        assert np.isclose(expected, observed, atol=1e-14, rtol=1e-11)
        if not observed and deleted is None:
            deleted = {'blank_maintenance_calls': blank, 'total_maintenance_calls': one.calls,
                       'depression_events': one.decays}
        if blank in checkpoints:
            retention.append({'blank_maintenance_calls': blank, 'weight': observed,
                              'present': bool(observed), 'decays_so_far': one.decays})
    assert deleted is not None
    return {'mode': mode, 'eta': eta(mode), 'exposure_milestones': milestones,
        'one_exposure_then_blank': retention, 'one_exposure_deletion': deleted,
        'independent_scalar_matches_actual_table': True}


def neuronal_fixture(mode):
    # Each input group125 cells; together they match the current250-active
    # scale. OutputC is24 cells. Each group starts at a separate40-cell boundary.
    width = 520
    features = {}
    for name, begin, count in (('A',0,125),('B',160,125),('C',320,24),('U',360,125)):
        vector = np.zeros(width, bool)
        vector[begin:begin+count] = True
        features[name] = vector
    features['AB'] = features['A'] | features['B']
    pfc = 前额叶联想区(width, 目标上限=350)
    pfc.联想, pfc.去抑 = table(mode), table(mode)
    scalar = Scalar(mode)
    results = []
    for exposure in range(1, max(MILESTONES)+1):
        # Independent artificially supplied transition pairs. No hidden
        # continuity, negative teaching, reward or reciprocal edge is inserted.
        pfc.学习(features['AB'], features['C'])
        predicted = scalar.advance(True)
        assert np.isclose(pfc.联想.查(0)[320], predicted, atol=1e-14, rtol=1e-12)
        if exposure not in MILESTONES:
            continue
        probes = []
        for cue in ('A','B','AB','U'):
            clone = copy.deepcopy(pfc)
            previous = features[cue]
            e = clone.驱动(previous)
            net = clone.门场(previous)
            activity = clone.微步(previous)
            target = 320
            threshold = clone.门槛+net[target//clone.每段]
            assert np.array_equal(activity, e >= clone.门槛+net[np.arange(width)//clone.每段])
            probes.append({'cue': cue, 'source_active_cells': int(previous.sum()),
                'target_C_active_cells': int((activity&features['C']).sum()),
                'all_active_cells': int(activity.sum()), 'target_cell': target,
                'E': float(e[target]), 'net_I': float(net[target//clone.每段]),
                'threshold': float(threshold), 'margin': float(e[target]-threshold),
                'interpretation': 'Partial cue C activation is permitted pattern completion. UnknownU has no learned edges and should remain inactive in this fixture.'})
        results.append({'exposures': exposure, 'representative_edge_0_to_320': predicted,
            'E_edges': pfc.联想.条数(), 'DI_edges': pfc.去抑.条数(), 'probes': probes})
    return {'mode': mode, 'parameters': {'width': width, 'A_cells':125, 'B_cells':125, 'C_cells':24,
        'PFC_firing_threshold':2., 'rest_I':1., 'feedback_I':.12, 'DI_cancellation':.2, 'target_high':350},
        'training': 'Each exposure calls original PFC.学习(artificialA|B, artificialC). Only connection plasticity differs across the three arms. No sensory data or autonomous route learning is claimed.',
        'milestones': results}


def append_small_cluster_bounds():
    """Add size/cap bounds and only one larger-cap candidate; no neural retrain."""
    path = HERE/'results'/'plasticity_parameter_review.json'
    report = json.loads(path.read_text(encoding='utf8'))
    original_fields = ('scalar_checks','cycle_equilibria','local_update_values','neuronal_fixtures')
    original_hash = hashlib.sha256(json.dumps({k:report[k] for k in original_fields},sort_keys=True).encode()).hexdigest()
    sizes = (24,31,64,120,250)
    caps = (.025,.1)
    bounds = []
    for cap in caps:
        for sources in sizes:
            maximum_E = sources*cap
            maximum_DI_cancellation = .2*maximum_E
            net_I = max(0.,1.-maximum_DI_cancellation)
            bounds.append({'cap':cap,'sources':sources,'maximum_E':maximum_E,
                'maximum_DI_cancellation':maximum_DI_cancellation,
                'absolute_zero_inhibition_threshold':2.,
                'can_cross_even_if_all_inhibition_cancelled':maximum_E>=2.,
                'optimistic_rest_strength1_net_I':net_I,
                'optimistic_rest_strength1_threshold':2.+net_I,
                'optimistic_rest_strength1_margin':maximum_E-2.-net_I,
                'can_cross_rest_strength1_gate':maximum_E>=2.+net_I})
    # Only W=.1 is an additional candidate. Eta/r/interval stay as proposed.
    candidate = []
    cap = .1
    for mode in ('soft_multiplicative','soft_weight_dependent'):
        q = (1.-.0005/cap)**INTERVAL
        if mode=='soft_multiplicative':
            peak = cap*(1.-q)/(1.-RATE*q)
        else:
            a=1./RATE-1.; b=(1.-a)*(1.-q)
            peak = cap*2.*(1.-q)/(b+math.sqrt(b*b+4.*a*(1.-q)))
        value=0.
        for _ in range(10000):
            value=value*RATE if mode=='soft_multiplicative' else value/(1.+(1./RATE-1.)*value/cap)
            for _ in range(INTERVAL):
                value += .0005*(1.-value/cap)
        assert np.isclose(value,peak,rtol=1e-12,atol=1e-14)
        per_size=[]
        for sources in sizes:
            E=sources*peak
            net_I=max(0.,1.-.2*E)
            per_size.append({'sources':sources,'equilibrium_peak_E':E,
                'rest_strength1_net_I':net_I,'rest_strength1_threshold':2.+net_I,
                'margin':E-2.-net_I,'can_cross_rest_strength1_gate':E>=2.+net_I,
                'can_cross_if_inhibition_zero':E>=2.})
        candidate.append({'mode':mode,'cap':cap,'eta':.0005,'decay_rate':RATE,
            'maintenance_interval':INTERVAL,'exposures_per_interval':INTERVAL,
            'cycle_peak_weight':peak,'sizes':per_size})
    review={'sizes':sizes,'cap_maximum_bounds':bounds,'only_additional_cap_candidate':candidate,
        'interpretation':'The cap limit applies to newly born weights that remain underW, not legacy weights already aboveW. No external or recalled current is included. Rest gate uses global strength1 and no prior target-site activity; local feedback or higher strength can only make that assumed gate harder.',
        'decisive_small_cluster_limit':'W=.025 gives N24/31/64 Emax=.6/.775/1.6, all below the original absolute firing threshold2 even with zero inhibition. Independent small-cluster propagation is mathematically impossible under those assumptions.',
        'rest_gate_source_requirement_at_W025':'With rest strength1 and DI coefficient.2, E must reach2.5, hence at least100 saturated sources atW=.025; the zero-inhibition absolute lower bound needs80.',
        'W01_limit':'W=.1 raises theoretical31-source Emax to3.1, but24-source Emax2.4 remains below the2.52 rest-gate threshold. With the sameeta.0005 and.9/25 decay, even the dense-exposure equilibrium for31 sources stays below its rest-gate threshold in both depression modes.',
        'large_cluster_tradeoff':'W=.1 permits250-source Emax25 rather than6.25. At an existingw=.0243, its next increment is.0003785, compared with.000014 atW=.025 and.0001 in the old additive scheme. EnlargingW can therefore strengthen already-common dense associations.',
        'learning_rate_and_decay_implication':'IncreasingW alone does not guarantee small-cluster firing at the attained equilibrium. Tuning total input, gain, decay and actual temporal overlap requires explicit real measurements; recall-current help must not be counted as a pureA shortcut.',
        'original_numeric_sections_sha256_unchanged':original_hash,
        'method':'Analytical source-count upper bounds and exact dense-exposure cycle fixed points, numerically checked. No added PFC training or production change.'}
    report['small_cluster_scale_review']=review
    report['parameter_recommendation']='Preserve the already-running three arms as diagnostics, but do not chooseW=.025 as a default for independent24/31/64-source PFC reasoning: its cap is below the absolute firing requirement. Only the additionalW=.1 candidate was analyzed; it also fails the24/31 rest-gate equilibrium with unchangedeta/decay, so no tested setting is endorsed as solving small-cluster shortcut learning.'
    assert hashlib.sha256(json.dumps({k:report[k] for k in original_fields},sort_keys=True).encode()).hexdigest()==original_hash
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    marker='\n## 小簇直接联想的上界核查\n'
    original_text=path.with_suffix('.md').read_text(encoding='utf8').split(marker)[0]
    notice='**补充的关键限制：W=.025堵住了24/31/64源细胞独立驱动后继的可能性，因为总兴奋上界仍低于原绝对门槛2。下面的小簇核查优先于把该cap作为默认配置的建议。**\n\n'
    if notice not in original_text:
        original_text=original_text.replace('\n\n','\n\n'+notice,1)
    lines=[marker.rstrip(),'','假设没有当前/回忆外来电流，所有相关新生边已达到上界；同时把去抑制边也按最有利的上界计算。原基础放电门槛是2。静息栏还取全局强度1、目标位点此前未活跃；若有局部反馈抑制会更难。','',
        '| 单边W | 源细胞数N | 最大E=N×W | 即使完全去抑能否跨2 | 静息+去抑后的门槛 | 静息条件能否激活 |','|---:|---:|---:|---|---:|---|']
    for row in bounds:
        lines.append(f'| {row["cap"]} | {row["sources"]} | {row["maximum_E"]:.3f} | '
            f'{row["can_cross_even_if_all_inhibition_cancelled"]} | {row["optimistic_rest_strength1_threshold"]:.3f} | {row["can_cross_rest_strength1_gate"]} |')
    lines += ['','W=.025对24/31/64细胞簇给出的上界分别为.600/.775/1.600，不是多学几次就能越过2。当前感觉或回放电流可以帮助，但那是受辅助驱动，不能用它声称小簇已独立走通快捷链。','',
        '仅进一步检查W=.1，保持强化.0005及.9/25完全相同。下面甚至假设每个维护周期25次都再次强化同一边，是对高频连接的有利条件：','',
        '| W=.1的衰减方式 | 稳态峰值单边w | 24源E | 31源E | 31源静息门槛 | 31源能否激活 | 250源E |','|---|---:|---:|---:|---:|---|---:|']
    for row in candidate:
        by={x['sources']:x for x in row['sizes']}
        lines.append(f'| {row["mode"]} | {row["cycle_peak_weight"]:.6f} | {by[24]["equilibrium_peak_E"]:.3f} | '
            f'{by[31]["equilibrium_peak_E"]:.3f} | {by[31]["rest_strength1_threshold"]:.3f} | '
            f'{by[31]["can_cross_rest_strength1_gate"]} | {by[250]["equilibrium_peak_E"]:.3f} |')
    lines += ['','因此W=.1也不能单独解决这组强化/衰减下的24或31源静息传递；同时它会显著增大250源的累计电流。不能由理论上限足够就忽略实际平衡点，也不能用大背景簇的成功替代小簇思考验收。','',
        '这次只有一个额外cap候选，没有扫描。保留正在运行的三组结果用于比较，但不据此确定小簇逻辑的默认权重函数；下一步需要同时测小簇可传递性、真实内容保留、冲突干扰和活动规模。上面原三组标量与人工PFC测量值未改。']
    path.with_suffix('.md').write_text(original_text+'\n'+'\n'.join(lines)+'\n',encoding='utf8')
    print('SMALL_CLUSTER_BOUNDS '+json.dumps({'cap025_impossible_even_zeroI_sources':[r['sources'] for r in bounds if r['cap']==.025 and not r['can_cross_even_if_all_inhibition_cancelled']],
        'cap01_equilibria':candidate}),flush=True)


def main():
    started = time.perf_counter()
    protected = (ROOT/'前额叶区.py', ROOT/'权重连接管理.py', HERE/'weight_dependent_plasticity.py')
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    scalar = [scalar_checks(mode) for mode in MODES]
    equilibria = [equilibrium(mode, count) for mode in MODES for count in (1,5,25)]
    fixtures = []
    for mode in MODES:
        value = neuronal_fixture(mode)
        fixtures.append(value)
        print('PLASTICITY_FIXTURE '+json.dumps({'mode':mode,
            'milestones':[{'n':row['exposures'],'weight':row['representative_edge_0_to_320'],
                'C_counts':{p['cue']:p['target_C_active_cells'] for p in row['probes']}}
                for row in value['milestones']]}), flush=True)
    local = []
    for w in (0., .0005, .001869, .02, .0243, .025):
        local.append({'weight':w, 'old_increment':.0001, 'new_increment':potentiation(w,'soft_multiplicative')-w,
            'new_increment_divided_by_old':(potentiation(w,'soft_multiplicative')-w)/.0001,
            'multiplicative_after_one_decay':w*RATE,
            'weight_dependent_after_one_decay':depression(w,'soft_weight_dependent'),
            'weight_dependent_retained_fraction':None if not w else depression(w,'soft_weight_dependent')/w})
    after = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert hashes == after
    report = {'status':'complete','source_sha256':hashes,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'parameters':{'W':CAP,'eta_old':.0001,'eta_soft':.0005,'r':RATE,'maintenance_interval':INTERVAL,'warning':1,'floor':FLOOR},
        'equations': {'growth_old':'w+eta_old', 'growth_soft':'w+eta_soft*max(0,1-w/W)',
            'depression_old':'r*w','depression_weight_dependent':'w/[1+(1/r-1)*(w/W)]',
            'soft_without_decay':'W-(W-w0)*(1-eta/W)^n, provided0<=w0<=W',
            'unused_soft_depression':'1/w_k=1/w0+k*(1/r-1)/W until floor deletion',
            'maintenance_phase':'Original维护 runs before学习. From an empty one-edge table with warning1, the first decay is call2, then27,52,...; it is not deferred until call25.'},
        'scalar_checks':scalar,'cycle_equilibria':equilibria,'local_update_values':local,'neuronal_fixtures':fixtures,
        'engineering_findings':[
            'Weak positive updates are up to5x larger, while w=.0243 grows only.14x the old increment. This is not uniformly stronger learning.',
            'The soft bound controls each newly learned edge, not incoming total current.250 simultaneous sources atW can supply6.25; disinhibition may additionally lower the local threshold.',
            'Weight-dependent forgetting is algebraic for unused weak edges. It preserves rare traces and one-off noise alike; track edge count, memory use and p95 as well as old/new recall.',
            'For equal learning/depression schedules, existing above-ceiling weights are preserved by potentiation; the ceiling is not a retroactive clamp. The supplied old max.0243 is below.025.',
            'The birth/serialization guard eta<=W is necessary for the no-overshoot argument. Current chosen eta/W=.02 satisfies it.',
            'Baseline-to-soft comparison changes both eta and growth shape; only soft_multiplicative versus soft_weight_dependent isolates the depression law.'],
        'parameter_recommendation':'Keep these three preregistered arms and their four scalars eta/W/r/interval for the real1200-step comparison; do not infer stability from cap alone or add a broad sweep. An optional fourth eta=.0001 soft-multiplicative arm would isolate growth shape from5x eta, only if causal attribution rather than practical comparison is required.',
        'boundary':'Scalar and artificially clamped original-PFC dynamics only. Partial-cue completion is an allowed positive result. Content retention, context conflict separation, responsive non-silent activity and performance require the independent real sensory trial.',
        'production_files_changed':False,'elapsed_seconds':time.perf_counter()-started}
    path=HERE/'results'/'plasticity_parameter_review.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 三种权重函数的参数核查','',
        '三种预设更新在数学上有效，但单边上界不等于总电流稳定。弱权重采用依赖权重的衰减后，会从指数遗忘变为代数遗忘；它同时保留罕见经历和偶然噪声。','',
        '固定W=.025、r=.9、维护间隔25、警戒1、删除下限.00005；旧强化.0001，软上限强化.0005。仅这三组，没有扫描。', '',
        '| 更新方案 | 曝光1 | 3 | 10 | 30 | 100 | 1000 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in scalar:
        lines.append('| '+row['mode']+' | '+' | '.join(f'{m["weight"]:.7f}' for m in row['exposure_milestones'])+' |')
    lines += ['', '这里维护发生在学习前。空表开始时警戒1在第2次维护首次触发，以后为第27、52次等。独立标量递推与实际连接表逐步核对。', '',
        '| 每次衰减之间的正曝光数 | 方案 | 平衡峰值权重 | 250源同时给同一细胞的E |', '|---:|---|---:|---:|']
    for row in equilibria:
        lines.append(f'| {row["positive_exposures_per_decay_cycle"]} | {row["mode"]} | '
            f'{row["peak_weight_analytic"]:.7f} | {row["coactive_250_sources_E_at_peak"]:.3f} |')
    lines += ['', '平衡值假设每周期先衰减、后连续给定次数的正曝光，已用离散迭代验证；实际内容出现频率和周期相位仍会影响权重。', '',
        '| 一次曝光后不再使用 | 删除前经历的衰减次数 | 删除发生于后续空白维护次数 |', '|---|---:|---:|']
    for row in scalar:
        d=row['one_exposure_deletion']
        lines.append(f'| {row["mode"]} | {d["depression_events"]} | {d["blank_maintenance_calls"]} |')
    lines += ['', '权重依赖衰减满足 1/wₖ=1/w₀+k(1/r−1)/W。三组的一次初始增量不同；软函数单次.0005，旧函数单次.0001。'
        '表中删除时刻按原严格小于floor判定，边界受浮点舍入影响最多一个维护周期。不要把弱边活得更久直接等同知识更可靠。', '',
        '在提供的旧分布中，w=.001869的新强化约为旧增量4.626倍，w=.02恰好相同，w=.0243只有旧增量.14倍。'
        'W=.025时250个同时活跃来源的上界电流是6.25；若目标的去抑制同时增大，净抑制还可能减小。需观察真实电流、活动和新旧内容，不只看最大单边权重。', '',
        '人工PFC对照保留原兴奋/抑制方程，仅替换为被比较的连接表。两个输入簇各125细胞，共同经历后继C24细胞；'
        '纯A、纯B、AB、未学U分别探测。这里单线索能唤起C属于联想补全能力，不按固定AND表判错。', '',
        '| 方案 | 曝光次数 | A单独唤起C细胞数 | AB唤起C细胞数 | 未学U唤起C细胞数 |', '|---|---:|---:|---:|---:|']
    for fixture in fixtures:
        for row in fixture['milestones']:
            p={x['cue']:x['target_C_active_cells'] for x in row['probes']}
            lines.append(f'| {fixture["mode"]} | {row["exposures"]} | {p["A"]} | {p["AB"]} | {p["U"]} |')
    lines += ['', '建议先保持这三组预设实测，不追加大扫描。第三组对第二组能单独检验衰减函数；第一组到第二组同时改变了增量和增长形状，不能把差别全归于上限。'
        '若以后专门需要拆分这两项，最多追加强化.0001的软上限乘性衰减对照。', '',
        '安全参数范围：0<eta≤W、0<r≤1、W有限正数。现值满足。旧权重大于W时原实现只停止进一步强化，并不截断旧权重；提供的旧max=.0243未越界。'
        '本次没有改正式文件或存档，没有真实感知、导航或知识规模成绩。长期判断仍需新旧内容保留、冲突区分、可用活动、连接数、内存与p95共同支持。']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    append_small_cluster_bounds()
    print('PLASTICITY_REVIEW_COMPLETE '+json.dumps({'elapsed_seconds':report['elapsed_seconds'],
        'one_exposure_deletion':{row['mode']:row['one_exposure_deletion'] for row in scalar}}),flush=True)


if __name__=='__main__':
    if '--append-size-bound' in sys.argv:
        append_small_cluster_bounds()
    else:
        main()
