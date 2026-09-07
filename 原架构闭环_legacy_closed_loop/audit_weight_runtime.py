"""Check optional weight laws survive storage and actual ongoing learning."""
from pathlib import Path
import json
import tempfile
import numpy as np
from original_runtime import OriginalRuntime
from checkpoint import save_tree,load_tree
from audit_continuous_learning import tree_digest


def main():
    reports=[]
    for depression in ('multiplicative','weight_dependent'):
        runtime=OriginalRuntime(8,8,8,24,time_neurons=128,
            pfc_connection_parameters={'强化量':.0005,'衰减率':.9,'警戒线':1,
                '衰减间隔':3,'消失下限':.00000001},
            pfc_plasticity={'ceiling':.025,'depression':depression})
        rng=np.random.default_rng(921)
        frames=[tuple(rng.random(n)<.4 for n in (8,8,8,24)) for _ in range(18)]
        for values in frames[:6]: runtime.record(*values)
        before=runtime.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'full_runtime.npz'
            save_tree(path,before)
            restored=OriginalRuntime.from_snapshot(load_tree(path))
        assert tree_digest(before)==tree_digest(restored.snapshot())
        for values in frames[6:]:
            runtime.record(*values);restored.record(*values)
            assert tree_digest(runtime.snapshot())==tree_digest(restored.snapshot())
        prior=runtime.snapshot()
        runtime.set_pfc_plasticity(None)
        reverted=runtime.snapshot()
        prior['parameters'].pop('pfc_plasticity')
        assert tree_digest(prior)==tree_digest(reverted)
        runtime.set_pfc_plasticity({'ceiling':.025,'depression':depression})
        assert tree_digest(runtime.snapshot())==tree_digest(restored.snapshot())
        reports.append(dict(depression=depression,exact_continuation_frames=12,
            safe_numeric_roundtrip=True,change_rule_keeps_weights_and_clock=True,
            maintenance_count=runtime.pfc.联想.维护次数))
    output=Path(__file__).resolve().parent/'results'/'weight_runtime_audit.json'
    output.write_text(json.dumps(dict(passed=True,cases=reports),indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,cases=reports)))


if __name__=='__main__': main()
