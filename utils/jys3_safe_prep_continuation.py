"""Matched 25k pure PPO continuation with an optional refundable fork advance."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from shutil import copyfile
from unittest.mock import patch

from utils import jys3_gae_policy_midpoint as mid
from utils import jys3_m5ft_targeted as T

C, L = T.C, T.L
ROOT = T.ROOT.parent / "jys3_safe_prep_continuation"
SOURCE = mid.SOURCE / "lambda95" / "s33"
END = 125000


class PrepEpisodes(T.Episodes):
    def reset(self):
        self.loan = None
        self.enabled = True
        return super().reset()

    def restore(self, state):
        # SUMO replays the current episode before restoring its accounting fields.
        super().restore(state)
        self.loan = state.get("loan")
        self.enabled = state.get("enabled", not self.actions)

    def state(self):
        return dict(super().state(), loan=self.loan, enabled=self.enabled)

    def drive(self, action):
        before = self.env.snapshot
        key = self.visible(self.obs, before)
        if self.enabled and key is not None:
            self.initial_ready.setdefault(key, self.is_ready(before))
        risky = self.stats["risk_moves"]
        nxt, reward, done, cut, info = super().drive(action)
        if not self.enabled:
            return nxt, reward, done, cut, info
        after = self.env.snapshot
        upcoming = self.visible(nxt, after)
        ready = upcoming is not None and self.is_ready(after)
        clear = info.get("speed") is not None and info["speed"] < 2 and (info["gap"] is None or info["gap"] > 25)
        terminal = done or cut or (clear and self.idle + 1 >= 50)
        adjustment = 0.0
        if self.loan is not None and (terminal or self.loan != upcoming or not ready):
            adjustment = -1.0
            self.loan = None
            self.stats["prep_refunds"] += 1
        if (not terminal and key is not None and key == upcoming and ready
                and not self.initial_ready[key] and key not in self.prepaid
                and after[0] == before[0] and after[1] != before[1]
                and not self.is_ready(before) and self.stats["risk_moves"] == risky):
            self.loan = key
            self.prepaid.add(key)
            adjustment += 1.0
            self.stats["prep_advances"] += 1
        self.prep_sum += adjustment
        return nxt, reward + adjustment, done, cut, info


def prepare(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "start.json").exists():
        resume = T.torch.load(source / "resume.pt", weights_only=False)
        assert resume["step"] == mid.STEP and resume["optimizer"]["state"]
        cp = T.torch.load(source / "policy_100000.pt", weights_only=True)
        weights = dict(cp["policy"])
        weights.update({"value_trunk."+k: v for k,v in cp["value_trunk"].items()})
        assert set(weights) == set(resume["model"]) and all(T.torch.equal(v,resume["model"][k]) for k,v in weights.items())
        for name in ("resume.pt", "updates.jsonl", "episodes.jsonl"):
            copyfile(source / name, directory / name)
            if name in resume["offsets"]:
                assert (directory / name).stat().st_size == resume["offsets"][name]
        L.save(directory / "start.json", {"resume": T.ft.sha(source / "resume.pt")})
    assert L.read(directory / "start.json")["resume"] == T.ft.sha(source / "resume.pt")


def train(arm):
    directory = ROOT / arm
    prepare(SOURCE, directory)
    episode_type = PrepEpisodes if arm == "advance" else T.Episodes
    with patch.object(T, "Episodes", episode_type):
        T.train(("A",33,"A"), directory=directory, marks=(END,), collision_reward=-14., gae_lambda=.95)


def eval_part(job):
    arm, panel, rep = job
    if panel == "fresh":
        C.old.PANELS[panel] = mid.FRESH
    C.evaluate("M5-FT",33,END,panel,rep,directory=ROOT/arm,lateral=True)


def assess(pool, arms, panels):
    jobs = [(arm, panel, rep) for arm in arms for panel, (_, n) in panels.items() for rep in range(n)]
    list(pool.map(eval_part, jobs))
    results = {}
    for arm in arms:
        counts = Counter()
        for panel, (base, n) in panels.items():
            for rep in range(n):
                rows = L.read(ROOT/arm/"parts"/f"{END}_{panel}_{rep}.json")
                assert [r["seed"] for r in rows] == [base+100*rep+i for i in range(1,17)]
                if panel != "fresh":
                    ref = L.read(SOURCE/"parts"/f"{mid.STEP}_{panel}_{rep}.json")
                    assert all(x["initial_sha256"] == y["initial_sha256"] for x,y in zip(ref,rows,strict=True))
                counts.update(r["outcome"] for r in rows)
        results[arm] = counts
    return results


def main():
    arms = ("control", "advance")
    assert T.ft.sha(SOURCE/"policy_100000.pt") == mid.ARMS["lambda95"]
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(train, arms))
        results = {"screen": assess(pool, arms, {"screen": C.old.PANELS["screen"]})}
        L.save(ROOT/"summary.json",results)
        print(results,flush=True)
        passing = [a for a in arms if results["screen"][a]["arrival"] >= 116 and results["screen"][a]["collision"] <= 6]
        if passing:
            results["holdout"] = assess(pool, arms, {k: C.old.PANELS[k] for k in L.HOLDS})
            results["fresh"] = assess(pool, passing, {"fresh": mid.FRESH})
            L.save(ROOT/"summary.json",results)
    assert T.ft.sha(SOURCE/"policy_100000.pt") == mid.ARMS["lambda95"]


if __name__ == "__main__":
    main()
