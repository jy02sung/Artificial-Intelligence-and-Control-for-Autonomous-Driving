"""Publish the 12 completed, matched checkpoint evaluations in order."""
import csv
import json
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/scratch_parallel_ppo600k_20261124"
REPO="mcw2_experiments"
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True,encoding="utf8",errors="replace").strip()

def report(rows):
    lines=["# DAgger2 기반 고교통 PPO 학습 곡선","",
           "충돌 벌점 −15, 학습률 0.000025 고정. SUMO 4개가 수집한 총 스텝 수를 기준으로 표시한다. 각 모델은 같은 16개 경로 × 교통 시드 100개에서 평가했다. 단일 환경 32,768스텝 결과는 다른 수집 방식이므로 표에서 제외했다.","",
           "| 총 학습 스텝 | 완주 | 충돌 | 시간초과 |","|---:|---:|---:|---:|"]
    for step,data in rows:
        counts=[data.get(k,0) for k in ("arrival","collision","timeout")]
        assert sum(counts)==1600
        lines.append("| "+" | ".join([f"{step:,}",*[f"{n} ({n/16:.2f}%)" for n in counts]])+" |")
    lines+=["","[실험 설정](protocol.json) · [경로·시드별 실패 분석과 영상](../scratch_failure_video_20261124/README.md)"]
    (OUT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf8")

def main():
    base=json.loads((ROOT/"results/scratch_bc_dagger_ppo_20261124/dagger2_holdout.json").read_bytes())
    rows=[(0,base["counts"])]
    for step in range(50000,600001,50000):
        result=OUT/f"{step}_eval.json"
        while not result.exists(): time.sleep(10)
        data=json.loads(result.read_bytes())
        assert [(x["route"],x["seed"]) for x in data["rows"]]==[(x["route"],x["seed"]) for x in base["rows"]]
        rows.append((step,data["counts"]))
        report(rows)
        paths=[result,OUT/f"{step}.pt",OUT/"README.md"]
        if step==600000: paths.append(OUT/"rates.csv")
        assert git("branch","--show-current")==REPO
        git("add","-f","--",*[str(p.relative_to(ROOT)) for p in paths])
        if git("diff","--cached","--name-only"):
            git("commit","-m",f"Record matched high-traffic PPO evaluation at {step:,} steps","--",*[str(p.relative_to(ROOT)) for p in paths])
        for attempt in range(5):
            try:
                git("push","origin",REPO)
                break
            except subprocess.CalledProcessError:
                if attempt==4: raise
                time.sleep(15)
        print(step,data["counts"],flush=True)
if __name__=="__main__":main()
