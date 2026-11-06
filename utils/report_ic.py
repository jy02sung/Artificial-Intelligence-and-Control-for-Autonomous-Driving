"""Render GitHub-ready results from completed runs; never infer unfinished outcomes."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ic_campaign_v2"

def render():
    rows = sorted((OUT/"runs").glob("*/complete.json"))
    lines = ["# 퇴계원IC PPO 실험 결과", "",
             "평가를 마친 조건만 표시한다. 1만 스텝은 기능 점검이며, 본 실험은 조건별 20만 스텝이다.", "",
             "|조건|seed|학습 스텝|평가 에피소드|완주|충돌|시간 초과|기준 대비 완주/충돌|학습 시간(분)|",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    results = [json.loads(p.read_text(encoding="utf8")) for p in rows]
    for p,r in zip(rows,results):
        evaluation=json.loads((p.parent/f"eval-{r['budget']}.json").read_text(encoding="utf8"))
        baseline=next((b for b in results if b["id"]==f"R0-{r['seed']}-{r['budget']}"),None)
        difference="—" if not baseline else " / ".join(f"{100*(r[k]-baseline[k]):+.1f}%p" for k in ("arrival","collision"))
        lines.append(f"|{r['id']}|{r['seed']}|{r['budget']:,}|{len(evaluation['rows'])}|{r['arrival']:.1%}|{r['collision']:.1%}|{r['timeout']:.1%}|{difference}|{r['training_seconds']/60:.1f}|")
    for p in rows:
        r=json.loads(p.read_text(encoding="utf8"))
        events=json.loads((p.parent/f"eval-{r['budget']}.json").read_text(encoding="utf8"))["rows"]
        lines+=["",f"<details><summary>{r['id']} 경로별 평가</summary>", "",
                "|경로|횟수|완주|충돌|시간 초과|","|---|---:|---:|---:|---:|"]
        for route in sorted({e["route"] for e in events}):
            group=[e for e in events if e["route"]==route]
            lines.append(f"|{route}|{len(group)}|{mean(e['arrival'] for e in group):.1%}|{mean(e['collision'] for e in group):.1%}|{mean(e['timeout'] for e in group):.1%}|")
        lines+=["","</details>"]
    return "\n".join(lines)+"\n"

if __name__=="__main__":
    (ROOT/"docs/ic-ppo-results.md").write_text(render(),encoding="utf8")
