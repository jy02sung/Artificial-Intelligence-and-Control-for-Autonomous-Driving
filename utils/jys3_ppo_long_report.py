"""Create a screening curve and confirm the best checkpoint after the earlier 32768-step budget."""
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from utils.jys3_ppo_long import DIR, END, MARKS, submit, finish, paired

def main():
    report=json.loads((DIR/'summary.json').read_bytes())
    rows={n:json.loads((DIR/f'screen_{n}.json').read_bytes()) for n in MARKS}
    later=max((n for n in MARKS if n>32768),key=lambda n:(*paired.score(rows[n]),-n))
    for n in {later,8192}:
        path=DIR/f'fresh_{n}.json'
        if not path.exists():
            with ProcessPoolExecutor(4) as pool:
                finish(submit(pool,n,'fresh',4),n,'fresh')
        data=json.loads(path.read_bytes())
        initial=json.loads((DIR/'fresh_0.json').read_bytes())
        assert [(r['seed'],r['case'],r['route'],r['start']) for r in data]==[(r['seed'],r['case'],r['route'],r['start']) for r in initial]
        report['fresh'][str(n)]=dict(Counter(r['outcome'] for r in data))
    report['best_after_32768']=later
    paired.save(DIR/'summary.json',report)
    with (DIR/'checkpoints.csv').open('w',newline='') as f:
        out=csv.writer(f)
        out.writerow(['steps','arrivals','collisions','wrong_lane_stalls','other_timeouts','n'])
        for n in MARKS:
            c=Counter(r['outcome'] for r in rows[n]);out.writerow([n,c['arrival'],c['collision'],c['stall'],c['timeout'],len(rows[n])])
    x=lambda n:75+n/END*670
    y=lambda count:265-count/32*220
    drawing=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="335" viewBox="0 0 800 335"><rect width="800" height="335" fill="white"/><g font-family="Malgun Gothic, sans-serif" fill="#24364a"><text x="60" y="27" font-size="18">고교통 32회 고정 평가 · 순수 PPO 완주율</text>']
    for cnt in range(0,33,8):
        drawing.append(f'<path d="M75 {y(cnt)} H745" stroke="#d8e1e6"/><text x="66" y="{y(cnt)+5}" text-anchor="end">{cnt*100//32}%</text>')
    for n in range(0,END+1,50000):
        drawing.append(f'<text x="{x(n)}" y="288" text-anchor="middle">{n//10000}만</text>')
    drawing.append('<text x="410" y="317" text-anchor="middle">학습 스텝</text>')
    line=' '.join(f"{x(n):.2f},{y(paired.score(rows[n])[0]):.2f}" for n in MARKS)
    drawing.append(f'<polyline points="{line}" fill="none" stroke="#007969" stroke-width="3"/>')
    for n in MARKS:
        drawing.append(f'<circle cx="{x(n)}" cy="{y(paired.score(rows[n])[0])}" r="4" fill="#007969"/>')
    drawing.append('</g></svg>')
    (DIR/'completion.svg').write_text(''.join(drawing),encoding='utf8')

if __name__=='__main__':main()
