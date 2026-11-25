from utils.jys3_ppo_600k import ROOT, ARMS, paired, BASE
from concurrent.futures import ProcessPoolExecutor

def batch(job):
    a,r=job
    paired.BASE["final600"]=BASE
    return paired.evaluate((f"four_600k/{a}",0,"final600",r))

if __name__=="__main__":
    jobs=[(a,r) for a in ARMS for r in range(4)]
    with ProcessPoolExecutor(4) as pool:
        for (a,r),rows in zip(jobs,pool.map(batch,jobs)):
            paired.save(ROOT/a/f"baseline_rep_{r}.json",rows)
