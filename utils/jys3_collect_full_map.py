"""Full-map collection. Resume with python -m utils.jys3_collect_full_map."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import msvcrt
import time
import pyarrow as pa
import pyarrow.parquet as pq
from utils import jys3_collect_pomdp_hf as collector
from utils import jys3_compact_pomdp as fmt

ROOT=fmt.SRC.parent/"jys3_dagger5_hf_full"
DATA=ROOT/"data"
CACHE=ROOT/"pending"
TARGET=10_000_000

def save(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
    os.replace(tmp,path)

def raw_path(ep):
    return (fmt.SRC if ep<16 else CACHE)/f"route_{ep%16+1:02}-{1706000000+ep}.parquet"

def episode(ep):
    path=raw_path(ep)
    if ep>=16:
        collector.DEST=CACHE
        collector.shard(ep)
    table=pq.read_table(path)
    rows=table.to_pylist()
    tracks=fmt.encode(rows,None)
    tracks=tracks.append_column("episode",pa.array([ep]*len(tracks),type=pa.int32()))
    arrays=[]
    for k in table.column_names:
        a=table[k].combine_chunks()
        if k in ("state","next_state"):
            keys=[f.name for f in a.type if f.name!="vehicles"]
            a=pa.StructArray.from_arrays([a.field(n) for n in keys],names=keys,mask=a.is_null())
        arrays.append(a)
    return pa.Table.from_arrays(arrays,names=table.column_names),tracks,rows[-1]["outcome"]

def block(pool,i,workers):
    eps=range(i*16,i*16+16)
    reused=sum(raw_path(ep).exists() for ep in eps)
    start=time.perf_counter()
    result=list(pool.map(episode,eps))
    with TemporaryDirectory(dir=ROOT) as tmp:
        stage=Path(tmp)/"block"
        stage.mkdir()
        size=0
        for index,name in enumerate(("steps","tracks")):
            table=pa.concat_tables([r[index] for r in result])
            p=stage/(name+".parquet")
            fmt.write(table,p)
            assert p.stat().st_size<200_000_000 and pq.read_table(p).equals(table)
            size+=p.stat().st_size
        summary=dict(index=i,workers=workers,reused=reused,seconds=time.perf_counter()-start,
                     steps=sum(len(r[0]) for r in result),bytes=size,outcomes=Counter(r[2] for r in result))
        save(stage/"summary.json",summary)
        stage.rename(DATA/f"block-{i:05}")
    if i:
        for ep in eps: raw_path(ep).unlink(missing_ok=True)
    return summary

def main():
    DATA.mkdir(exist_ok=True,parents=True)
    CACHE.mkdir(exist_ok=True)
    lock=(ROOT/"collection.lock").open("a+b")
    lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ck=hashlib.sha256(collector.MODEL.read_bytes()).hexdigest()
    assert ck==json.loads((fmt.SRC/"pilot.json").read_text())["checkpoint_sha256"]
    counts=Counter()
    steps=size=i=0
    def update(m):
        nonlocal i,steps,size
        assert m["index"]==i
        counts.update(m["outcomes"]);steps+=m["steps"];size+=m["bytes"];i+=1
    for b in sorted(DATA.glob("block-*")):
        assert b.name==f"block-{i:05}"
        update(json.loads((b/"summary.json").read_text()))
    while steps<TARGET:
        workers=2 if i<=2 else 3
        with ProcessPoolExecutor(workers) as pool:
            while steps<TARGET and (workers==3 or i<=2):
                m=block(pool,i,workers)
                update(m)
                save(ROOT/"progress.json",dict(blocks=i,episodes=i*16,steps=steps,bytes=size,
                     outcomes=counts,workers=workers,complete=steps>=TARGET,checkpoint_sha256=ck))
                print(json.dumps(m),flush=True)
    lock.close()
if __name__=="__main__": main()
