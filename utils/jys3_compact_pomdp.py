"""Lossless millimetre track encoding for the rounded dagger5 pilot.

load_episode() reconstructs the course schema; the restricted version retains
all vehicles within the XY radius plus every ID in detected_vehicles.
"""
from collections import defaultdict
from pathlib import Path
import json
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC=Path(__file__).resolve().parents[1]/"results/jys3_dagger5_hf_pilot"
OUT=SRC.parent/"jys3_dagger5_compact"
FIELDS=("x","y","speed","lane_pos")
META=("id","lane","lane_id","road_id")
TS=pa.schema(dict(start=pa.int32(),id=pa.string(),lane=pa.int32(),lane_id=pa.string(),road_id=pa.string(),
                  **{f:pa.list_(pa.int32()) for f in FIELDS}))

def write(table,path):
    tmp=path.with_suffix(".partial")
    pq.write_table(table,tmp,compression="zstd",compression_level=15,use_dictionary=True)
    tmp.replace(path)

def delta(a):
    a=np.rint(np.asarray(a,dtype=np.float64)*1000).astype(np.int32)
    return np.diff(a,2,prepend=[0,0]).astype(np.int32).tolist()

def unpack(a):
    return (np.cumsum(np.cumsum(a))/1000).astype(np.float32).tolist()

def keep(s,det,radius):
    vehicles=s["vehicles"]
    if vehicles is None or radius is None: return vehicles
    return [v for v in vehicles if v["id"] in det or
            (v["x"]-s["ego_x"])**2+(v["y"]-s["ego_y"])**2<=radius**2]

def encode(rows,radius):
    runs=[]
    previous={}
    frames=[(r["state"],{v["id"] for v in r["detected_vehicles"]}) for r in rows]
    frames.append((rows[-1]["next_state"],set()))
    for t,(s,det) in enumerate(frames):
        current={}
        for v in keep(s,det,radius) or []:
            key=tuple(v[k] for k in META)
            run=previous.get(key)
            if run is None:
                run=dict(start=t,**dict(zip(META,key)),**{k:[] for k in FIELDS})
                runs.append(run)
            for k in FIELDS: run[k].append(v[k])
            current[key]=run
        previous=current
    for run in runs:
        for k in FIELDS: run[k]=delta(run[k])
    return pa.Table.from_pylist(runs,schema=TS)

def load_episode(root,name):
    rows=pq.read_table(root/"steps"/name).to_pylist()
    states=[r["state"] for r in rows]+[rows[-1]["next_state"]]
    for s in states: s["vehicles"]=[] if s["ego_alive"] else None
    for run in pq.read_table(root/"tracks"/name).to_pylist():
        seq=zip(*(unpack(run[k]) for k in FIELDS))
        for t,values in enumerate(seq,run["start"]):
            states[t]["vehicles"].append({**{k:run[k] for k in META},**dict(zip(FIELDS,values))})
    for s in states:
        if s["vehicles"]: s["vehicles"].sort(key=lambda v:v["id"])
    return rows

def main():
    sizes=defaultdict(int)
    chunks=defaultdict(int)
    rows_total=0
    for file in sorted(SRC.glob("*.parquet")):
        table=pq.read_table(file)
        rows=table.to_pylist()
        rows_total+=len(rows)
        md=pq.ParquetFile(file).metadata.row_group(0)
        for i in range(md.num_columns):
            col=md.column(i)
            chunks[col.path_in_schema.split(".list")[0]]+=col.total_compressed_size
        for label,radius in [("full",None),("r200",200),("r120",120)]:
            root=OUT/label
            for folder in ("steps","tracks"): (root/folder).mkdir(exist_ok=True,parents=True)
            stepfile=root/"steps"/file.name
            if label=="full":
                arrays=[]
                for name in table.column_names:
                    a=table[name].combine_chunks()
                    if name in ("state","next_state"):
                        keys=[f.name for f in a.type if f.name!="vehicles"]
                        a=pa.StructArray.from_arrays([a.field(k) for k in keys],names=keys,mask=a.is_null())
                    arrays.append(a)
                write(pa.Table.from_arrays(arrays,names=table.column_names),stepfile)
            elif not stepfile.exists(): os.link(OUT/"full"/"steps"/file.name,stepfile)
            path=root/"tracks"/file.name
            write(encode(rows,radius),path)
            restored=load_episode(root,file.name)
            for expected,actual in zip(rows,restored):
                for k in expected:
                    if k not in ("state","next_state"):
                        assert expected[k]==actual[k]
                        continue
                    s=expected[k]
                    if s is None:
                        assert actual[k] is None
                        continue
                    det={v["id"] for v in expected["detected_vehicles"]} if k=="state" else set()
                    vv=keep(s,det,radius)
                    assert actual[k]=={**s,"vehicles":sorted(vv,key=lambda v:v["id"]) if vv is not None else None}
            sizes[label]+=stepfile.stat().st_size+path.stat().st_size
        print(file.name,dict(sizes),flush=True)
    report=dict(steps=rows_total,bytes=sizes,source_column_bytes=chunks)
    (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")

if __name__=="__main__": main()
