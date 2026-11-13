"""Fixed 15,000 vehicles/hour HV diagnostics and validation."""
import argparse,concurrent.futures,functools,json,os,shutil,subprocess
from pathlib import Path
import xml.etree.ElementTree as E
from utils import hv_flow_audit as a
from utils.hv_flow_repair import init_worker
from utils.hv_measured_traffic import allocate
ROOT=Path(__file__).resolve().parents[1]

def prepare(out,mode,duration):
 d=out/'inputs'/mode;d.mkdir(parents=True,exist_ok=True)
 for n in ('network.net.xml','highway.rou.xml','run.sumocfg'):shutil.copy2(ROOT/'scenarios/hv_stable'/n,d/n)
 tree=E.parse(d/'highway.rou.xml');flows=tree.findall('flow')
 rates=allocate(15000,[float(f.get('vehsPerHour')) for f in flows]);counts=allocate(round(15000*duration/3600),rates)
 for f,rate,count in zip(flows,rates,counts):
  f.attrib.pop('vehsPerHour');f.set('number',str(count));f.set('begin','0');f.set('end',str(duration))
  if mode=='speed10':f.set('departSpeed','10')
 tree.write(d/'highway.rou.xml',encoding='utf8',xml_declaration=True)
 a.save(d/'manifest.json',dict(mode=mode,total_vph=15000,duration=duration,planned=sum(counts),rates={f.get('route'):v for f,v in zip(flows,rates)},changes='only departSpeed=max -> 10 m/s' if mode=='speed10' else 'proportional scale of stable 10200 vph to 15000 vph',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={n:a.sha(d/n) for n in ('network.net.xml','highway.rou.xml','run.sumocfg')},code_hashes={n:a.sha(ROOT/'utils'/n) for n in ('hv_high_traffic.py','hv_flow_audit.py')}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--modes',nargs='+',default=['baseline']);p.add_argument('--seeds',nargs='+',type=int,default=[7101,7102,7103]);p.add_argument('--duration',type=int,default=3600);p.add_argument('--horizon',type=int,default=7200);p.add_argument('--output',default='results/hv_high15000_v1');args=p.parse_args()
 assert set(args.modes)<= {'baseline','speed10'}
 out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);lock=out/'active.lock'
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  for mode in args.modes:prepare(out,mode,args.duration)
  a.save(out/'status.json',dict(stage='running',pid=os.getpid(),seeds=args.seeds,modes=args.modes))
  with concurrent.futures.ProcessPoolExecutor(max_workers=6,initializer=init_worker,initargs=(str(out),)) as pool:
   futures=[pool.submit(functools.partial(a.run,horizon=args.horizon,junction_checks=True),(m,s)) for m in args.modes for s in args.seeds]
   for f in concurrent.futures.as_completed(futures):print(f.result(),flush=True)
  a.save(out/'status.json',dict(stage='complete'))
 except Exception as exc:a.save(out/'status.json',dict(stage='error',error=repr(exc)));raise
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
