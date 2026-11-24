"""Measure SUMO native trip duration after the same ego reset as jys3."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import csv
import tempfile
from utils.jys3 import environment, traci, m

def run(job):
    stage, base = job
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        env = environment(stage, 11, Path(tmp))
        try:
            for route, lane in product(m.ROUTES, (0, 1)):
                env.ego_depart_lane = str(min(lane, env.counts[env.routes[route][0]]-1))
                env.reset(seed=base+lane, options={"route":route})
                ego = env.ego_id
                traci.vehicle.setSpeedMode(ego,31)
                traci.vehicle.setLaneChangeMode(ego,1621)
                traci.vehicle.setSpeed(ego,-1)
                result = "timeout"
                for steps in range(1, env.max_steps+1):
                    traci.simulationStep()
                    if ego in traci.simulation.getCollidingVehiclesIDList():
                        result="collision"
                        break
                    if ego in traci.simulation.getStartingTeleportIDList():
                        result="teleport"
                        break
                    if ego in traci.simulation.getArrivedIDList():
                        result="arrival"
                        break
                rows.append((stage,route,base+lane,env.ego_depart_lane,result,steps))
        finally:
            env.close()
    return rows

if __name__ == "__main__":
    jobs=product(("none","high_real_ratio"),(992100,992200,992300,992400))
    with ProcessPoolExecutor(4) as pool:
        rows=[row for group in pool.map(run,jobs) for row in group]
    path=m.ROOT/"results"/"jys3_route_duration.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("traffic","route","seed","depart_lane","outcome","steps"))
        w.writerows(rows)
