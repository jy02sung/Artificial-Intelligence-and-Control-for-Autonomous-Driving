import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from env import ic_experiment as ic
from env.mdp_config import ACTION
from env.mdp_config import OBSERVATION
from env.mdp_config import REWARD as REWARD_DEFAULTS
from env.mdp_config import SIMULATION
from env.navigation_observation import RouteNavigation
from env.road_config import EGO
from env.road_config import ROAD
from env.road_config import TRAFFIC
BASE_SCENARIO = Path(__file__).resolve().parents[1] / "scenarios/hv_high15000_entryprep"


def load_network_metadata(net_path: Path, rou_path: Path, nav_horizon: float = 400.0):
    """Parse network.net.xml and highway.rou.xml into ClarityEnv road & RouteNavigation."""
    net = ET.parse(str(net_path))
    edges = {e.get("id"): e for e in net.findall("edge") if not e.get("function")}
    paths = {
        e.get("id"): e.get("edges").split()
        for e in ET.parse(str(rou_path)).findall("route")
        if e.get("id", "").startswith("route_")
    }
    connections = [
        (c.get("from"), c.get("to"), int(c.get("fromLane")), int(c.get("toLane")))
        for c in net.findall("connection")
        if c.get("from") in edges and c.get("to") in edges
    ]
    lengths = {k: max(float(l.get("length")) for l in e.findall("lane")) for k, e in edges.items()}
    internal = {
        l.get("id"): float(l.get("length"))
        for e in net.findall("edge")
        if e.get("function") == "internal"
        for l in e.findall("lane")
    }
    links = {}
    for c in net.findall("connection"):
        key = (c.get("from"), c.get("to"))
        links[key] = max(links.get(key, 0.0), internal.get(c.get("via"), 0.0))

    counts = {k: len(e.findall("lane")) for k, e in edges.items()}
    nav = RouteNavigation(counts, connections, lengths, horizon=nav_horizon, link_lengths=links)
    solid_edges = [
        k for k, e in edges.items()
        if any(l.get("changeLeft") or l.get("changeRight") for l in e.findall("lane"))
    ]
    road = dict(
        ROAD,
        use_osm=True,
        routes=paths,
        edges=[(k, e.get("from"), e.get("to"), len(e.findall("lane"))) for k, e in edges.items()],
        connections=connections,
        solid_lane_edges=solid_edges,
    )
    ic.RAMPS = {k for k, e in edges.items() if e.get("type", "").endswith("_link")}
    ic.ZIP = ()
    return road, nav


def prepare_scenario(stage: str, cache_dir: Path) -> Path:
    """Create a scenario folder for stage ('none', 'low', 'high', 'real_ratio') and return run.sumocfg."""
    dest = cache_dir / stage
    dest.mkdir(parents=True, exist_ok=True)
    net_dst = dest / "network.net.xml"
    rou_dst = dest / "highway.rou.xml"
    cfg_dst = dest / "run.sumocfg"

    shutil.copy2(str(BASE_SCENARIO / "network.net.xml"), str(net_dst))
    shutil.copy2(str(BASE_SCENARIO / "run.sumocfg"), str(cfg_dst))

    tree = ET.parse(str(BASE_SCENARIO / "highway.rou.xml"))
    root = tree.getroot()

    if stage == "none":
        for flow in list(root.findall("flow")):
            root.remove(flow)
    elif stage == "low":
        for flow in root.findall("flow"):
            num = int(flow.get("number", "100"))
            flow.set("number", str(max(1, round(num * 0.2))))
    elif stage in ("real_ratio", "high_real_ratio"):
        # 2025 observed mix: Passenger 77%, Small Delivery 18%, Heavy Truck 4%, Bus 1%
        # Add delivery and truck vTypes and update carDist probabilities.
        deliv = ET.Element("vType", {
            "id": "delivery_idm", "vClass": "delivery", "guiShape": "delivery",
            "length": "5.5", "width": "2.0", "carFollowModel": "IDM",
            "accel": "1.6", "decel": "3.5", "emergencyDecel": "8",
            "tau": "1.3", "minGap": "2.2", "maxSpeed": "27.78", "speedDev": "0.1",
        })
        truck = ET.Element("vType", {
            "id": "truck_idm", "vClass": "truck", "guiShape": "truck",
            "length": "9.0", "width": "2.5", "carFollowModel": "IDM",
            "accel": "1.1", "decel": "3.2", "emergencyDecel": "7",
            "tau": "1.5", "minGap": "2.8", "maxSpeed": "25.0", "speedDev": "0.08",
        })
        dist = root.find("vTypeDistribution")
        if dist is not None:
            idx = list(root).index(dist)
            root.insert(idx, deliv)
            root.insert(idx + 1, truck)
            dist.set(
                "vTypes",
                "car_krauss car_idm car_eidm car_acc "
                "car_krauss_violator car_idm_violator car_eidm_violator car_acc_violator "
                "delivery_idm truck_idm bus_idm",
            )
            dist.set(
                "probabilities",
                "0.2641 0.2264 0.1509 0.1132 0.0054 0.0046 0.0031 0.0023 0.1800 0.0400 0.0100",
            )
        if stage == "real_ratio":
            for flow in root.findall("flow"):
                num = int(flow.get("number", "100"))
                flow.set("number", str(max(1, round(num * 0.5))))

    tree.write(str(rou_dst), encoding="utf-8", xml_declaration=True)
    return cfg_dst


def build_configs(args):
    action_cfg = dict(ACTION)
    action_cfg["lane_change_safety"] = dict(
        ACTION["lane_change_safety"],
        enabled=bool(args.safety),
        min_front_gap=float(args.min_front_gap),
        min_rear_gap=float(args.min_rear_gap),
        front_tau=0.3,
        rear_tau=float(args.rear_tau),
        approach_horizon=float(args.approach_horizon),
        spawn_protect=15.0,
    )
    reward_cfg = dict(
        REWARD_DEFAULTS,
        speed_weight=0.1,
        collision_penalty=args.collision_penalty,
        arrival_bonus=args.arrival_bonus,
        close_gap_threshold=6.0,
        close_gap_penalty=0.1,
        blocked_penalty=0.05,
        blocked_gap=20.0,
        blocked_speed_frac=0.7,
        lane_change_penalty=args.lc_penalty,
        invalid_action_penalty=0.0,
        ramp_speed_compliance=False,
        early_lane_guidance=False,
    )
    warmup = 5 if args.stage == "none" else (150 if args.smoke else 300)
    sim_cfg = dict(
        SIMULATION,
        step_length=0.2,
        max_steps=3000,   # 3000 * 0.2s = 600s max per episode
        warmup_steps=warmup,
    )
    ego_cfg = dict(EGO, max_speed=27.78, depart_lane="best", depart_speed="max")
    traffic_cfg = dict(TRAFFIC, prefill={"enabled": False})
    return action_cfg, reward_cfg, sim_cfg, ego_cfg, traffic_cfg


def directions(net, connections):
    """Rank outgoing roads by signed turn angle, independent of current lane."""
    vectors = {}
    for e in ET.parse(net).findall("edge"):
        if e.get("function"):
            continue
        points = [tuple(map(float,p.split(",")[:2])) for p in e.find("lane").get("shape").split()]
        def heading(a,b):
            return math.atan2(b[1]-a[1],b[0]-a[0])
        vectors[e.get("id")] = heading(points[0],points[min(3,len(points)-1)]), heading(points[max(-4,-len(points))],points[-1])
    choices = {}
    for src,dst,_,_ in connections:
        choices.setdefault(src,set()).add(dst)
    output = {}
    for src, exits in choices.items():
        if len(exits)<2:
            continue
        def angle(dst):
            delta = vectors[dst][0]-vectors[src][1]
            return math.atan2(math.sin(delta),math.cos(delta))
        for i,dst in enumerate(sorted(exits,key=angle)):
            output[src,dst] = 2*i/(len(exits)-1)-1
    return output
