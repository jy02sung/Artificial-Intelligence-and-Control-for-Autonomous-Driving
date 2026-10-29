"""configs/road_config.py 를 읽어 SUMO 도로 파일들을 자동 생성하는 모듈.

SUMO는 도로를 여러 XML 파일로 정의한다:
    highway.nod.xml  : 노드(교차점/도로 끝점)의 좌표
    highway.edg.xml  : 엣지(도로 구간) — 어느 노드에서 어느 노드로, 차선 수, 제한속도
    highway.net.xml  : 위 두 파일을 netconvert 도구로 컴파일한 "완성된 도로망"
                       (SUMO가 실제로 읽는 파일. 직접 작성하기엔 복잡해서 도구로 생성)
    highway.rou.xml  : 차량 타입(vType)과 교통 수요(flow/route) 정의
    highway.sumocfg  : 위 파일들을 묶어주는 설정 파일 (시뮬레이션 실행의 진입점)

train.py / test.py 가 시작할 때마다 build()를 호출하므로,
road_config.py만 수정하면 도로가 자동으로 다시 만들어진다.
"""
import os
import subprocess

from sumolib import checkBinary  # SUMO 실행파일(sumo, netconvert 등) 경로를 찾아주는 유틸
from env.mdp_config import SIMULATION
try:
    from env.road_config import SCENERY
except ImportError:      # 구버전 설정 하위호환
    SCENERY = {"trees": False}


def build(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """SUMO 파일들을 out_dir에 생성하고, sumocfg 파일 경로를 반환한다.

    Args:
        road:    configs.road_config.ROAD    (도로 형상)
        ego:     configs.road_config.EGO     (RL 차량 속성)
        traffic: configs.road_config.TRAFFIC (배경 교통류)
        out_dir: 생성 위치 (보통 프로젝트의 sumo/ 폴더)
    """
    if road.get("layout") == "toegyewon_ic":
        return _ic_build(road, ego, traffic, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────
    # 1) 노드 파일: 직선 도로의 양 끝점 2개
    #    n0 = 원점(출발), n1 = length만큼 떨어진 지점(도착)
    # ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "highway.nod.xml"), "w") as f:
        f.write(f"""<nodes>
    <node id="n0" x="0" y="0"/>
    <node id="n1" x="{road['length']}" y="0"/>
</nodes>
""")

    # ──────────────────────────────────────────────────────
    # 2) 엣지 파일: n0 → n1 을 잇는 도로 구간 하나
    #    numLanes = 차선 수, speed = 제한속도(m/s)
    # ──────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "highway.edg.xml"), "w") as f:
        f.write(f"""<edges>
    <edge id="e0" from="n0" to="n1" numLanes="{road['num_lanes']}" speed="{road['speed_limit']}"/>
</edges>
""")

    # ──────────────────────────────────────────────────────
    # 3) 라우트 파일: 차량 타입 2종 + 배경 교통 flow
    #
    #    <vType>  : 차량의 물리 속성 템플릿
    #       ego_type : RL이 조종할 차 (빨간색, color="1,0,0" = RGB)
    #    <vTypeDistribution> : 배경차 컨트롤러 믹스.
    #       TRAFFIC["controllers"]에 정의된 여러 차량추종모델(Krauss/IDM/
    #       EIDM/ACC 등)을 probability 비율로 섞는다. flow가 이 분포를
    #       type으로 쓰면, 투입되는 차량마다 SUMO가 확률적으로 타입을 뽑는다.
    #    <route>  : 지나갈 엣지들의 나열. 여기선 e0 하나뿐.
    #    <flow>   : 교통 수요 정의. period="X"는 "X초마다 정확히 1대씩"
    #               → 등간격(uniform) 생성. (vehsPerHour와 등가지만
    #               "비슷한 간격으로 새로 생성"을 명시적으로 보장)
    #               ego는 여기에 없고, 환경(sumo_env.py)이 reset마다
    #               traci.vehicle.add()로 직접 투입한다.
    # ──────────────────────────────────────────────────────
    controllers = traffic.get("controllers")
    if controllers:
        # probability 합이 1이 아니어도 되도록 자동 정규화
        total_p = sum(float(c.get("probability", 1.0))
                      for c in controllers.values())
        vtype_lines = []
        for name, c in controllers.items():
            prob = float(c.get("probability", 1.0)) / total_p
            attrs = [
                f'id="car_{name}"',
                f'probability="{prob:.4f}"',
                f'carFollowModel="{c["carFollowModel"]}"',
                f'accel="{c.get("accel", 2.5)}"',
                f'decel="{c.get("decel", 4.5)}"',
                f'tau="{c.get("tau", 1.0)}"',
                f'minGap="{c.get("min_gap", 2.5)}"',
                # 컨트롤러별 희망속도 오버라이드 (없으면 공통 max_speed).
                # 컨트롤러마다 속도를 다르게 주면 차선 간 속도차가 생겨
                # "추월할 이유"가 자연스럽게 만들어진다.
                f'maxSpeed="{c.get("max_speed", traffic["max_speed"])}"',
                # speedDev: 같은 타입 안에서도 개체별 희망속도 편차 (0~)
                f'speedDev="{c.get("speed_dev", 0.1)}"',
                f'color="{c.get("color", "1,1,0")}"',
                'vClass="passenger"', 'guiShape="passenger/sedan"',
            ]
            # sigma는 Krauss 계열에만 의미가 있으므로 지정된 경우에만 넣는다
            if "sigma" in c:
                attrs.append(f'sigma="{c["sigma"]}"')
            # ── 배경차 차선 유지 (keep_lane, 기본 True) ──
            # 배경차의 자율 차선변경(전략/협조/속도이득/우측유지)을 전부 꺼서
            # 각자 자기 차선을 지키게 한다. 이유:
            #   배경차가 자유롭게 돌아다니면 (측정: 150스텝에 477회 변경)
            #   차선 간 틈이 수시로 닫혀서, ego가 틈을 노려 파고드는
            #   "칼치기 추월"을 학습할 안정적인 기회 자체가 사라진다.
            #   배경차가 차선을 유지하면 "느린 차 뒤 → 옆 틈으로 변경 →
            #   가속"이라는 추월 패턴이 재현 가능한 형태로 나타난다.
            # 특정 컨트롤러만 돌아다니게 하려면 그 항목에 "keep_lane": False.
            if c.get("keep_lane", True):
                attrs += ['lcStrategic="0"', 'lcCooperative="0"',
                          'lcSpeedGain="0"', 'lcKeepRight="0"']
            vtype_lines.append("        <vType " + " ".join(attrs) + "/>")
        vtypes_block = ('    <vTypeDistribution id="carDist">\n'
                        + "\n".join(vtype_lines)
                        + "\n    </vTypeDistribution>")
        flow_type = "carDist"
    else:
        # controllers 설정이 없으면 예전 방식(단일 Krauss)으로 동작 (하위호환)
        vtypes_block = (f'    <vType id="car" accel="2.5" decel="4.5" '
                        f'maxSpeed="{traffic["max_speed"]}" '
                        f'sigma="{traffic.get("sigma", 0.5)}" '
                        f'vClass="passenger" guiShape="passenger/sedan"/>')
        flow_type = "car"

    with open(os.path.join(out_dir, "highway.rou.xml"), "w") as f:
        f.write(f"""<routes>
    <vType id="ego_type" accel="{ego['accel']}" decel="{ego['decel']}" maxSpeed="{ego['max_speed']}" color="1,0,0" vClass="passenger" guiShape="passenger/sedan"/>
{vtypes_block}
    <route id="r0" edges="e0"/>
    <flow id="traffic" type="{flow_type}" route="r0" begin="0" end="100000"
          period="{3600.0 / traffic['vehs_per_hour']:.2f}"
          departLane="{traffic['depart_lane']}" departSpeed="{traffic['depart_speed']}"/>
</routes>
""")

    # ──────────────────────────────────────────────────────
    # 4) GUI 기본 시각화 설정
    #    GUI 관련 값의 source of truth는 env/mdp_config.py의 SIMULATION 하나뿐이다.
    #    view_road.py / train.py / test.py / 데이터 수집 모두 build()를 거치므로
    #    gui_zoom/gui_delay를 바꾸면 생성되는 highway.gui.xml에도 즉시 반영된다.
    # ──────────────────────────────────────────────────────
    # GUI 시작 화면: "도로 시작 구간"이 차선까지 보이게 자동 확대해둔다.
    #   (12km 전체 맞춤으로 시작하면 도로가 가느다란 선으로만 보여 직관성이
    #    떨어지므로, ego가 등장할 시작부를 확대해 대기 화면으로 삼는다.
    #    ▶ 재생 후에는 env가 ego를 추적(trackVehicle)하며 시점을 가져간다.)
    # 좌표/배율은 도로 길이·차선 수에서 "계산"하므로 설정을 바꿔도 안 어긋난다:
    #   zoom(%) = 도로길이 / 보고싶은폭(gui_view_width) × 100
    #     (SUMO zoom은 "네트워크 전체 = 100%" 기준)
    #   y 중심 = -차선폭(3.2m) × 차선수 / 2   (SUMO는 차선을 y<0 쪽으로 그림)
    gui_delay = float(SIMULATION.get("gui_delay", 100.0))
    view_width = float(SIMULATION.get("gui_view_width", 500.0))   # 시작 화면 시야(m)
    zoom = float(road["length"]) / view_width * 100.0
    center_x = view_width * 0.4                                   # 시작점이 화면 왼쪽에 오게
    center_y = -3.2 * int(road["num_lanes"]) / 2.0
    gui_settings_path = os.path.join(out_dir, "highway.gui.xml")
    with open(gui_settings_path, "w") as f:
        f.write(f"""<viewsettings>
    <scheme name="real world"/>
    <viewport zoom="{zoom:.0f}" x="{center_x:.1f}" y="{center_y:.1f}" angle="0"/>
    <delay value="{gui_delay}"/>
</viewsettings>
""")

    # ──────────────────────────────────────────────────────
    # 4.7) 도로변 장식(나무) — 순수 시각 요소
    #
    # SUMO의 <poly>(다각형)로 "초록 수관 + 갈색 줄기"를 그려 도로
    # 양옆에 심는다. additional 파일로 로드되며 차량 물리와 완전히
    # 무관하다 (충돌/관측/보상 어디에도 안 잡힘).
    # 좌표계: 차선은 y ∈ [0, -3.2×차선수] 영역 → 왼쪽 숲은 y > 0,
    # 오른쪽 숲은 y < -3.2×차선수 - side_offset 에 배치.
    # ──────────────────────────────────────────────────────
    _write_scenery(out_dir, road)

    # ──────────────────────────────────────────────────────
    # 5) sumocfg: "이 net 파일과 이 route 파일로 시뮬레이션해라"
    # ──────────────────────────────────────────────────────
    cfg_path = os.path.join(out_dir, "highway.sumocfg")
    with open(cfg_path, "w") as f:
        f.write("""<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="highway.rou.xml"/>
        <additional-files value="highway.scenery.xml"/>
    </input>
    <gui_only>
        <gui-settings-file value="highway.gui.xml"/>
    </gui_only>
</configuration>
""")

    # ──────────────────────────────────────────────────────
    # 6) netconvert 실행: nod + edg → net.xml 컴파일
    #    (SUMO 설치 시 함께 깔리는 커맨드라인 도구)
    # ──────────────────────────────────────────────────────
    netconvert = checkBinary("netconvert")
    subprocess.run(
        [netconvert,
         "--node-files", os.path.join(out_dir, "highway.nod.xml"),
         "--edge-files", os.path.join(out_dir, "highway.edg.xml"),
         "-o", os.path.join(out_dir, "highway.net.xml")],
        check=True,           # 실패하면 예외를 던져서 바로 알 수 있게
        capture_output=True,  # 출력은 조용히 삼킴
    )
    return cfg_path


def _blob(cx: float, cy: float, r: float, rng, n: int = 14,
          roughness: float = 0.35) -> str:
    """유기적인(울퉁불퉁한) 수관 실루엣 다각형을 만든다.

    실제 나무를 위에서 내려다보면 완전한 원이 아니라 잎 뭉치들이
    만드는 불규칙한 테두리다. 꼭짓점마다 반지름을 무작위로 흔들되,
    이웃 꼭짓점과 평균을 내서(smoothing) 뾰족하지 않고 부드럽게
    울퉁불퉁한 실루엣을 얻는다.
    """
    import math
    raw = [1.0 + rng.uniform(-roughness, roughness) for _ in range(n)]
    sm = [(raw[i - 1] + raw[i] + raw[(i + 1) % n]) / 3.0 for i in range(n)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * sm[i]
        pts.append(f"{cx + rr * math.cos(a):.2f},{cy + rr * math.sin(a):.2f}")
    return " ".join(pts)


def _write_scenery(out_dir: str, road: dict):
    """도로변 나무를 highway.scenery.xml (additional 파일)로 생성.

    항공뷰(top-down) 조경처럼 보이도록 나무 한 그루를 여러 겹으로 그린다:
      layer 1  그림자  : 수관을 남동쪽으로 살짝 밀어낸 반투명 검정 blob
                         (해가 북서쪽에 있는 항공사진 느낌)
      layer 2  수관 몸통: 중간톤 초록의 불규칙 실루엣
      layer 3  잎 뭉치  : 수관 안에 어두운/짙은 클럼프 1~2개
                         → 캐노피의 입체적인 질감
      layer 4  하이라이트: 북서쪽으로 치우친 밝은 초록 작은 blob
    배치도 실제 조경처럼: 단독목 / 2~3그루 군락 / 낮은 관목이 섞인다.
    SCENERY["trees"]가 False면 빈 additional 파일 생성.
    """
    import random
    path = os.path.join(out_dir, "highway.scenery.xml")
    cfg = SCENERY if isinstance(SCENERY, dict) else {}
    if not cfg.get("trees", False):
        with open(path, "w") as f:
            f.write("<additional/>\n")
        return

    rng = random.Random(int(cfg.get("seed", 7)))
    spacing = float(cfg.get("tree_spacing", 45.0))
    jitter = float(cfg.get("tree_jitter", 0.5))
    side = float(cfg.get("side_offset", 6.0))
    length = float(road["length"])
    road_bottom = -3.2 * int(road["num_lanes"])

    # (몸통, 어두운 클럼프, 밝은 하이라이트) 초록 팔레트 — 수종별 톤
    palettes = [
        ("46,110,44", "32,84,32", "96,158,84"),
        ("38,98,40",  "26,72,28", "84,146,74"),
        ("56,122,50", "40,92,38", "110,168,92"),
        ("30,88,36",  "20,64,26", "76,138,70"),
    ]
    bush_greens = ["70,124,58", "84,136,64", "62,112,52"]

    lines = ["<additional>"]
    tid = [0]

    def draw_tree(x, y, r):
        body, dark, light = rng.choice(palettes)
        i = tid[0]; tid[0] += 1
        # 그림자 (남동쪽 +x, -y 로 오프셋, 반투명)
        lines.append(f'    <poly id="sh_{i}" color="20,26,18,70" fill="1" '
                     f'layer="1" shape="{_blob(x + r*0.35, y - r*0.35, r*1.02, rng)}"/>')
        # 수관 몸통
        lines.append(f'    <poly id="tr_{i}" color="{body}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng)}"/>')
        # 잎 뭉치 1~2개 (어두운 톤, 수관 내부 무작위 위치)
        for k in range(rng.randint(1, 2)):
            ox = rng.uniform(-0.35, 0.35) * r
            oy = rng.uniform(-0.35, 0.35) * r
            lines.append(f'    <poly id="cl_{i}_{k}" color="{dark}" fill="1" '
                         f'layer="3" shape="{_blob(x + ox, y + oy, r * rng.uniform(0.4, 0.6), rng, n=10)}"/>')
        # 하이라이트 (북서쪽 -x, +y — 그림자 반대 방향)
        lines.append(f'    <poly id="hl_{i}" color="{light}" fill="1" '
                     f'layer="4" shape="{_blob(x - r*0.28, y + r*0.28, r*0.34, rng, n=8, roughness=0.25)}"/>')

    def draw_bush(x, y):
        i = tid[0]; tid[0] += 1
        g = rng.choice(bush_greens)
        r = rng.uniform(1.0, 1.8)
        lines.append(f'    <poly id="bsh_{i}" color="20,26,18,55" fill="1" '
                     f'layer="1" shape="{_blob(x + 0.3, y - 0.3, r, rng, n=8)}"/>')
        lines.append(f'    <poly id="bs_{i}" color="{g}" fill="1" '
                     f'layer="2" shape="{_blob(x, y, r, rng, n=8)}"/>')

    for side_sign, base_y in [(+1, side), (-1, road_bottom - side)]:
        x = rng.uniform(0, spacing)
        while x < length:
            y = base_y + side_sign * rng.uniform(0, 4.0)
            roll = rng.random()
            if roll < 0.12:
                draw_bush(x, y)                        # 15%: 낮은 관목
            elif roll < 0.67:
                draw_tree(x, y, rng.uniform(2.6, 4.6))  # 55%: 단독목 (실제 수관 스케일)
            else:
                # 25%: 2~3그루 군락 (겹치듯 붙여 심어 숲 덩어리 느낌)
                n = rng.randint(2, 4)
                for _ in range(n):
                    gx = x + rng.uniform(-5.5, 5.5)
                    gy = y + side_sign * rng.uniform(-2.0, 4.5)
                    draw_tree(gx, gy, rng.uniform(2.2, 3.8))
                x += spacing * 0.4                     # 군락 뒤엔 간격 조금 더
            x += spacing * rng.uniform(1.0 - jitter, 1.0 + jitter)
    lines.append("</additional>")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ==================================================================
# 퇴계원 IC 생성부. build(road, ego, traffic, out_dir)와 highway.*를 유지.
# ==================================================================
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def _ic_build(road, ego, traffic, out_dir):
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    for nid, (x, y, z) in road["nodes"].items():
        add(nodes, "node", id=nid, x=x, y=y, z=z, type="priority")
    save(directory/"highway.nod.xml", nodes)
    edges = ET.Element("edges")
    for eid, start, end, count, speed, priority, points in road["edges"]:
        attrs = dict(id=eid, numLanes=count, speed=speed, priority=priority,
                     spreadType="center", width=road["lane_width"])
        attrs.update({"from": start, "to": end})
        if points:
            attrs["shape"] = " ".join(",".join(f"{v:.3f}" for v in p) for p in points)
        add(edges, "edge", **attrs)
    save(directory/"highway.edg.xml", edges)
    connections = ET.Element("connections")
    for before, after, src, dst in road["connections"]:
        add(connections, "connection", **{"from": before, "to": after,
                                         "fromLane": src, "toLane": dst})
    save(directory/"highway.con.xml", connections)
    subprocess.run([
        checkBinary("netconvert"), "--node-files", "highway.nod.xml",
        "--edge-files", "highway.edg.xml", "--connection-files", "highway.con.xml",
        "--output-file", "highway.net.xml", "--no-turnarounds", "true",
        "--offset.disable-normalization", "true", "--junctions.corner-detail", "8",
        "--junctions.internal-link-detail", "12", "--junctions.minimal-shape", "true",
    ], cwd=str(directory), check=True, capture_output=True)
    _ic_write_routes(directory, road, ego, traffic)
    _ic_write_display(directory, road)
    configuration = ET.Element("configuration")
    sections = {
        "input": {"net-file": "highway.net.xml", "route-files": "highway.rou.xml",
                  "additional-files": "highway.scenery.xml"},
        "time": {"begin": 0, "step-length": SIMULATION["step_length"]},
        "processing": {"time-to-teleport": -1, "collision.action": "warn",
                       "collision.check-junctions": "true"},
        "random_number": {"seed": traffic.get("seed", 42)},
        "gui_only": {"gui-settings-file": "highway.gui.xml", "window-size": "1180,1080"},
    }
    for name, values in sections.items():
        section = add(configuration, name)
        for option, value in values.items():
            add(section, option, value=value)
    path = directory/"highway.sumocfg"
    save(path, configuration)
    return str(path)


def _ic_write_routes(directory, road, ego, traffic):
    routes = ET.Element("routes")
    controllers = traffic.get("controllers") or {}
    if not controllers:
        raise ValueError("IC traffic.controllers must define at least one vehicle type")
    total = sum(float(c.get("probability", 1)) for c in controllers.values())
    if total <= 0:
        raise ValueError("Traffic controller probabilities must have a positive sum")
    ids, probabilities = [], []
    for name, c in controllers.items():
        probability = float(c.get("probability", 1))
        if probability < 0:
            raise ValueError("Negative controller probability")
        attrs = dict(id="car_"+name, vClass="passenger", length=4.8,
                     carFollowModel=c["carFollowModel"],
                     accel=c.get("accel", 2.6), decel=c.get("decel", 4.5),
                     emergencyDecel=9, tau=c.get("tau", 1.2),
                     minGap=c.get("min_gap", 2.5),
                     maxSpeed=c.get("max_speed", traffic["max_speed"]),
                     speedFactor=1, speedDev=c.get("speed_dev", .05))
        if "sigma" in c:
            attrs["sigma"] = c["sigma"]
        if c.get("keep_lane", False):
            attrs.update(lcStrategic=0, lcCooperative=0, lcSpeedGain=0, lcKeepRight=0)
        add(routes, "vType", **attrs)
        if probability > 0:
            ids.append("car_"+name); probabilities.append(str(probability/total))
    add(routes, "vTypeDistribution", id="carDist", vTypes=" ".join(ids),
        probabilities=" ".join(probabilities))
    # 원본 reset()이 사용하는 차량 타입 ID와 경로 ID를 유지합니다.
    add(routes, "vType", id="ego_type", accel=ego["accel"], decel=ego["decel"],
        maxSpeed=ego["max_speed"], color="1,0,0", vClass="passenger",
        length=4.8, minGap=2.5, guiShape="passenger/sedan")
    for rid, edges in road["routes"].items():
        add(routes, "route", id=rid, edges=" ".join(edges))
    add(routes, "route", id="r0", edges=" ".join(road["routes"][road["ego_route"]]))
    flows = traffic["flows"]
    base_rate = sum(float(rate) for rate, color in flows.values())
    scale = float(traffic["vehs_per_hour"])/base_rate if base_rate else 0
    if scale < 0:
        raise ValueError("Traffic volume must be non-negative")
    end = float(traffic.get("flow_end", 500))
    for rid, (rate, color) in flows.items():
        if rate * scale > 0:
            add(routes, "flow", id="flow_"+rid, type="carDist", route=rid,
                begin=0, end=end, vehsPerHour=rate*scale,
                departLane=traffic["depart_lane"], departSpeed=traffic["depart_speed"], color=color)
    # 일반 SUMO 제어기의 빨간 시연차. 학습 환경이 생성하는 ego와 별도 ID.
    if traffic.get("demo_vehicle", True):
        add(routes, "vType", id="ego_demo_type", vClass="passenger", length=4.8,
            minGap=2.5, accel=2.6, decel=4.5, emergencyDecel=9, tau=1.2,
            maxSpeed=33.33, speedFactor=1, speedDev=.05,
            carFollowModel="Krauss", sigma=0, color="1,.05,.05")
        add(routes, "vehicle", id="ego_demo", type="ego_demo_type",
            route=road["ego_route"], depart=min(40, end/2), departLane="best", departSpeed=12)
    save(directory/"highway.rou.xml", routes)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def save(path, root):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def shape(points):
    return " ".join(f"{p[0]:.3f},{p[1]:.3f}" for p in points)


def shifted(points, offset):
    result = []
    for i, p in enumerate(points):
        a = points[max(0, i-1)]
        b = points[min(len(points)-1, i+1)]
        dx, dy = b[0]-a[0], b[1]-a[1]
        n = math.hypot(dx, dy) or 1
        result.append((p[0]+dy/n*offset, p[1]-dx/n*offset, p[2]))
    return result


def line(root, pid, points, color, layer, width):
    add(root, "poly", id=pid, type="display", shape=shape(points), color=color,
        fill="false", layer=f"{layer:.3f}", lineWidth=width)


def rectangle(root, pid, center, direction, length, width, color, layer):
    x, y = center
    dx, dy = direction
    n = math.hypot(dx, dy)
    ux, uy = dx/n, dy/n
    vx, vy = -uy, ux
    points = [(x+a*ux+b*vx, y+a*uy+b*vy)
              for a,b in [(-length/2,-width/2), (length/2,-width/2),
                          (length/2,width/2), (-length/2,width/2)]]
    add(root,"poly",id=pid,type="bridge",shape=shape(points),color=color,
        fill="true",layer=layer)


def marking(root, pid, points, dashed=False):
    # 선도 높이에 맞춰 그립니다. native 고정 레이어 차선선은 trueZ와 충돌합니다.
    travelled = 0
    serial = 0
    for a,b in zip(points,points[1:]):
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length < .01:
            continue
        distance = 0
        while distance < length - 1e-6:
            step = min(length-distance, 6 if not dashed else 10-(travelled % 10))
            if not dashed or travelled % 20 < 10:
                p = tuple(a[d]+(b[d]-a[d])*distance/length for d in range(3))
                q = tuple(a[d]+(b[d]-a[d])*(distance+step)/length for d in range(3))
                line(root,f"{pid}_{serial}",[p,q],"239,243,247",
                     min(p[2],q[2])+.25, .12 if dashed else .15)
                serial += 1
            distance += step
            travelled += step


def _ic_write_display(directory, road):
    view = ET.Element("viewsettings")
    scheme = add(view,"scheme",name="Toegyewon IC - aerial geometry")
    add(scheme,"opengl",trueZ="true",dither="false")
    add(scheme,"background",backgroundColor="235,240,233",showGrid="false")
    edges = add(scheme,"edges",laneEdgeMode=22,laneShowBorders="false",
                showLinkDecals="false",showLinkRules="false",
                widthExaggeration=1,showDirection="false")
    colors = add(edges,"colorScheme",name="by height at geometry-segment start",interpolated="true")
    for height,color in [(0,"78,88,102"),(4,"78,104,130"),(road["bridge_height"],"58,99,140")]:
        add(colors,"entry",color=color,threshold=height)
    vehicles=add(scheme,"vehicles",vehicleMode=0,vehicleQuality=2,
                 vehicle_exaggeration=SIMULATION["gui_vehicle_scale"],vehicle_minSize=0,
                 vehicle_constantSize="false",showBlinker="true")
    add(scheme,"junctions",drawShape="false",junctionMode=0)
    add(scheme,"pois",poiName_show="true",poiName_size=50,
        poiName_color="32,48,62",poi_exaggeration=.2)
    add(scheme,"polys",polyName_show="false")
    add(view,"delay",value=SIMULATION["gui_delay"])
    add(view,"viewport",x=SIMULATION["gui_center_x"], y=SIMULATION["gui_center_y"],
        zoom=SIMULATION["gui_overview_zoom"])
    save(directory/"highway.gui.xml",view)

    decor=ET.Element("additional")
    # 물리 연결과 무관한 시각화 전용 폴리곤입니다.
    # 상판은 지상 차량(z+1)보다 높고 교량 차량(z+1)보다 낮은 레이어.
    # Elevated spans follow actual geometric crossings, so changing road coordinates
    # cannot leave a bridge deck at the old schematic location.
    net=ET.parse(directory/"highway.net.xml").getroot()
    for pid,center,direction,length,width,h in _ic_bridge_spans(net,road):
        rectangle(decor,pid+"_shadow",(center[0]+2,center[1]-2),direction,
                  length+3,width+2,"32,42,50,90",.5)
        rectangle(decor,pid+"_slab",center,direction,length,width+1.0,"174,184,195",h-.4)
        rectangle(decor,pid+"_deck",center,direction,length,width,"58,99,140",h-.2)
        ux,uy=direction
        n=math.hypot(ux,uy);ux/=n;uy/=n
        for side in [-1,1]:
            off=side*(width/2+.25)
            a=(center[0]-ux*length/2-uy*off,center[1]-uy*length/2+ux*off)
            b=(center[0]+ux*length/2-uy*off,center[1]+uy*length/2+ux*off)
            line(decor,pid+"_rail_"+str(side),[a,b],"230,232,224",h+.15,.45)

    net=ET.parse(directory/"highway.net.xml").getroot()
    for junction in net.findall("junction"):
        if junction.get("id", "").startswith(":") or not junction.get("shape"):
            continue
        z=float(junction.get("z",0))
        color="58,99,140" if z>4 else "78,88,102"
        add(decor,"poly",id="surface_"+junction.get("id"),type="junction_display",
            shape=junction.get("shape"),fill="true",color=color,layer=z-.05)
    for edge in net.findall("edge"):
        if edge.get("id","").startswith(":"):
            continue
        lanes=edge.findall("lane")
        # 차로별 native 포장 사이의 가는 배경색 틈을 같은 지상 포장색으로 메웁니다.
        borders = []
        flat_ground = True
        for lane, side in [(lanes[0], 1), (lanes[-1], -1)]:
            points = [tuple(map(float, p.split(","))) for p in lane.get("shape").split()]
            points = [p if len(p) == 3 else (*p, 0.) for p in points]
            flat_ground = flat_ground and all(abs(p[2]) < .001 for p in points)
            borders.append(shifted(points, side * float(lane.get("width", road["lane_width"]))/2))
        if len(borders[0]) == len(borders[1]):
            # Fill gaps between native lanes at their own elevation, including bridges.
            for j in range(len(borders[0])-1):
                points=[borders[0][j],borders[0][j+1],borders[1][j+1],borders[1][j]]
                z=min(q[2] for q in points)
                f=min(1,max(0,z/road["bridge_height"]))
                color=",".join(str(round(a+(b-a)*f)) for a,b in zip((78,88,102),(58,99,140)))
                add(decor,"poly",id="pavement_"+edge.get("id")+"_"+str(j),
                    type="pavement_display",shape=shape(points),fill="true",color=color,layer=z-.1)
        for i,lane in enumerate(lanes):
            points=[tuple(float(v) for v in p.split(",")) for p in lane.get("shape").split()]
            points=[p if len(p)==3 else (*p,0.) for p in points]
            width=float(lane.get("width",road["lane_width"]))
            right=shifted(points,width/2-.06)
            marking(decor,lane.get("id")+"_right",right,dashed=(i>0))
            if i==len(lanes)-1:
                marking(decor,lane.get("id")+"_left",shifted(points,-width/2+.06))
    # Painted, non-drivable widening taper: added lanes begin after the narrow
    # exit is clear of the motorway, with a gradual shoulder instead of a square cap.
    narrow=net.find("edge[@id='north_exit']/lane")
    wide=net.find("edge[@id='north_exit_widen']/lane")
    if narrow is not None and wide is not None:
        def lane_points(lane):
            pts=[tuple(map(float,p.split(","))) for p in lane.get("shape").split()]
            return [p if len(p)==3 else (*p,0.) for p in pts]
        ns=shifted(lane_points(narrow),float(narrow.get("width"))/2)
        ws=shifted(lane_points(wide),float(wide.get("width"))/2)
        start=len(ns)-1;distance=0
        while start>0 and distance<30:
            distance+=math.dist(ns[start][:2],ns[start-1][:2]);start-=1
        a,b,c=ns[start],ns[-1],ws[0]
        add(decor,"poly",id="north_exit_widening",type="painted_taper",
            shape=shape(ns[start:]+[c]),fill="true",color="78,88,102",layer=-.08)
        line(decor,"north_exit_widening_border",[a,c],"239,243,247",.26,.15)
        for j in range(1,8):
            t=j/8
            inner=tuple(a[k]+(b[k]-a[k])*t for k in range(3))
            outer=tuple(a[k]+(c[k]-a[k])*max(0,t-.06) for k in range(3))
            line(decor,"north_exit_widening_hatch_"+str(j),[inner,outer],"239,243,247",.27,.12)

    for pid,x,y in [
        ("N_HIGHWAY", -20,300),
        ("S_HIGHWAY", 245,-500),
        ("GEUMGANG-RO", 480,225),
        ("WEAVING_4_LANES", 310,18),
        ("RAMP_OVER_HIGHWAY", 330,-170),
    ]:
        add(decor,"poi",id=pid,x=x,y=y,color="32,48,62",layer=20)
    save(directory/"highway.scenery.xml",decor)



def _ic_bridge_spans(net, road):
    """Find grade-separated crossings from the generated lane geometry.

    Road node sharing determines traffic connectivity; this function only draws
    decks. A crossing never creates an intersection or lane connection.
    """
    edges=[]
    for edge in net.findall("edge"):
        if edge.get("function")=="internal":continue
        lanes=edge.findall("lane")
        center=lanes[len(lanes)//2]
        pts=[tuple(map(float,p.split(","))) for p in center.get("shape").split()]
        pts=[p if len(p)==3 else (*p,0.) for p in pts]
        # Offset middle lane back to the carriageway center for an even lane count.
        if len(lanes)%2==0:pts=shifted(pts,float(center.get("width",road["lane_width"]))/2)
        edges.append((edge.get("id"),pts,sum(float(l.get("width",road["lane_width"])) for l in lanes)))
    out=[];seen=[]
    for i,(eid,ps,width) in enumerate(edges):
        for other,qs,ow in edges[i+1:]:
            for a,b in zip(ps,ps[1:]):
                rx,ry=b[0]-a[0],b[1]-a[1]
                for c,d in zip(qs,qs[1:]):
                    sx,sy=d[0]-c[0],d[1]-c[1]
                    den=rx*sy-ry*sx
                    if abs(den)<1e-9:continue
                    u=((c[0]-a[0])*sy-(c[1]-a[1])*sx)/den
                    v=((c[0]-a[0])*ry-(c[1]-a[1])*rx)/den
                    if not (0<=u<=1 and 0<=v<=1):continue
                    za=a[2]+u*(b[2]-a[2]);zb=c[2]+v*(d[2]-c[2])
                    if abs(za-zb)<3.5:continue
                    x,y=a[0]+u*rx,a[1]+u*ry
                    high,direction,deck_width,under_width,h=(eid,(rx,ry),width,ow,za) if za>zb else (other,(sx,sy),ow,width,zb)
                    if any(k==high and math.hypot(x-xx,y-yy)<2 for k,xx,yy in seen):continue
                    seen.append((high,x,y))
                    sine=abs(den)/(math.hypot(rx,ry)*math.hypot(sx,sy))
                    length=min(75,under_width/max(.25,sine)+7)
                    out.append(("bridge_"+str(len(out)),(x,y),direction,length,deck_width,h))
    return out
