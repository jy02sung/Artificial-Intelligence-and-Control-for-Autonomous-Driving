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
import xml.etree.ElementTree as ET

from sumolib import checkBinary  # SUMO 실행파일(sumo, netconvert 등) 경로를 찾아주는 유틸
from env.mdp_config import SIMULATION
try:
    from env.road_config import SCENERY
except ImportError:      # 구버전 설정 하위호환
    SCENERY = {"trees": False}


# 2D 뷰의 노면과 교량 상판에 같은 높이 색을 사용한다 (상대높이 m → RGB).
HEIGHT_COLORS = {0: "109,116,127", 2: "83,109,137", 7: "202,122,42", 12: "20,111,117"}


def build(road: dict, ego: dict, traffic: dict, out_dir: str) -> str:
    """SUMO 파일들을 out_dir에 생성하고, sumocfg 파일 경로를 반환한다.

    Args:
        road:    configs.road_config.ROAD    (도로 형상)
        ego:     configs.road_config.EGO     (RL 차량 속성)
        traffic: configs.road_config.TRAFFIC (배경 교통류)
        out_dir: 생성 위치 (보통 프로젝트의 sumo/ 폴더)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ROAD에 좌표가 없는 종전 직선 도로 설정도 같은 형식으로 생성한다.
    nodes = road.get("nodes", {"n0": (0, 0, 0), "n1": (road["length"], 0, 0)})
    edges = road.get("edges", {"e0": ("n0", "n1", "highway", [])})
    types = {"highway": (road["num_lanes"], road["speed_limit"]), **road.get("types", {})}

    def save_xml(root, filename):
        path = os.path.join(out_dir, filename)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        return path

    node_xml = ET.Element("nodes")
    for name, (x, y, z) in nodes.items():
        ET.SubElement(node_xml, "node", id=name, x=str(x), y=str(y), z=str(z))
    node_path = save_xml(node_xml, "highway.nod.xml")

    edge_xml = ET.Element("edges")
    for name, (start, end, kind, via) in edges.items():
        lanes, speed = types[kind]
        points = [nodes[start], *via, nodes[end]]
        shape = " ".join(f"{x},{y},{z}" for x, y, z in points)
        ET.SubElement(edge_xml, "edge", {
            "id": name, "from": start, "to": end, "numLanes": str(lanes),
            "speed": str(speed), "shape": shape,
            "priority": str(lanes), "spreadType": "center" if "nodes" in road else "right",
        })
    edge_path = save_xml(edge_xml, "highway.edg.xml")

    connection_args = []
    if road.get("connections"):
        con_xml = ET.Element("connections")
        for before, lane_in, after, lane_out in road["connections"]:
            ET.SubElement(con_xml, "connection", {
                "from": before, "to": after, "fromLane": str(lane_in), "toLane": str(lane_out),
            })
        connection_args = ["--connection-files", save_xml(con_xml, "highway.con.xml")]

    # ──────────────────────────────────────────────────────
    # 3) 라우트 파일: 차량 타입 2종 + 배경 교통 flow
    #
    #    <vType>  : 차량의 물리 속성 템플릿
    #       ego_type : RL이 조종할 차 (빨간색, color="1,0,0" = RGB)
    #    <vTypeDistribution> : 배경차 컨트롤러 믹스.
    #       TRAFFIC["controllers"]에 정의된 여러 차량추종모델(Krauss/IDM/
    #       EIDM/ACC 등)을 probability 비율로 섞는다. flow가 이 분포를
    #       type으로 쓰면, 투입되는 차량마다 SUMO가 확률적으로 타입을 뽑는다.
    #    <route>  : 경로별로 지나갈 엣지들의 나열.
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
            # 개별 컨트롤러 설정이 있으면 공통 차로 유지 설정보다 우선한다.
            if c.get("keep_lane", traffic.get("keep_lane", True)):
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

    route_xml = ET.Element("routes")
    ET.SubElement(route_xml, "vType", {
        "id": "ego_type", "accel": str(ego["accel"]), "decel": str(ego["decel"]),
        "maxSpeed": str(ego["max_speed"]), "color": "1,0,0",
        "vClass": "passenger", "guiShape": "passenger/sedan",
    })
    route_xml.append(ET.fromstring(vtypes_block))
    routes = road.get("routes", {"r0": ("e0", 1)})
    total_weight = sum(weight for _, weight in routes.values())
    for name, (route_edges, _) in routes.items():
        ET.SubElement(route_xml, "route", id=name, edges=route_edges)
    for name, (_, weight) in routes.items():
        rate = traffic["vehs_per_hour"] * weight / total_weight
        if rate > 0:
            ET.SubElement(route_xml, "flow", {
                "id": f"traffic_{name}" if "routes" in road else "traffic", "type": flow_type, "route": name,
                "begin": "0", "end": "100000", "period": f"{3600 / rate:.2f}",
                "departLane": str(traffic["depart_lane"]), "departSpeed": str(traffic["depart_speed"]),
            })
    save_xml(route_xml, "highway.rou.xml")

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
    viewport = "" if "nodes" in road else (
        f'    <viewport zoom="{zoom:.0f}" x="{center_x:.1f}" y="{center_y:.1f}" angle="0"/>\n'
    )  # 다중 도로망은 SUMO가 시작 화면에 전체 범위를 맞춘다.
    scheme = '<scheme name="real world"/>'
    if road.get("bridges"):
        entries = "\n".join(
            f'                <entry threshold="{height}" color="{color}"/>'
            for height, color in HEIGHT_COLORS.items()
        )
        # SUMO 1.27: lane mode 22 = 각 형상 구간의 높이, junction mode 3 = 노드 높이.
        scheme = f'''<scheme name="Toegyewon - elevation (m)">
        <opengl trueZ="1"/>
        <background backgroundColor="246,246,241"/>
        <edges laneEdgeMode="22" laneShowBorders="0" showLinkDecals="0" showLinkRules="0">
            <colorScheme name="by height at geometry-segment start" interpolated="1">
{entries}
            </colorScheme>
        </edges>
        <junctions junctionMode="3">
            <colorScheme name="by height" interpolated="1">
{entries}
            </colorScheme>
        </junctions>
        <vehicles vehicleMode="0" vehicleQuality="2"/>
        <legend showColorLegend="1"/>
    </scheme>'''
    gui_settings_path = os.path.join(out_dir, "highway.gui.xml")
    with open(gui_settings_path, "w") as f:
        f.write(f"""<viewsettings>
    {scheme}
{viewport}    <delay value="{gui_delay}"/>
</viewsettings>
""")

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
         "--node-files", node_path,
         "--edge-files", edge_path, *connection_args,
         "-o", os.path.join(out_dir, "highway.net.xml")],
        check=True,           # 실패하면 예외를 던져서 바로 알 수 있게
        stdout=subprocess.DEVNULL,  # 실패 이유와 형상 경고는 stderr로 보인다.
    )
    _write_scenery(out_dir, road)  # 변환된 차로 좌표에 맞춰 교량 표시를 배치한다.
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


def _bridge_markings(out_dir, bridges):
    """교차점 양쪽으로 10 m씩 연장한 상판과 난간을 GUI 추가 도형으로 만든다."""
    from sumolib import geomhelper, net

    network = net.readNet(os.path.join(out_dir, "highway.net.xml"))
    polygons = []
    for upper, lower_edges, height in bridges:
        deck = network.getEdge(upper)
        center = deck.getShape()
        offsets = []
        for lower in lower_edges:
            offsets.extend(geomhelper.intersectsAtLengths2D(center, network.getEdge(lower).getShape()))
        if not offsets:
            continue
        start = max(0, min(offsets) - 10)
        end = min(geomhelper.polyLength(center), max(offsets) + 10)
        parts = geomhelper.splitPolygonAtLengths2D(center, [start, end])
        span = parts[1 if start > 0 else 0]
        width = sum(lane.getWidth() for lane in deck.getLanes())
        # 레이어는 지상 차량보다 높고 상부 차량보다 낮아 지나는 순서가 보인다.
        for part, extra_width, layer, color in (
            ("rail", 2.8, height - 0.4, "53,61,66"),
            ("deck", 0, height - 0.2, HEIGHT_COLORS[height]),
        ):
            outline = geomhelper.line2boundary(span, width + extra_width)
            shape = " ".join(f"{x:.2f},{y:.2f}" for x, y in outline)
            polygons.append(f'    <poly id="bridge_{upper}_{part}" type="bridge" '
                            f'color="{color}" layer="{layer}" fill="1" shape="{shape}"/>')
    return polygons


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
    bridges = road.get("bridges", [])
    markings = _bridge_markings(out_dir, bridges) if bridges else []
    if not cfg.get("trees", False):
        with open(path, "w") as f:
            f.write("\n".join(["<additional>", *markings, "</additional>"]) + "\n")
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

    lines = ["<additional>", *markings]
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
