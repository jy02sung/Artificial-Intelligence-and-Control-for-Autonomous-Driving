"""Presentation GUI rendering, live 2D occlusion overlay, and demo scenario builders."""
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None

from utils.extend_north_entry import extend

VEHICLE_TYPE_COLORS = {
    "car_krauss": "0.82,0.84,0.88",
    "car_idm": "0.82,0.84,0.88",
    "car_eidm": "0.82,0.84,0.88",
    "car_acc": "0.82,0.84,0.88",
    "car_krauss_violator": "1.0,0.52,0.12",
    "car_idm_violator": "1.0,0.52,0.12",
    "car_eidm_violator": "1.0,0.52,0.12",
    "car_acc_violator": "1.0,0.52,0.12",
    "delivery_idm": "0.18,0.78,0.92",
    "truck_idm": "0.98,0.78,0.10",
    "truck_slow": "0.98,0.62,0.05",
    "bus_idm": "0.18,0.45,0.98",
    "bus_slow": "0.12,0.32,0.88",
    "ego_type": "1.0,0.20,0.25",
}

PASSENGER_GUI_SHAPES = {
    "car_krauss": "passenger/sedan",
    "car_idm": "passenger/sedan",
    "car_eidm": "passenger/hatchback",
    "car_acc": "passenger/wagon",
}

# Opposite far-end starting lane per route to force multi-lane cross-weaving.
CROSS_DEPART_LANES = {
    "route_01": "best",
    "route_02": "best",
    "route_03": "3",
    "route_04": "3",
    "route_05": "3",
    "route_06": "0",
    "route_07": "2",
    "route_08": "0",
    "route_09": "best",
    "route_10": "1",
    "route_11": "1",
    "route_12": "1",
    "route_13": "4",
    "route_14": "4",
    "route_15": "4",
    "route_16": "0",
}

# Showcase scripted starting lanes (SUMO lane 0 = rightmost, N-1 = leftmost).
SCRIPTED_DEPART_LANES = {
    **CROSS_DEPART_LANES,
    "route_05": "2",  # 4-lane middle-left: allows both right/left eye-contact and overtaking
    "route_13": "4",  # 5-lane far-left (Korean lane 1): 5-lane full weave + X-crossing
    "route_08": "2",  # 3-lane leftmost (Korean lane 1): natural rightward slow-truck overtake -> merge -> zipper
}

# Tune per-entrance density within lane capacity so every route stays dense and flowing without parking-lot gridlock.
DEMO_FLOW_COUNTS = {
    "flow_route_01": 450,
    "flow_route_02": 450,
    "flow_route_03": 600,
    "flow_route_04": 600,
    "flow_route_05": 600,
    "flow_route_06": 4650,
    "flow_route_07": 1050,
    "flow_route_08": 1050,
    "flow_route_09": 1400,
    "flow_route_10": 550,
    "flow_route_11": 550,
    "flow_route_12": 550,
    "flow_route_13": 450,
    "flow_route_14": 450,
    "flow_route_15": 450,
    "flow_route_16": 3900,
}

# Extend multi-lane straight approaches to ~850-1050m for long overtaking interactions.
DEMO_OVERTAKE_EXTENSIONS = {
    "52873356": 560,
    "86198728": 560,
    "544295537": 500,
    "781184662#1": 450,
}


def write_demo_gui_settings(dest_dir: Path, delay_ms: float = 60.0) -> Path:
    """Write SUMO GUI view-settings using SUMO's built-in 'real world' scheme (asphalt roads + detailed vehicle bodies)."""
    gui_path = dest_dir / "highway.gui.xml"
    view = ET.Element("viewsettings")
    ET.SubElement(view, "scheme", {"name": "real world"})
    ET.SubElement(view, "delay", {"value": str(float(delay_ms))})
    tree = ET.ElementTree(view)
    tree.write(str(gui_path), encoding="utf-8", xml_declaration=True)
    return gui_path


def attach_gui_settings_to_sumocfg(cfg_path: Path, gui_filename: str = "highway.gui.xml") -> None:
    """Ensure <gui-settings-file> points to highway.gui.xml inside run.sumocfg."""
    tree = ET.parse(str(cfg_path))
    root = tree.getroot()
    gui_only = root.find("gui_only")
    if gui_only is None:
        gui_only = ET.SubElement(root, "gui_only")
    gsf = gui_only.find("gui-settings-file")
    if gsf is None:
        gsf = ET.SubElement(gui_only, "gui-settings-file")
    gsf.set("value", gui_filename)
    tree.write(str(cfg_path), encoding="utf-8", xml_declaration=True)


def apply_real_vehicle_colors_and_shapes(root: ET.Element) -> None:
    """Replace triangle defaults with passenger sedan/hatchback/wagon shapes and distinct colors."""
    for flow in root.findall("flow"):
        flow.attrib.pop("color", None)
    for vt in root.findall("vType"):
        vid = vt.get("id", "")
        if vid in PASSENGER_GUI_SHAPES and not vt.get("guiShape"):
            vt.set("guiShape", PASSENGER_GUI_SHAPES[vid])
        if vid in VEHICLE_TYPE_COLORS:
            vt.set("color", VEHICLE_TYPE_COLORS[vid])


def configure_dense_or_overtake_flows(root: ET.Element, include_slow_trucks: bool = True) -> None:
    """Increase per-entrance density, spread vehicles across all lanes, and mix slow trucks/buses."""
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
        "tau": "1.5", "minGap": "2.8", "maxSpeed": "23.5", "speedDev": "0.08",
    })
    truck_slow = ET.Element("vType", {
        "id": "truck_slow", "vClass": "truck", "guiShape": "truck",
        "length": "9.5", "width": "2.5", "carFollowModel": "IDM",
        "accel": "0.9", "decel": "3.0", "emergencyDecel": "7",
        "tau": "1.6", "minGap": "3.0", "maxSpeed": "14.5", "speedDev": "0.05",
    })
    bus_slow = ET.Element("vType", {
        "id": "bus_slow", "vClass": "bus", "guiShape": "bus",
        "length": "12.0", "width": "2.5", "carFollowModel": "IDM",
        "accel": "1.0", "decel": "3.2", "emergencyDecel": "7",
        "tau": "1.6", "minGap": "3.0", "maxSpeed": "15.5", "speedDev": "0.05",
    })
    dist = root.find("vTypeDistribution")
    if dist is not None:
        idx = list(root).index(dist)
        new_elems = [deliv, truck] + ([truck_slow, bus_slow] if include_slow_trucks else [])
        for offset, elem in enumerate(new_elems):
            root.insert(idx + offset, elem)
        if include_slow_trucks:
            dist.set(
                "vTypes",
                "car_krauss car_idm car_eidm car_acc "
                "car_krauss_violator car_idm_violator car_eidm_violator car_acc_violator "
                "delivery_idm truck_idm truck_slow bus_idm bus_slow",
            )
            dist.set(
                "probabilities",
                "0.220 0.200 0.130 0.110 0.015 0.015 0.010 0.010 0.175 0.045 0.035 0.015 0.020",
            )
        else:
            dist.set(
                "vTypes",
                "car_krauss car_idm car_eidm car_acc "
                "car_krauss_violator car_idm_violator car_eidm_violator car_acc_violator "
                "delivery_idm truck_idm bus_idm",
            )
            dist.set(
                "probabilities",
                "0.220 0.200 0.130 0.110 0.015 0.015 0.010 0.010 0.200 0.065 0.025",
            )
    for flow in root.findall("flow"):
        fid = flow.get("id", "")
        if fid in DEMO_FLOW_COUNTS:
            flow.set("number", str(DEMO_FLOW_COUNTS[fid]))
        flow.set("departLane", "random")
        flow.set("departSpeed", "desired")


def configure_scripted_flows(root: ET.Element) -> None:
    """Keep light-medium background flow (~45%) so roads feel alive without jamming scripted actors."""
    configure_dense_or_overtake_flows(root, include_slow_trucks=True)
    for flow in root.findall("flow"):
        num = int(flow.get("number", "300"))
        flow.set("number", str(max(120, round(num * 0.45))))


def build_demo_overtake_network(base_net: Path, dest_net: Path) -> None:
    """Extend straight multi-lane entry edges to ~850-1050m using existing extend helper."""
    shutil.copy2(str(base_net), str(dest_net))
    for i, (edge_id, extra_m) in enumerate(DEMO_OVERTAKE_EXTENSIONS.items()):
        src = base_net if i == 0 else dest_net
        extend(src, dest_net, distance=extra_m, edge_id=edge_id)


def _shadow_quad_from_box(
    sx: float, sy: float, vis: float,
    cx: float, cy: float, ux: float, uy: float, rx: float, ry: float, hl: float, hw: float,
):
    """Build a 4-vertex polygon projecting an occluder's shadow out to radius `vis` from sensor S."""
    corners = (
        (cx + hl * ux + hw * rx, cy + hl * uy + hw * ry),
        (cx + hl * ux - hw * rx, cy + hl * uy - hw * ry),
        (cx - hl * ux - hw * rx, cy - hl * uy - hw * ry),
        (cx - hl * ux + hw * rx, cy - hl * uy + hw * ry),
    )
    center_ang = math.atan2(cy - sy, cx - sx)
    with_ang = []
    for px, py in corners:
        ang = ((math.atan2(py - sy, px - sx) - center_ang + math.pi) % (2.0 * math.pi)) - math.pi
        dist = math.hypot(px - sx, py - sy)
        with_ang.append((ang, dist, px, py))
    left_most = max(with_ang, key=lambda item: item[0])
    right_most = min(with_ang, key=lambda item: item[0])
    far_pts = []
    for _, _, px, py in (left_most, right_most):
        dx, dy = px - sx, py - sy
        norm = max(1e-6, math.hypot(dx, dy))
        far_pts.append((sx + (dx / norm) * vis, sy + (dy / norm) * vis))
    # Polygon order: near-left -> far-left -> far-right -> near-right
    return [
        (left_most[2], left_most[3]),
        far_pts[0],
        far_pts[1],
        (right_most[2], right_most[3]),
    ]


class DemoOcclusionOverlay:
    """Manages subtle occlusion cones/markers and a clean live situation caption badge in SUMO-GUI."""

    def __init__(self):
        self._drawn_ids = set()
        self._badge_cache = {}
        self._active_caption = None
        self._badge_dir = Path(tempfile.gettempdir()) / "mcw_hud_badges"

    def clear(self, traci_mod) -> None:
        for pid in list(self._drawn_ids):
            try:
                traci_mod.polygon.remove(pid)
            except Exception:
                pass
        self._drawn_ids.clear()
        self._active_caption = None
        try:
            traci_mod.poi.remove("mcw_caption_poi")
        except Exception:
            pass

    def _get_badge_image(self, text: str):
        if not text or Image is None:
            return None
        cached = self._badge_cache.get(text)
        if cached and cached[0].exists():
            return cached
        self._badge_dir.mkdir(parents=True, exist_ok=True)
        fname = f"badge_{zlib.crc32(text.encode('utf-8')):08x}.png"
        out_path = self._badge_dir / fname
        font = None
        for fp in (r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf"):
            if Path(fp).exists():
                try:
                    font = ImageFont.truetype(fp, 34)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        dummy = Image.new("RGBA", (10, 10))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = 22, 13
        bw, bh = tw + pad_x * 2, th + pad_y * 2
        img = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(
            [(0, 0), (bw - 1, bh - 1)],
            radius=12,
            fill=(18, 24, 34, 215),
            outline=(195, 208, 224, 150),
            width=2,
        )
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(246, 249, 252, 255))
        img.save(str(out_path))
        self._badge_cache[text] = (out_path, bw, bh)
        return self._badge_cache[text]

    def _upsert(self, traci_mod, pid: str, shape, color, fill: bool, layer: int, line_width: float = 1.0) -> None:
        if pid in self._drawn_ids:
            try:
                traci_mod.polygon.setShape(pid, shape)
                try:
                    traci_mod.polygon.setColor(pid, color)
                except Exception:
                    pass
                return
            except Exception:
                self._drawn_ids.discard(pid)
        try:
            traci_mod.polygon.add(pid, shape, color, fill=fill, polygonType="mcw_fov", layer=layer, lineWidth=line_width)
            self._drawn_ids.add(pid)
        except Exception:
            pass

    def render(
        self,
        traci_mod,
        sx: float,
        sy: float,
        visibility: float,
        occ_boxes,
        occ_targets,
        ex: Optional[float] = None,
        ey: Optional[float] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Draw subtle occlusion shadow cones, hidden vehicle markers, and a small scenario caption badge."""
        active_ids = set()

        # 1) Subtle occlusion shadow cones behind opaque heavy vehicles (layer=3: under vehicles)
        for idx, (_vid, cx, cy, ux, uy, rx, ry, hl, hw) in enumerate(occ_boxes):
            quad = _shadow_quad_from_box(sx, sy, visibility, cx, cy, ux, uy, rx, ry, hl, hw)
            pid = f"mcw_occ_cone_{idx}"
            self._upsert(traci_mod, pid, quad, (215, 70, 80, 62), fill=True, layer=3, line_width=0.3)
            active_ids.add(pid)

        # 2) Thin magenta outline rings on vehicles hidden behind trucks/buses
        for idx, (tcx, tcy) in enumerate(occ_targets):
            r = 3.1
            ring_t = [
                (tcx + r * math.cos(2.0 * math.pi * k / 12), tcy + r * math.sin(2.0 * math.pi * k / 12))
                for k in range(12)
            ]
            ring_t.append(ring_t[0])
            pid = f"mcw_occ_tgt_{idx}"
            self._upsert(traci_mod, pid, ring_t, (230, 85, 255, 200), fill=False, layer=6, line_width=0.45)
            active_ids.add(pid)

        # 3) Clean, small situation caption badge offset neatly to the side of ego (no circles/bars/arrows)
        if caption and ex is not None and ey is not None:
            badge_info = self._get_badge_image(caption)
            if badge_info is not None:
                badge_path, bw, bh = badge_info
                scale = 0.16
                px = float(ex) + (bw * scale) * 0.5 + 16.0
                py = float(ey) + 8.0
                if self._active_caption != caption:
                    try:
                        traci_mod.poi.remove("mcw_caption_poi")
                    except Exception:
                        pass
                    try:
                        traci_mod.poi.add(
                            "mcw_caption_poi", px, py, (255, 255, 255, 255),
                            poiType="hud", layer=10, imgFile=str(badge_path),
                            width=bw * scale, height=bh * scale,
                        )
                        self._active_caption = caption
                    except Exception:
                        pass
                else:
                    try:
                        traci_mod.poi.setPosition("mcw_caption_poi", px, py)
                    except Exception:
                        self._active_caption = None

        stale = self._drawn_ids - active_ids
        for pid in list(stale):
            try:
                traci_mod.polygon.remove(pid)
            except Exception:
                pass
            self._drawn_ids.discard(pid)


class ScriptedDemoDirector:
    """Spawns and steers scripted HV actors on highlight routes to trigger overtaking, eye-contact merges, and X-crossing congestion."""

    def __init__(self):
        self.route = ""
        self._fired = set()
        self._speed_caps = {}
        self._keep_lanes = set()
        self._cutin_plan = None
        self._last_caption = ""
        self._last_caption_step = -999

    def reset(self, route: str) -> None:
        self.route = str(route or "")
        self._fired = set()
        self._speed_caps = {}
        self._keep_lanes = set()
        self._cutin_plan = None
        self._last_caption = ""
        self._last_caption_step = -999

    def get_status_caption(
        self,
        traci_mod,
        step_count: int,
        ego_id: str,
        occ_count: int = 0,
        lc_cmd: int = 0,
        lead_gap: Optional[float] = None,
    ) -> str:
        """Return a concise situational caption explaining current scenario intent & negotiation."""
        try:
            if ego_id not in traci_mod.vehicle.getIDList():
                return self._last_caption or ""
            edge = traci_mod.vehicle.getRoadID(ego_id)
            pos = float(traci_mod.vehicle.getLanePosition(ego_id))
            v = float(traci_mod.vehicle.getSpeed(ego_id))
        except Exception:
            return self._last_caption or ""

        raw_caption = "본선 주행 · 목표 경로 순항"
        if self.route == "route_05":
            if edge == "52873356":
                if pos < 210.0:
                    if occ_count > 0 or v < 21.0:
                        raw_caption = "눈치싸움 · 버스 뒤 사각지대 경계 후 추월"
                    else:
                        raw_caption = "추월 · 전방 저속 트럭 추월 시도"
                elif pos < 470.0:
                    if "r05_cutin_spawn" in self._fired:
                        raw_caption = "컷인 대응 · 끼어든 차량과 안전거리 유지"
                    else:
                        raw_caption = "가속 추월 · 본선 속도 회복"
                else:
                    raw_caption = "눈치싸움 · 우측 출구 호송열 틈새 합류 대기"
            elif edge in ("37927512", "170059991", "170059994") or edge.startswith(":1000493460") or edge.startswith(":1812112753"):
                raw_caption = "X자 교차로 · 좌측 합류 차량과 엇갈리며 통과"
            else:
                raw_caption = "진출 완료 · 목표 경로 순항"

        elif self.route == "route_13":
            if edge == "86198728":
                if pos < 270.0:
                    if occ_count > 0 or v < 19.0:
                        raw_caption = "눈치싸움 · 버스 뒤 숨은 차·트럭 피하며 차로 횡단"
                    else:
                        raw_caption = "추월 · 1차로 저속 트럭 우측 추월"
                else:
                    raw_caption = "다단 직조 · 다차로 플라툰 간격 보며 주행"
            elif edge in ("170059996", "170059995", "170059994") or edge.startswith(":1812112753") or edge.startswith(":436249330"):
                raw_caption = "X자 교차로 · 교차 차량 양보 후 우측 차로 진입"
            elif edge == "1063527748":
                raw_caption = "합류 정리 · 진출 차로 확보"
            else:
                raw_caption = "진출 완료 · 목표 경로 순항"

        elif self.route == "route_08":
            if edge == "544295537":
                if pos < 270.0:
                    if v < 15.0 or (lead_gap is not None and lead_gap < 40.0):
                        raw_caption = "눈치싸움 · 저속 트럭 뒤 감속 후 우측 틈 확인"
                    elif lc_cmd != 0:
                        raw_caption = "추월 · 1차로 저속 트럭 우측 추월"
                    else:
                        raw_caption = "추월 준비 · 전방 트럭 및 옆 차로 간격 탐색"
                else:
                    raw_caption = "합류 눈치싸움 · 우측 대기열 우회 후 차선 진입"
            elif edge in ("169929488#0", "169929488#1", "544292616") or edge.startswith(":1808903"):
                raw_caption = "램프 연결 · 본선 합류 흐름 합승"
            elif edge in ("844484877",) or edge.startswith(":844484877"):
                raw_caption = "지퍼 합류 · 합류 차량과 교대 진입 대기"
            else:
                raw_caption = "진출 완료 · 지퍼 합류 통과"
        else:
            if occ_count > 0 and lead_gap is not None and lead_gap < 45.0:
                raw_caption = "사각지대 경계 · 대형차 뒤 숨은 차량 주의"
            elif lc_cmd != 0:
                raw_caption = "차선 변경 · 안전 간격 확보 후 이동"

        # Hold each caption for >= 18 steps (~3.6s) so text never flickers in video
        if not self._last_caption or (raw_caption != self._last_caption and (step_count - self._last_caption_step) >= 18):
            self._last_caption = raw_caption
            self._last_caption_step = int(step_count)
        return self._last_caption

    def _resolve_subroute(self, traci_mod, route_id: str, edge_id: str) -> str:
        try:
            edges = list(traci_mod.route.getEdges(route_id))
            if not edges or edges[0] == edge_id or edge_id not in edges:
                return route_id
            sub_rid = f"__sub_{route_id}_{edge_id}"
            if sub_rid not in traci_mod.route.getIDList():
                traci_mod.route.add(sub_rid, edges[edges.index(edge_id):])
            return sub_rid
        except Exception:
            return route_id

    def _clear_corridor(self, traci_mod, edge_id: str, lane_idx: int, pos_min: float, pos_max: float, ego_id: str) -> None:
        try:
            lane_id = f"{edge_id}_{lane_idx}"
            for other in list(traci_mod.lane.getLastStepVehicleIDs(lane_id)):
                if other == ego_id or other.startswith("act"):
                    continue
                opos = float(traci_mod.vehicle.getLanePosition(other))
                if pos_min <= opos <= pos_max:
                    try:
                        traci_mod.vehicle.remove(other)
                    except Exception:
                        pass
        except Exception:
            pass

    def _spawn(self, traci_mod, vid: str, route_id: str, type_id: str,
               edge_id: str, lane_idx: int, pos: float, speed: float,
               speed_cap: float = None, ego_id: str = "ego",
               keep_lane: bool = True, clear_radius: float = 24.0) -> bool:
        try:
            lane_id = f"{edge_id}_{lane_idx}"
            lane_len = float(traci_mod.lane.getLength(lane_id))
            pos = max(2.0, min(pos, lane_len - 12.0))
            self._clear_corridor(traci_mod, edge_id, lane_idx, pos - clear_radius, pos + clear_radius, ego_id)
            spawn_route = self._resolve_subroute(traci_mod, route_id, edge_id)
            try:
                max_v = float(traci_mod.vehicletype.getMaxSpeed(type_id))
                ds = "desired" if pos > lane_len - 55.0 else f"{min(float(speed), max(1.0, max_v * 0.98)):.1f}"
            except Exception:
                ds = "desired"
            traci_mod.vehicle.add(
                vid, spawn_route, typeID=type_id, depart="now",
                departLane=str(lane_idx), departPos=f"{pos:.1f}", departSpeed=ds,
            )
            self._speed_caps[vid] = float(speed_cap if speed_cap is not None else speed)
            if keep_lane:
                self._keep_lanes.add(vid)
            return True
        except Exception:
            return False

    def _apply_pending_caps(self, traci_mod) -> None:
        if not self._speed_caps:
            return
        active = set(traci_mod.vehicle.getIDList())
        for vid in list(self._speed_caps):
            if vid in active:
                cap = self._speed_caps.pop(vid)
                try:
                    traci_mod.vehicle.setSpeedMode(vid, 31)
                    traci_mod.vehicle.setMaxSpeed(vid, cap)
                    if vid in self._keep_lanes:
                        # 514: strategic-only + collision-avoiding (allows downstream route LCs, stops random lane drift)
                        traci_mod.vehicle.setLaneChangeMode(vid, 514)
                        self._keep_lanes.discard(vid)
                except Exception:
                    pass

    def step(self, traci_mod, step_count: int, ego_id: str = "ego") -> None:
        self._apply_pending_caps(traci_mod)
        try:
            if ego_id not in traci_mod.vehicle.getIDList():
                return
            edge = traci_mod.vehicle.getRoadID(ego_id)
            if edge.startswith(":"):
                return
            pos = float(traci_mod.vehicle.getLanePosition(ego_id))
            lane = int(traci_mod.vehicle.getLaneIndex(ego_id))
        except Exception:
            return

        if self.route == "route_05":
            self._step_route_05(traci_mod, step_count, edge, lane, pos, ego_id)
        elif self.route == "route_13":
            self._step_route_13(traci_mod, step_count, edge, lane, pos, ego_id)
        elif self.route == "route_08":
            self._step_route_08(traci_mod, step_count, edge, lane, pos, ego_id)

    def _step_route_05(self, traci_mod, step_count: int, edge: str, lane: int, pos: float, ego_id: str) -> None:
        if "r05_init" not in self._fired and edge == "52873356":
            self._fired.add("r05_init")
            self._clear_corridor(traci_mod, "52873356", 2, pos + 8.0, pos + 250.0, ego_id)
            self._clear_corridor(traci_mod, "52873356", 1, 0.0, pos + 260.0, ego_id)
            self._clear_corridor(traci_mod, "52873356", 3, 0.0, pos + 160.0, ego_id)
            self._spawn(traci_mod, "act05_bus_r", "route_06", "bus_idm", "52873356", 1, pos + 18.0, 16.5, 16.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act05_hid_car", "route_06", "car_idm", "52873356", 1, pos + 65.0, 14.5, 14.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act05_pass_l", "route_06", "car_krauss_violator", "52873356", 3, pos + 8.0, 27.5, 27.5, ego_id, keep_lane=True)

        if "r05_truck" not in self._fired and edge == "52873356" and pos >= 35.0:
            self._fired.add("r05_truck")
            self._clear_corridor(traci_mod, "52873356", 2, pos + 15.0, pos + 220.0, ego_id)
            self._spawn(traci_mod, "act05_truck", "route_06", "truck_slow", "52873356", 2, pos + 100.0, 10.0, 8.8, ego_id, keep_lane=True)

        if "r05_cutin_spawn" not in self._fired and edge == "52873356" and pos >= 310.0:
            self._fired.add("r05_cutin_spawn")
            try:
                ego_v = float(traci_mod.vehicle.getSpeed(ego_id))
            except Exception:
                ego_v = 18.0
            spawn_v = max(14.5, min(20.0, ego_v + 1.5))
            adj = 1 if lane == 0 else (lane + 1 if lane in (1, 2) else 2)
            if self._spawn(traci_mod, "act05_cutin", "route_06", "car_eidm_violator", "52873356", adj, pos + 42.0, spawn_v, spawn_v, ego_id, keep_lane=False):
                self._cutin_plan = ("act05_cutin", step_count + 5, 14.5)

        if self._cutin_plan and step_count >= self._cutin_plan[1]:
            cid, _, slow_spd = self._cutin_plan
            self._cutin_plan = None
            try:
                if cid in traci_mod.vehicle.getIDList():
                    cur_ego_lane = int(traci_mod.vehicle.getLaneIndex(ego_id))
                    traci_mod.vehicle.setLaneChangeMode(cid, 512)
                    traci_mod.vehicle.changeLane(cid, cur_ego_lane, 2.5)
                    traci_mod.vehicle.slowDown(cid, slow_spd, 1.5)
            except Exception:
                pass

        if "r05_convoy" not in self._fired and edge == "52873356" and pos >= 520.0:
            self._fired.add("r05_convoy")
            self._spawn(traci_mod, "act05_conv1", "route_05", "bus_idm", "52873356", 0, pos + 170.0, 16.0, 16.0, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act05_conv2", "route_05", "car_idm", "52873356", 0, pos + 132.0, 16.0, 16.0, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act05_conv3", "route_05", "delivery_idm", "52873356", 0, pos + 88.0, 16.0, 16.0, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act05_conv4", "route_06", "car_eidm", "52873356", 1, pos + 108.0, 16.5, 16.5, ego_id, keep_lane=True)

        if "r05_xcross" not in self._fired and edge == "170059991" and pos >= 10.0:
            self._fired.add("r05_xcross")
            self._clear_corridor(traci_mod, "170059991", 2, pos + 12.0, 128.0, ego_id)
            self._clear_corridor(traci_mod, "170059994", 3, 0.0, 51.0, ego_id)
            self._spawn(traci_mod, "act05_weave", "route_03", "truck_idm", "170059991", 1, 62.0, 8.5, 8.0, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act05_x1", "route_13", "delivery_idm", "170059995", 0, 34.0, 12.5, 12.5, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act05_x2", "route_13", "car_idm_violator", "170059995", 0, 80.0, 13.0, 13.0, ego_id, keep_lane=False)

    def _step_route_13(self, traci_mod, step_count: int, edge: str, lane: int, pos: float, ego_id: str) -> None:
        if "r13_init" not in self._fired and edge == "86198728":
            self._fired.add("r13_init")
            self._clear_corridor(traci_mod, "86198728", 4, pos + 8.0, pos + 240.0, ego_id)
            self._clear_corridor(traci_mod, "86198728", 3, 0.0, pos + 240.0, ego_id)
            self._clear_corridor(traci_mod, "86198728", 2, 0.0, pos + 240.0, ego_id)
            self._clear_corridor(traci_mod, "86198728", 1, 0.0, pos + 160.0, ego_id)
            self._spawn(traci_mod, "act13_side", "route_16", "car_krauss_violator", "86198728", 3, pos + 22.0, 27.5, 27.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act13_p3b", "route_16", "bus_slow", "86198728", 3, pos + 72.0, 16.5, 16.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act13_hid", "route_16", "car_idm", "86198728", 2, pos + 112.0, 15.0, 15.0, ego_id, keep_lane=True)

        if "r13_truck" not in self._fired and edge == "86198728" and pos >= 30.0:
            self._fired.add("r13_truck")
            self._clear_corridor(traci_mod, "86198728", 4, pos + 15.0, pos + 220.0, ego_id)
            self._spawn(traci_mod, "act13_truck", "route_16", "truck_slow", "86198728", 4, pos + 98.0, 10.2, 9.2, ego_id, keep_lane=True)

        if "r13_weave" not in self._fired and edge == "86198728" and pos >= 300.0:
            self._fired.add("r13_weave")
            self._spawn(traci_mod, "act13_m2a", "route_16", "bus_slow", "86198728", 2, pos + 120.0, 15.0, 15.0, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act13_m2b", "route_16", "car_idm", "86198728", 2, pos + 72.0, 15.0, 15.0, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act13_m1a", "route_16", "delivery_idm", "86198728", 1, pos + 98.0, 15.0, 15.0, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act13_m1b", "route_16", "car_eidm", "86198728", 1, pos + 52.0, 15.5, 15.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act13_r0a", "route_16", "bus_idm", "86198728", 0, pos + 175.0, 14.5, 14.5, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act13_r0b", "route_13", "car_eidm", "86198728", 0, pos + 135.0, 18.0, 18.0, ego_id, keep_lane=False)

        if "r13_xcross" not in self._fired and edge == "170059996":
            self._fired.add("r13_xcross")
            self._clear_corridor(traci_mod, "170059994", 2, 0.0, 51.0, ego_id)
            self._clear_corridor(traci_mod, "170059991", 2, 0.0, 128.0, ego_id)
            self._spawn(traci_mod, "act13_x1", "route_05", "delivery_idm", "170059991", 2, 35.0, 16.0, 16.0, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act13_x2", "route_05", "car_idm", "170059991", 2, 85.0, 17.0, 17.0, ego_id, keep_lane=False)

    def _step_route_08(self, traci_mod, step_count: int, edge: str, lane: int, pos: float, ego_id: str) -> None:
        if "r08_init" not in self._fired and edge == "544295537":
            self._fired.add("r08_init")
            self._clear_corridor(traci_mod, "544295537", 2, pos + 8.0, pos + 320.0, ego_id)
            self._clear_corridor(traci_mod, "544295537", 1, 0.0, pos + 280.0, ego_id)
            self._clear_corridor(traci_mod, "544295537", 0, 0.0, pos + 180.0, ego_id)
            self._spawn(traci_mod, "act08_bus", "route_08", "bus_idm", "544295537", 1, pos + 14.0, 19.5, 19.5, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act08_hid", "route_08", "car_idm", "544295537", 1, pos + 80.0, 17.5, 17.5, ego_id, keep_lane=True)

        if "r08_truck" not in self._fired and edge == "544295537" and pos >= 40.0:
            self._fired.add("r08_truck")
            self._clear_corridor(traci_mod, "544295537", 2, pos + 15.0, pos + 220.0, ego_id)
            self._clear_corridor(traci_mod, "544295537", 1, pos + 115.0, pos + 240.0, ego_id)
            self._spawn(traci_mod, "act08_truck", "route_08", "truck_slow", "544295537", 2, pos + 96.0, 10.2, 7.6, ego_id, keep_lane=True)

        if "r08_merge" not in self._fired and edge == "544295537" and pos >= 280.0:
            self._fired.add("r08_merge")
            self._spawn(traci_mod, "act08_q1", "route_07", "truck_idm", "544295537", 0, pos + 185.0, 11.5, 11.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act08_q2", "route_07", "car_idm", "544295537", 0, pos + 145.0, 11.5, 11.5, ego_id, keep_lane=True)
            self._spawn(traci_mod, "act08_mid", "route_08", "delivery_idm", "544295537", 1, pos + 150.0, 15.5, 15.5, ego_id, keep_lane=False)

        if "r08_zipper" not in self._fired and edge == "544295537" and pos >= 650.0:
            self._fired.add("r08_zipper")
            self._spawn(traci_mod, "act08_z1", "route_02", "car_eidm", "169728849", 0, 240.0, 15.0, 15.0, ego_id, keep_lane=False)
            self._spawn(traci_mod, "act08_z2", "route_02", "delivery_idm", "169728849", 0, 185.0, 15.0, 15.0, ego_id, keep_lane=False)
