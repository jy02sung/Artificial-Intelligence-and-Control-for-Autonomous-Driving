"""Build measured-demand SUMO scenarios for the pmh zipper experiments.

Demand comes from the team's observed-bounds inputs
(docs/experiments/ic-observed-bounds-inputs-20261113.csv): mainline routes
are 2025-10-16 evening-peak counts (stations 10010 / 10031) split with an
assumed exit share; local roads keep the documented 4,800 veh/h assumption.
The HV vehicle types and departure settings come unchanged from
scenarios/hv_high15000_entryprep.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scenarios/hv_high15000_entryprep'
INPUTS = ROOT / 'docs/experiments/ic-observed-bounds-inputs-20261113.csv'
DEFAULT_CASE = 'low_exit25'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_counts(case=DEFAULT_CASE):
    rows = [r for r in csv.DictReader(INPUTS.open(encoding='utf-8')) if r['case'] == case]
    assert len(rows) == 16, (case, len(rows))
    return {r['route']: int(r['vehicles_in_3600s']) for r in rows}, \
        {r['route']: r['provenance'] for r in rows}


def build(out, network, case=DEFAULT_CASE, scale=1.0, routes_from=BASE / 'highway.rou.xml'):
    """Write network.net.xml, highway.rou.xml, run.sumocfg and manifest.json into out."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(network, out / 'network.net.xml')
    shutil.copy2(BASE / 'run.sumocfg', out / 'run.sumocfg')
    counts, provenance = measured_counts(case)
    tree = ET.parse(routes_from)
    for flow in tree.findall('flow'):
        number = int(round(counts[flow.get('route')] * scale))
        if number == 0:          # scale=0 gives an empty (no-traffic) scenario
            tree.getroot().remove(flow)
        else:
            flow.set('number', str(number))
    assert not tree.findall('vehicle')
    tree.write(out / 'highway.rou.xml', encoding='utf8', xml_declaration=True)
    manifest = dict(case=case, scale=scale, total_vph=round(sum(counts.values()) * scale),
                    route_counts={k: int(round(v * scale)) for k, v in counts.items()},
                    provenance=provenance, source_inputs=str(INPUTS.relative_to(ROOT)),
                    source_inputs_sha=sha(INPUTS), base_scenario=str(BASE.relative_to(ROOT)),
                    network_source=str(Path(network).resolve().relative_to(ROOT)),
                    files={n: sha(out / n) for n in ('network.net.xml', 'highway.rou.xml', 'run.sumocfg')},
                    limitations=['Mainline = observed evening-peak envelope; exit share is assumed.',
                                 'Local roads 4,800 veh/h is an unmeasured assumption.',
                                 'Bus 5% / violators 2% / controller mix are assumptions; trucks not modelled.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest
