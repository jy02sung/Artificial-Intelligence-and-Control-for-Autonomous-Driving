from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import torch


class Demand:
    """Scale only the sixteen flow counts in the simulator's private route file."""
    def __init__(self, env):
        cfg = Path(env.sim.cfg_path)
        route_file = ET.parse(cfg).find("input/route-files").attrib["value"]
        self.path = cfg.parent / route_file
        self.original = self.path.read_bytes()
        self.high = ET.fromstring(self.original)
        flows = self.high.findall("flow")
        assert len(flows)==16 and sum(int(f.get("number")) for f in flows)==15000
        assert all(float(f.get("end"))-float(f.get("begin"))==3600 for f in flows)

    def set(self, vph):
        if vph == 15000:
            self.path.write_bytes(self.original)
        else:
            assert vph == 3000
            scaled = deepcopy(self.high)
            for flow in scaled.findall("flow"):
                n = int(flow.attrib["number"])
                assert n % 5 == 0
                flow.set("number", str(n//5))
            ET.ElementTree(scaled).write(self.path, encoding="utf-8", xml_declaration=True)
        flows = ET.parse(self.path).findall("flow")
        assert sum(int(f.get("number")) for f in flows)==vph


def is_value(name):
    return name.startswith(("v_head.", "value_trunk."))


def separate(agent):
    policy = agent.policy
    policy.value_trunk = deepcopy(policy.trunk)
    agent.optimizer.add_param_group({"params":policy.value_trunk.parameters()})
    # A value-only backward pass must produce no gradients on any actor parameter.
    policy(torch.ones(1,50))[-1].sum().backward()
    for name, param in policy.named_parameters():
        assert (param.grad is not None) == is_value(name), name
    agent.optimizer.zero_grad(set_to_none=True)
