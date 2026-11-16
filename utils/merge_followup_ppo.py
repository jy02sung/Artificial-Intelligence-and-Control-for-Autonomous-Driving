"""Follow-up 1: same acceleration-only C update, all actor weights trainable."""
import torch
from utils.merge_adapter_ppo import MergeAdapterPPO

class MergeFollowupPPO(MergeAdapterPPO):
    def configure_adapter(self):
        for p in self.policy.parameters():p.requires_grad_(True)
        with torch.no_grad():self.policy.trunk[0].weight[:,60:63]=0
        self.actor_parameters=[p for n,p in self.policy.named_parameters() if not n.startswith('v_head.')]
        self.optimizer=torch.optim.Adam([{'params':self.actor_parameters,'lr':3e-5},{'params':list(self.policy.v_head.parameters()),'lr':3e-4}])
