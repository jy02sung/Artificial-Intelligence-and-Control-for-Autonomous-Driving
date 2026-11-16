"""Reward-only cost for accelerating into observed junction conflict risk."""
from env.safety_strength_trials import SafetyStrengthEnv

def acceleration_risk_penalty(risk, acceleration, terminal):
    return 0. if terminal else -.4 * risk * min(max(float(acceleration),0.),1.)

class RiskActionEnv(SafetyStrengthEnv):
    def step(self,action):
        risk=self.conflict_features()[0]
        obs,reward,done,cut,info=super().step(action)
        penalty=acceleration_risk_penalty(risk,action[0],done)
        self.reward_counts['risk_action_total']+=penalty
        self.reward_counts['risk_action_events']+=int(penalty<0)
        self.reward_counts['bonus_total']+=penalty
        info['route_selection_reward']+=penalty
        info['route_reward_parts']['risk_action']=penalty
        return obs,reward+penalty,done,cut,info
