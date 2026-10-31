"""Road-rule experiments. Native SUMO gap/collision checks are never disabled."""
import hashlib
import json
import traci
from build import HERE

def is_violator(ident,seed,fraction,bus=False):
    draw=int.from_bytes(hashlib.sha256(f'{seed}:{ident}'.encode()).digest()[:8],'big')/2**64
    return not bus and ident!='ego' and draw<fraction

def emergency_needed(speed,leader_speed,gap,decel,reaction,ttc_limit):
    closing=speed-leader_speed
    required=speed*reaction+max(0,(speed**2-leader_speed**2)/(2*decel))+2.5
    return closing>0 and gap/closing<ttc_limit and gap<required

def safe_gap(gap,speed,headway,minimum):
    return gap>=max(minimum,speed*headway)

class RulePolicy:
    def __init__(self,seed,event_path):
        self.config=json.loads((HERE/'marking_rules.json').read_text(encoding='utf-8'))
        self.inventory=json.loads((HERE/'rules_v2/inventory.json').read_text(encoding='utf-8'))
        self.seed=seed;self.known=set();self.violators=set();self.last_emergency=-999.
        self.events=event_path.open('w',encoding='utf-8')
        self.counts=dict(assigned=0,violators=0,solid_crossings=0,violator_crossings=0,
                         ordinary_solid_crossings=0,ego_emergency_requests=0,ego_emergency_crossings=0)

    def close(self):self.events.close()

    def solid(self,edge,a,b):
        return min(a,b) in self.inventory.get(edge,{}).get('solid_boundaries',[]) and abs(a-b)==1

    def context(self,states):
        if 'ego' not in states:return [0.]*4
        v=states['ego'];n=self.inventory.get(v['edge'],{}).get('lanes',0)
        return [float(v['lane']+1<n),float(v['lane']>0),
                float(self.solid(v['edge'],v['lane'],v['lane']+1)),
                float(self.solid(v['edge'],v['lane'],v['lane']-1))]

    def before_step(self,states):
        time=traci.simulation.getTime()
        for ident in states:
            if ident in self.known:continue
            self.known.add(ident);self.counts['assigned']+=ident!='ego'
            if is_violator(ident,self.seed,self.config['violator_fraction'],traci.vehicle.getVehicleClass(ident)=='bus'):
                self.violators.add(ident);self.counts['violators']+=1
                traci.vehicle.setVehicleClass(ident,'custom1')
                traci.vehicle.setColor(ident,(255,135,0,255))
        decision={'emergency':False,'target_lane':None,'brake':False,'reason':'normal'}
        if 'ego' not in states:return decision
        # An exception only lasts for this one simulation step.
        traci.vehicle.setVehicleClass('ego','passenger')
        traci.vehicle.setLaneChangeMode('ego',1621)
        traci.vehicle.setParameter('ego','lcReason','normal')
        ego=states['ego'];p=self.config['emergency']
        if ego['edge'].startswith(':'):return decision
        leader=traci.vehicle.getLeader('ego',100)
        if not leader or not leader[0]:return decision
        gap=leader[1]+traci.vehicle.getMinGap('ego')
        lv=traci.vehicle.getSpeed(leader[0]);decel=traci.vehicle.getEmergencyDecel('ego')
        decision.update(leader=leader[0],physical_gap_m=gap,leader_speed=lv)
        if not emergency_needed(ego['speed'],lv,gap,decel,p['reaction_s'],p['ttc_s']):return decision
        # Request braking as well as considering a lateral escape. Default speedMode stays on.
        traci.vehicle.setAcceleration('ego',-decel,.2)
        decision.update(brake=True,reason='braking_deficit_no_safe_escape')
        if time-self.last_emergency<p['cooldown_s']:return decision
        n=self.inventory.get(ego['edge'],{}).get('lanes',0)
        for direction in (1,-1):
            target=ego['lane']+direction
            if not 0<=target<n:continue
            best={int(row[0].rsplit('_',1)[1]):row for row in traci.vehicle.getBestLanes('ego')}
            if target not in best or best[target][1]<max(20,ego['speed']*2):continue
            right=int(direction<0)
            front=traci.vehicle.getNeighbors('ego',2+right)
            rear=traci.vehicle.getNeighbors('ego',right)
            if any(not safe_gap(g+traci.vehicle.getMinGap('ego'),ego['speed'],p['front_headway_s'],p['min_gap_m']) for _,g in front):continue
            if any(not safe_gap(g+traci.vehicle.getMinGap(v),traci.vehicle.getSpeed(v),p['rear_headway_s'],p['min_gap_m']) for v,g in rear):continue
            traci.vehicle.setVehicleClass('ego','custom2')
            traci.vehicle.setLaneChangeMode('ego',512)  # safety gaps retained; only our explicit lateral request
            traci.vehicle.changeLane('ego',target,.2)
            traci.vehicle.setParameter('ego','lcReason','emergency_escape')
            decision.update(emergency=True,target_lane=target,reason='safe_escape_requested')
            self.last_emergency=time;self.counts['ego_emergency_requests']+=1
            break
        return decision

    def after_step(self,before,after,decision):
        time=traci.simulation.getTime()
        for ident,v in before.items():
            nxt=after.get(ident)
            if not nxt or nxt['edge']!=v['edge'] or nxt['lane']==v['lane']:continue
            solid=self.solid(v['edge'],v['lane'],nxt['lane'])
            emergency=ident=='ego' and decision['emergency']
            event=dict(time_s=time,vehicle=ident,edge=v['edge'],from_lane=v['lane'],to_lane=nxt['lane'],
                       solid=solid,violator=ident in self.violators,emergency=emergency)
            if solid:
                self.counts['solid_crossings']+=1
                if ident in self.violators:self.counts['violator_crossings']+=1
                elif emergency:self.counts['ego_emergency_crossings']+=1
                else:self.counts['ordinary_solid_crossings']+=1
            self.events.write(json.dumps(event)+'\n')
        if decision['brake']:
            self.events.write(json.dumps(dict(time_s=time,ego_decision=decision))+'\n')
