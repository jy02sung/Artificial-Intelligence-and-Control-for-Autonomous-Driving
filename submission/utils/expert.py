"""Longitudinal control and lane choice for the final rolling-merge teacher."""
import numpy as np
from utils.driving_env import MAX_LANES


def required_direction(x):
    lane, count = np.rint(x[32:34] * 5).astype(int)
    links = x[40:].reshape(5, 2)[:count]
    eligible = links[:, int(x[34] > 0)] if x[35] >= 0 else links.max(axis=1)
    choices = np.flatnonzero(eligible)
    if not len(choices) or eligible[lane]:
        return 0
    return int(np.sign(choices[np.argmin(abs(choices-lane))] - lane))


def _base_action(obs, env):
    speed = float(obs[0]) * env.vmax
    limit = float(obs[31]) * env.vmax
    lane, count = np.rint(obs[32:34] * MAX_LANES).astype(int)
    sign, distance = obs[34:36]
    links = obs[40:].reshape(MAX_LANES, 2)[:count]
    eligible = links[:, int(sign > 0)] if distance >= 0 else links.max(axis=1)
    options = np.flatnonzero(eligible)
    desired = lane if not options.size else min(options, key=lambda i: abs(i-lane))
    direction = int(np.sign(desired - lane))
    target = max(1., limit)
    lead_gap = max(0., float(obs[9]) * env.visibility)
    closing = -float(obs[10]) * env.vmax
    command = 0
    if direction and not obs[38]:
        index = 1 + 4 * (2-direction)
        front_gap = float(obs[index]) * env.visibility
        front_closing = -float(obs[index+1]) * env.vmax
        rear_gap = -float(obs[index+2]) * env.visibility
        rear_closing = float(obs[index+3]) * env.vmax
        room = (front_gap > 3.+0.45*speed+max(0.,front_closing)
                and rear_gap > 3.+0.50*max(0.,speed+rear_closing)+max(0.,rear_closing)
                and obs[37 if direction==1 else 36] == 0.)
        if room:
            command = direction
            if front_gap < lead_gap:
                lead_gap, closing = front_gap, front_closing
        else:
            target = min(target, max(2., np.sqrt(2.8*float(obs[39])*400.)))
    spacing = 2.5 + max(0., 1.05*speed + speed*closing/(2*np.sqrt(2.2*3.2)))
    interaction = (spacing/max(.2,lead_gap))**2 if lead_gap < env.visibility else 0.
    accel = 2.2 * (1.-(speed/target)**4-interaction)
    return np.array([np.clip(accel/(env.max_accel if accel>=0 else env.max_decel),-1,1),command], np.float32)


def _approach_action(obs, env):
    action = _base_action(obs, env)
    lane, count = np.rint(obs[32:34]*5).astype(int)
    links = obs[40:].reshape(5,2)[:count]
    eligible = links[:,int(obs[34]>0)] if obs[35]>=0 else links.max(axis=1)
    if not eligible.any() or eligible[lane] or action[1] or obs[38]:
        return action
    # Reserve longitudinal space while waiting for a safe lateral gap.
    remaining = 400.*float(obs[39])
    speed = float(obs[0])*env.vmax
    target = np.sqrt(2.*2.5*max(0.,remaining-14.))
    braking = np.clip((target-speed)/1.3,-env.max_decel,env.max_accel)
    action[0] = min(action[0], braking/(env.max_decel if braking<0 else env.max_accel))
    return action


def _decision(obs, env, mode):
    if obs.shape != (50,):
        raise ValueError('A single 50-dimensional policy observation is required')
    original = _approach_action(obs, env)
    direction = required_direction(obs)
    if direction == 0 or obs[38] or original[1]:
        return original, False, direction, 'unchanged'
    if mode == 'old':
        return original, True, direction, 'old_wait'

    i = 1 + 4*(2-direction)
    speed = float(obs[0])*env.vmax
    limit = max(1., float(obs[31])*env.vmax)
    remaining = 400.*float(obs[39])
    front = float(obs[i])*env.visibility
    front_closing = -float(obs[i+1])*env.vmax
    rear = -float(obs[i+2])*env.visibility
    rear_closing = float(obs[i+3])*env.vmax
    rear_speed = max(0., speed+rear_closing)
    overlap = bool(obs[37 if direction==1 else 36])
    front_req = 3.+.45*speed+max(0.,front_closing)
    rear_req = 3.+.50*rear_speed+max(0.,rear_closing)
    assert overlap or front<=front_req or rear<=rear_req
    lead = max(0.,float(obs[9])*env.visibility)
    closing = -float(obs[10])*env.vmax

    # Only the conditional second trial changes the lateral decision.
    relaxed_room = (mode=='relaxed' and remaining<25. and speed<4. and not overlap
                    and front>front_req and rear>3.+.45*rear_speed+max(0.,rear_closing))
    if relaxed_room and front<lead:
        lead, closing = front, front_closing
    desired_gap = 2.5+max(0.,1.05*speed+speed*closing/(2*np.sqrt(2.2*3.2)))
    interaction = (desired_gap/max(.2,lead))**2 if lead<env.visibility else 0.
    idm = 2.2*(1.-(speed/limit)**4-interaction)
    if relaxed_room:
        accel, cmd, reason = idm, direction, 'low_speed_gap'
    else:
        # Start from the original blocked-merge target, before its 14 m reserve cap.
        target = min(limit,max(2.,np.sqrt(2.8*remaining)))
        revised = mode == 'revised'
        margin = 13. if revised else 10.
        stop_cap = np.sqrt(2.*2.5*max(0.,remaining-margin))
        front_blocked = overlap or front <= front_req
        if front_blocked and (not revised or remaining > 25.):
            front_speed = max(0.,speed-front_closing) if front<env.visibility else speed
            drop = min(3., .15*limit) if revised else 3.5
            floor = max(5., .70*limit) if revised and limit <= 14. else 5.
            target = min(target,max(floor,front_speed-drop))
            reason = 'yield_alongside'
        elif not front_blocked and lead>15. and remaining>30. and (not revised or stop_cap > speed+3.):
            target = min(limit,max(target,rear_speed+3.))
            reason = 'advance_ahead'
        else:
            reason = 'approach_end'
        # Gentle speed alignment; following and the stop envelope can brake harder.
        alignment = max(-2.5,(target-speed)/1.3)
        accel = min(idm,alignment,(stop_cap-speed)/1.3)
        cmd = 0
    scale = env.max_accel if accel>=0 else env.max_decel
    return np.array([np.clip(accel/scale,-1.,1.),cmd],np.float32), not relaxed_room, direction, reason


def control(x,env,mode,*,revised=False,short_reserve=False):
    action,waiting,d,reason=_decision(x,env,'revised' if revised else 'strict')
    if mode=='strict' or not waiting:
        return action,waiting,d,reason
    i=1+4*(2-d)
    v=float(x[0])*env.vmax
    front=float(x[i])*env.visibility
    rear=-float(x[i+2])*env.visibility
    remain=float(x[39])*400.
    # A slow vehicle close to the intended lane can stop while the ego is still approaching.
    vf=max(0.,v+float(x[i+1])*env.vmax)
    vr=max(0.,v+float(x[i+3])*env.vmax)
    if remain>70. or not ((front<20. and vf<13.) or (rear<18. and vr<13.)):
        return action,waiting,d,reason
    reserve={'reserve20':20.,'reserve30':30.,'queue_tail':24.}[mode]
    if short_reserve and mode == 'queue_tail' and float(x[31])*env.vmax <= 14.:
        reserve = 12.
    target=np.sqrt(2*2.5*max(0.,remain-reserve))
    if mode=='queue_tail':
        target=min(target,max(0.,vf-4.+.35*(front-10.)))
    accel=min(float(action[0])* (env.max_decel if action[0]<0 else env.max_accel),(target-v)/1.3)
    a=np.array([np.clip(accel/(env.max_decel if accel<0 else env.max_accel),-1.,1.),0.],np.float32)
    return a,waiting,d,'reserve'
