# Matched four-controller collection

User requested four separate datasets under identical conditions, preserving all old data, with quantitative comparisons for later BC+PPO.

- Keep v2 geometry/rule supervisor, 0.2s steps, 90s ego departure, 300s horizon, two routes and 0.6/1/1.4 demand scales.
- Collect 5 seeds per route/load (30 episodes per controller, 120 total). Paired seeds shared across Krauss/IDM/EIDM/ACC.
- Change ego car-following model only; common accel/decel/emergencyDecel/tau/minGap/length/width/speedFactor/LC2013 fixed. Model-specific defaults retained and SUMO version recorded.
- Materialize background flow vehicles/types/departures and fixed speedFactor=1 to eliminate random type/speed-factor draw differences. Interacting trajectories can still diverge, including stochastic car-following streams.
- Compare against newly collected matched IDM, never the historical different experiment.
- Separate per-condition repeats 0/1/2 train, 3 validation, 4 test. Preserve failed episodes and label eligible BC demonstrations. No model training.
- Metrics: success/collision/timeout pp differences; all-episode 300s failure-penalized time; common-success travel times; stopped time; speed; acceleration and jerk RMS; hard-braking events/minute; TTC exposure; rule events. Baseline zero => relative change undefined, report absolute difference. Paired stratified bootstrap for key metrics; exploratory results only.
- Test deterministic demand and single-attribute changes, split leakage, denominator-zero and failure handling, numeric metric fixtures. Validate all collected NPZ/JSONL; write reproducible summaries and plots to the repository, full raw data in ignored timestamped results.

Known limitations retained: road markings only partially verified; no sensor occlusion; observed kinematic actions are not controller commands; 50m observations can hide state used by the teacher. BC+PPO results require new training and held-out evaluation, not this teacher comparison.
