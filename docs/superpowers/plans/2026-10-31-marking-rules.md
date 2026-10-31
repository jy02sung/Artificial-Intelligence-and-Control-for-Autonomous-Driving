# Road markings and exception rules implementation plan

User approved scope: image-based markings, minority violating traffic, emergency-only ego exceptions, preserve baseline and collect a new dataset. Execute in existing user-selected jys checkout.

- [ ] Survey imagery, retain provenance and explicitly identify approximate/unverified boundaries.
- [x] Generate a separate rule network and routes; preserve the original network and all datasets.
- [x] Implement ordinary restrictions, deterministic 5% violating cars, and brake-first ego exception with target-lane safety checks.
- [x] Test rules and real SUMO behavior, including a controlled emergency case separate from natural traffic.
- [x] Collect 12 new episodes (two routes, three loads, two seeds), validate recorded actions and rule events, report actual outcomes.

Survey remains partial: 11 multi-lane edges unresolved and paint endpoints approximate; see RULES_V2.md. Completed collection is explicitly an exploratory partially verified map, not a fully surveyed road reproduction.

Architecture: SUMO lane change permissions allow custom1 (violators) and custom2 (temporary emergency permission) across marked solid lines. Road access permissions remain separate. Native car following and lane-change collision checks stay enabled. A split of the weaving edge represents its final solid segment. Single-lane ramps have edge lines but no adjacent changeable lane. This is a simulation research policy, not a certified safety controller.

Defaults chosen by implementer: 5% of non-bus background cars, deterministic hash assignment; TTC < 1.5s plus stopping-distance deficit; safe target front and rear headways; 5s exception cooldown. Finite 300s horizon and traffic densities unchanged. Evidence is visual interpretation of Naver aerial imagery viewed 2026-10-31; capture date not available. Do not call approximate paint endpoints surveyed coordinates, or treat obscured sections as verified.

Review focus: no ordinary solid-line changes; no ego solid crossing without emergency decision; safe rear gap; independent bridge traffic excluded from gap checks; no synthetic emergency samples mixed into natural dataset; route continuity after edge splitting.
