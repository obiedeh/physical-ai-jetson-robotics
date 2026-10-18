# Failure modes of the simulation trial campaigns

Simulation lane; never physical evidence. Every trial of every report is sorted
into one mode. A lift counts from the report's own lift height; a nudge is a
rise above 2 mm; a reach is the gripper centre within 3 cm of the cup's
centre, known only for campaigns that recorded the gripper pose; a hold is a
trial the safety client refused for more than half its steps; the last column
is the share of all steps the client held, whatever the mode; the success rate
carries its Wilson 95% interval, which is about forty points wide at twenty
trials and twenty-five at fifty. A campaign a
study reran under a changed rule is marked superseded and left out of the totals.

| policy | step | trials | success rate [95% interval] | success | steps held |
|---|---|---|---|---|---|
| staged_expert_v1 | None | 50 | 100% [93, 100] | 50 | 0.0% |
| staged_expert_v1 | None | 50 | 100% [93, 100] | 50 | 0.0% |
| all current | | 100 | | 100 | |

Placement, for the trials that lifted the cup: final distance of the cup
from the mark's centre against the task's radius.

| policy | step | lifted | within radius | median (cm) | p90 (cm) |
|---|---|---|---|---|---|
| staged_expert_v1 | None | 50 | 50 of 50 at 3 cm | 0.4 | 0.7 |
