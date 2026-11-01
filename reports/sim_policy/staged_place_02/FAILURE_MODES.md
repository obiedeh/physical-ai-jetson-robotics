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

| policy | step | trials | success rate [95% interval] | success | lifted, then dropped | reached the mark, not held there | moved the cup, no lift | reached the cup, no grasp | never reached the cup | steps held |
|---|---|---|---|---|---|---|---|---|---|---|
| simplace02_n122 | 20000 | 50 | 32% [21, 46] | 16 | 11 | 11 | 4 | 4 | 4 | 0.1% |
| all current | | 50 | | 16 | 11 | 11 | 4 | 4 | 4 | |

Placement, for the trials that lifted the cup: final distance of the cup
from the mark's centre against the task's radius.

| policy | step | lifted | within radius | median (cm) | p90 (cm) |
|---|---|---|---|---|---|
| simplace02_n122 | 20000 | 38 | 18 of 38 at 3 cm | 4.3 | 13.0 |
