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

| policy | step | trials | success rate [95% interval] | success | lifted, then dropped | moved the cup, no lift | reached the cup, no grasp | never reached the cup | held by the safety client most of the trial | steps held |
|---|---|---|---|---|---|---|---|---|---|---|
| simcup17oct3_n045 | 20000 | 50 | 70% [56, 81] | 35 | 0 | 9 | 0 | 6 | 0 | 0.0% |
| simplace01_n131 | 20000 | 50 | 0% [0, 7] | 0 | 2 | 14 | 13 | 16 | 5 | 8.5% |
| simplace01_n131 | 20000 | 50 | 0% [0, 7] | 0 | 2 | 18 | 12 | 13 | 5 | 8.5% |
| all current | | 150 | | 35 | 4 | 41 | 25 | 35 | 10 | |

Placement, for the trials that lifted the cup: final distance of the cup
from the mark's centre against the task's radius.

| policy | step | lifted | within radius | median (cm) | p90 (cm) |
|---|---|---|---|---|---|
| simplace01_n131 | 20000 | 2 | 0 of 2 at 3 cm | 7.6 | 7.6 |
| simplace01_n131 | 20000 | 2 | 0 of 2 at 3 cm | 11.8 | 11.8 |
