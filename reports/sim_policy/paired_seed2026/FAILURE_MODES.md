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
| simcup17oct3appr_n045 | 20000 | 50 | 0% [0, 7] | 0 | 0 | 3 | 0 | 40 | 7 | 10.2% |
| simcup17oct3appr_n045 | 15000 | 50 | 0% [0, 7] | 0 | 0 | 2 | 1 | 44 | 3 | 5.7% |
| simcup17oct3appr_n045 | 20000 | 50 | 0% [0, 7] | 0 | 0 | 3 | 0 | 43 | 4 | 7.8% |
| simcup17oct3long_n045 | 60000 | 50 | 50% [37, 63] | 25 | 0 | 13 | 0 | 12 | 0 | 0.1% |
| simcup17oct3long_n045 | 20000 | 50 | 42% [29, 56] | 21 | 0 | 18 | 5 | 6 | 0 | 1.1% |
| simcup17oct3long_n045 | 40000 | 50 | 58% [44, 71] | 29 | 0 | 5 | 1 | 15 | 0 | 0.0% |
| simcup17oct3long_n045 | 60000 | 50 | 52% [39, 65] | 26 | 0 | 14 | 1 | 9 | 0 | 0.0% |
| simcup17oct3_n035 | 20000 | 50 | 0% [0, 7] | 0 | 0 | 2 | 0 | 48 | 0 | 1.8% |
| simcup17oct3_n040 | 20000 | 50 | 8% [3, 19] | 4 | 2 | 12 | 3 | 24 | 5 | 11.0% |
| simcup17oct3_n045 | 20000 | 50 | 18% [10, 31] | 9 | 3 | 10 | 6 | 22 | 0 | 3.3% |
| simcup17oct3_n045 | 5000 | 50 | 0% [0, 7] | 0 | 0 | 4 | 0 | 46 | 0 | 0.0% |
| simcup17oct3_n045 | 10000 | 50 | 2% [0, 10] | 1 | 0 | 16 | 2 | 31 | 0 | 0.3% |
| simcup17oct3_n045 | 15000 | 50 | 12% [6, 24] | 6 | 0 | 14 | 0 | 30 | 0 | 0.0% |
| simcup17oct3_n045 | 20000 | 50 | 56% [42, 69] | 28 | 0 | 7 | 1 | 14 | 0 | 0.5% |
| simcup17oct3s23_n035 | 20000 | 50 | 8% [3, 19] | 4 | 0 | 18 | 1 | 27 | 0 | 0.1% |
| simcup17oct3s23_n040 | 20000 | 50 | 10% [4, 21] | 5 | 0 | 17 | 10 | 18 | 0 | 7.3% |
| simcup17oct3s23_n045 | 20000 | 50 | 50% [37, 63] | 25 | 1 | 11 | 7 | 6 | 0 | 0.0% |
| simcup17oct3v2_n045 | 20000 | 50 | 0% [0, 7] | 0 | 0 | 0 | 0 | 50 | 0 | 2.4% |
| simcup17oct3v2_n045 | 20000 | 50 | 0% [0, 7] | 0 | 0 | 0 | 0 | 50 | 0 | 16.0% |
| simcup17oct3w3_n045 | 20000 | 50 | 24% [14, 37] | 12 | 0 | 3 | 5 | 30 | 0 | 0.9% |
| simcup17oct3w3_n045 | 20000 | 50 | 22% [13, 35] | 11 | 0 | 7 | 2 | 30 | 0 | 0.5% |
| all current | | 1050 | | 206 | 6 | 179 | 45 | 595 | 19 | |
