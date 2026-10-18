# Dataset limits audit

Dataset `/home/oedeh/synria-sim-datasets/pick_cup_17oct3_frozen_2026-10-18/dataset`, gripper 50mm, against the
operator limits verified by Obinna Edeh on 2026-10-17.
A frame presses a bound when a commanded or observed joint is within 0.5 deg of it;
it is outside when past the exact bound; it is beyond tolerance when past the
bound by more than the 1 deg recording tolerance. The policy client clamps every
command to the exact bound and holds on an observed state outside it.

Episodes: 50; frames: 21746; episodes pressing a bound: 41; frames pressing: 5233; frames outside: 1010; frames beyond tolerance: 0.

| joint | bounds | episodes pressing | frames pressing | lower / upper | frames outside | beyond tolerance | commanded range | observed range |
|---|---|---|---|---|---|---|---|---|
| Joint1 | -155.0 to 144.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -32.4 to 28.5 deg | -32.4 to 28.5 deg |
| Joint2 | -87.0 to 100.0 deg | 17 | 1141 | 0 / 1141 | 912 | 0 | -44.3 to 100.9 deg | -44.4 to 100.6 deg |
| Joint3 | -35.0 to 135.0 deg | 21 | 952 | 952 / 0 | 98 | 0 | -34.6 to 102.8 deg | -35.5 to 100.9 deg |
| Joint4 | -112.0 to 88.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -7.0 to 14.1 deg | -7.4 to 14.2 deg |
| Joint5 | -72.0 to 68.0 deg | 7 | 72 | 72 / 0 | 0 | 0 | -71.9 to 67.1 deg | -71.9 to 67.2 deg |
| Joint6 | -171.0 to 176.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -12.6 to 11.7 deg | -12.5 to 11.7 deg |
| Gripper | 0.000 to 0.025 m | 26 | 3224 | 3067 / 157 | 0 | 0 | 0.0000 to 0.0250 m | 0.0000 to 0.0250 m |

| episode | frames | pressing | outside | beyond tolerance | joints pressed |
|---|---|---|---|---|---|
| 0 | 708 | 29 | 8 | 0 | Joint3 lower, Gripper lower |
| 3 | 470 | 33 | 0 | 0 | Joint3 lower, Gripper lower |
| 4 | 487 | 1 | 0 | 0 | Gripper lower |
| 7 | 382 | 196 | 0 | 0 | Gripper lower |
| 9 | 463 | 71 | 71 | 0 | Joint2 upper |
| 10 | 426 | 128 | 125 | 0 | Joint2 upper, Gripper lower |
| 11 | 528 | 90 | 70 | 0 | Joint2 upper, Joint3 lower |
| 12 | 452 | 43 | 12 | 0 | Joint3 lower |
| 13 | 405 | 39 | 17 | 0 | Joint3 lower |
| 14 | 322 | 135 | 0 | 0 | Gripper lower |
| 15 | 459 | 117 | 31 | 0 | Joint3 lower, Joint5 lower |
| 16 | 451 | 8 | 0 | 0 | Joint3 lower |
| 17 | 433 | 25 | 0 | 0 | Joint3 lower, Joint5 lower, Gripper lower |
| 18 | 566 | 25 | 15 | 0 | Joint3 lower |
| 19 | 585 | 145 | 69 | 0 | Joint2 upper, Joint3 lower, Joint5 lower, Gripper lower |
| 20 | 322 | 55 | 44 | 0 | Joint2 upper, Joint5 lower |
| 21 | 297 | 158 | 0 | 0 | Joint5 lower, Gripper lower |
| 22 | 358 | 120 | 0 | 0 | Joint2 upper, Joint5 lower, Gripper upper |
| 23 | 392 | 67 | 0 | 0 | Joint2 upper, Joint3 lower, Gripper upper |
| 25 | 402 | 1 | 0 | 0 | Joint5 lower |
| 26 | 461 | 307 | 0 | 0 | Gripper lower |
| 27 | 434 | 313 | 0 | 0 | Joint3 lower, Gripper lower |
| 28 | 390 | 162 | 0 | 0 | Joint3 lower |
| 29 | 338 | 218 | 0 | 0 | Joint3 lower, Gripper lower |
| 30 | 319 | 257 | 56 | 0 | Joint2 upper, Joint3 lower, Gripper lower |
| 31 | 372 | 43 | 43 | 0 | Joint2 upper |
| 32 | 511 | 89 | 88 | 0 | Joint2 upper, Gripper lower |
| 33 | 406 | 313 | 99 | 0 | Joint2 upper, Gripper lower |
| 34 | 512 | 79 | 75 | 0 | Joint2 upper |
| 35 | 320 | 242 | 60 | 0 | Joint2 upper, Gripper lower |
| 36 | 334 | 254 | 3 | 0 | Joint2 upper, Gripper lower |
| 37 | 527 | 48 | 23 | 0 | Joint2 upper, Gripper lower |
| 40 | 466 | 68 | 0 | 0 | Joint3 lower, Gripper upper |
| 41 | 479 | 63 | 58 | 0 | Joint2 upper, Joint3 lower |
| 42 | 334 | 137 | 43 | 0 | Joint2 upper, Joint3 lower |
| 43 | 366 | 233 | 0 | 0 | Joint3 lower, Gripper lower |
| 44 | 396 | 256 | 0 | 0 | Gripper lower |
| 45 | 587 | 75 | 0 | 0 | Joint3 lower |
| 46 | 331 | 214 | 0 | 0 | Joint3 lower, Gripper lower |
| 47 | 458 | 171 | 0 | 0 | Gripper lower |
| 48 | 982 | 205 | 0 | 0 | Gripper lower |
