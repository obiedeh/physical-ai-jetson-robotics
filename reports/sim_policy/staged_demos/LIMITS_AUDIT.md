# Dataset limits audit

Dataset `/home/oedeh/synria-sim-datasets/staged_place_01/dataset`, gripper 50mm, against the
operator limits verified by Obinna Edeh on 2026-10-17.
A frame presses a bound when a commanded or observed joint is within 0.5 deg of it;
it is outside when past the exact bound; it is beyond tolerance when past the
bound by more than the 1 deg recording tolerance. The policy client clamps every
command to the exact bound and holds on an observed state outside it.

Episodes: 150; frames: 41172; episodes pressing a bound: 150; frames pressing: 24567; frames outside: 215; frames beyond tolerance: 91.

| joint | bounds | episodes pressing | frames pressing | lower / upper | frames outside | beyond tolerance | commanded range | observed range |
|---|---|---|---|---|---|---|---|---|
| Joint1 | -155.0 to 144.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -59.3 to 46.6 deg | -59.2 to 46.5 deg |
| Joint2 | -87.0 to 100.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -33.7 to 87.9 deg | -34.0 to 90.9 deg |
| Joint3 | -35.0 to 135.0 deg | 16 | 314 | 314 / 0 | 215 | 91 | -36.4 to 23.1 deg | -37.1 to 22.2 deg |
| Joint4 | -112.0 to 88.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -59.0 to 84.0 deg | -58.6 to 83.4 deg |
| Joint5 | -72.0 to 68.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -43.1 to 65.0 deg | -45.2 to 65.0 deg |
| Joint6 | -171.0 to 176.0 deg | 0 | 0 | 0 / 0 | 0 | 0 | -83.0 to 56.7 deg | -82.2 to 55.8 deg |
| Gripper | 0.000 to 0.025 m | 150 | 24567 | 24567 / 0 | 0 | 0 | 0.0000 to 0.0220 m | 0.0000 to 0.0143 m |

| episode | frames | pressing | outside | beyond tolerance | joints pressed |
|---|---|---|---|---|---|
| 0 | 437 | 235 | 0 | 0 | Joint3 lower, Gripper lower |
| 1 | 277 | 159 | 0 | 0 | Gripper lower |
| 2 | 292 | 159 | 0 | 0 | Gripper lower |
| 3 | 446 | 235 | 0 | 0 | Joint3 lower, Gripper lower |
| 4 | 232 | 157 | 0 | 0 | Gripper lower |
| 5 | 278 | 157 | 0 | 0 | Gripper lower |
| 6 | 284 | 159 | 0 | 0 | Gripper lower |
| 7 | 252 | 150 | 0 | 0 | Gripper lower |
| 8 | 264 | 152 | 0 | 0 | Gripper lower |
| 9 | 243 | 154 | 0 | 0 | Gripper lower |
| 10 | 232 | 157 | 0 | 0 | Gripper lower |
| 11 | 241 | 155 | 0 | 0 | Gripper lower |
| 12 | 251 | 153 | 0 | 0 | Gripper lower |
| 13 | 431 | 230 | 9 | 0 | Joint3 lower, Gripper lower |
| 14 | 229 | 157 | 0 | 0 | Gripper lower |
| 15 | 239 | 149 | 0 | 0 | Gripper lower |
| 16 | 236 | 156 | 0 | 0 | Gripper lower |
| 17 | 226 | 155 | 0 | 0 | Gripper lower |
| 18 | 250 | 152 | 0 | 0 | Gripper lower |
| 19 | 274 | 158 | 0 | 0 | Gripper lower |
| 20 | 236 | 154 | 0 | 0 | Gripper lower |
| 21 | 255 | 150 | 0 | 0 | Gripper lower |
| 22 | 393 | 200 | 0 | 0 | Joint3 lower, Gripper lower |
| 23 | 232 | 153 | 0 | 0 | Gripper lower |
| 24 | 243 | 152 | 0 | 0 | Gripper lower |
| 25 | 260 | 151 | 0 | 0 | Gripper lower |
| 26 | 246 | 152 | 0 | 0 | Gripper lower |
| 27 | 435 | 234 | 27 | 19 | Joint3 lower, Gripper lower |
| 28 | 440 | 236 | 0 | 0 | Joint3 lower, Gripper lower |
| 29 | 262 | 155 | 0 | 0 | Gripper lower |
| 30 | 243 | 157 | 0 | 0 | Gripper lower |
| 31 | 243 | 153 | 0 | 0 | Gripper lower |
| 32 | 283 | 159 | 0 | 0 | Gripper lower |
| 33 | 271 | 155 | 0 | 0 | Gripper lower |
| 34 | 234 | 157 | 0 | 0 | Gripper lower |
| 35 | 239 | 157 | 0 | 0 | Gripper lower |
| 36 | 278 | 159 | 0 | 0 | Gripper lower |
| 37 | 230 | 156 | 0 | 0 | Gripper lower |
| 38 | 241 | 152 | 0 | 0 | Gripper lower |
| 39 | 239 | 154 | 0 | 0 | Gripper lower |
| 40 | 248 | 150 | 0 | 0 | Gripper lower |
| 41 | 257 | 149 | 0 | 0 | Gripper lower |
| 42 | 232 | 156 | 0 | 0 | Gripper lower |
| 43 | 242 | 154 | 0 | 0 | Gripper lower |
| 44 | 234 | 151 | 0 | 0 | Gripper lower |
| 45 | 250 | 152 | 0 | 0 | Gripper lower |
| 46 | 255 | 154 | 0 | 0 | Gripper lower |
| 47 | 235 | 153 | 0 | 0 | Gripper lower |
| 48 | 235 | 157 | 0 | 0 | Gripper lower |
| 49 | 238 | 157 | 0 | 0 | Gripper lower |
| 50 | 233 | 150 | 0 | 0 | Gripper lower |
| 51 | 260 | 155 | 0 | 0 | Gripper lower |
| 52 | 238 | 153 | 0 | 0 | Gripper lower |
| 53 | 236 | 151 | 0 | 0 | Gripper lower |
| 54 | 245 | 152 | 0 | 0 | Gripper lower |
| 55 | 253 | 154 | 0 | 0 | Gripper lower |
| 56 | 248 | 149 | 0 | 0 | Gripper lower |
| 57 | 278 | 158 | 0 | 0 | Gripper lower |
| 58 | 276 | 159 | 0 | 0 | Gripper lower |
| 59 | 263 | 150 | 0 | 0 | Gripper lower |
| 60 | 274 | 157 | 0 | 0 | Gripper lower |
| 61 | 226 | 152 | 0 | 0 | Gripper lower |
| 62 | 261 | 152 | 0 | 0 | Gripper lower |
| 63 | 233 | 151 | 0 | 0 | Gripper lower |
| 64 | 251 | 153 | 0 | 0 | Gripper lower |
| 65 | 233 | 151 | 0 | 0 | Gripper lower |
| 66 | 421 | 221 | 11 | 0 | Joint3 lower, Gripper lower |
| 67 | 262 | 151 | 0 | 0 | Gripper lower |
| 68 | 238 | 155 | 0 | 0 | Gripper lower |
| 69 | 228 | 156 | 0 | 0 | Gripper lower |
| 70 | 255 | 151 | 0 | 0 | Gripper lower |
| 71 | 247 | 151 | 0 | 0 | Gripper lower |
| 72 | 275 | 156 | 0 | 0 | Gripper lower |
| 73 | 241 | 151 | 0 | 0 | Gripper lower |
| 74 | 241 | 151 | 0 | 0 | Gripper lower |
| 75 | 262 | 155 | 0 | 0 | Gripper lower |
| 76 | 274 | 157 | 0 | 0 | Gripper lower |
| 77 | 232 | 156 | 0 | 0 | Gripper lower |
| 78 | 227 | 155 | 0 | 0 | Gripper lower |
| 79 | 251 | 151 | 0 | 0 | Gripper lower |
| 80 | 243 | 154 | 0 | 0 | Gripper lower |
| 81 | 251 | 150 | 0 | 0 | Gripper lower |
| 82 | 260 | 153 | 0 | 0 | Gripper lower |
| 83 | 248 | 151 | 0 | 0 | Gripper lower |
| 84 | 254 | 153 | 0 | 0 | Gripper lower |
| 85 | 441 | 242 | 16 | 0 | Joint3 lower, Gripper lower |
| 86 | 436 | 242 | 26 | 17 | Joint3 lower, Gripper lower |
| 87 | 265 | 153 | 0 | 0 | Gripper lower |
| 88 | 378 | 187 | 0 | 0 | Gripper lower |
| 89 | 242 | 150 | 0 | 0 | Gripper lower |
| 90 | 275 | 158 | 0 | 0 | Gripper lower |
| 91 | 242 | 152 | 0 | 0 | Gripper lower |
| 92 | 230 | 152 | 0 | 0 | Gripper lower |
| 93 | 263 | 156 | 0 | 0 | Gripper lower |
| 94 | 436 | 241 | 26 | 15 | Joint3 lower, Gripper lower |
| 95 | 246 | 153 | 0 | 0 | Gripper lower |
| 96 | 241 | 157 | 0 | 0 | Gripper lower |
| 97 | 441 | 228 | 0 | 0 | Gripper lower |
| 98 | 272 | 157 | 0 | 0 | Gripper lower |
| 99 | 385 | 193 | 0 | 0 | Gripper lower |
| 100 | 231 | 154 | 0 | 0 | Gripper lower |
| 101 | 268 | 154 | 0 | 0 | Gripper lower |
| 102 | 270 | 153 | 0 | 0 | Gripper lower |
| 103 | 259 | 153 | 0 | 0 | Gripper lower |
| 104 | 255 | 150 | 0 | 0 | Gripper lower |
| 105 | 232 | 158 | 0 | 0 | Gripper lower |
| 106 | 410 | 207 | 8 | 0 | Joint3 lower, Gripper lower |
| 107 | 243 | 155 | 0 | 0 | Gripper lower |
| 108 | 231 | 155 | 0 | 0 | Gripper lower |
| 109 | 236 | 152 | 0 | 0 | Gripper lower |
| 110 | 233 | 151 | 0 | 0 | Gripper lower |
| 111 | 277 | 153 | 0 | 0 | Gripper lower |
| 112 | 275 | 153 | 0 | 0 | Gripper lower |
| 113 | 239 | 159 | 0 | 0 | Gripper lower |
| 114 | 254 | 151 | 0 | 0 | Gripper lower |
| 115 | 233 | 158 | 0 | 0 | Gripper lower |
| 116 | 228 | 150 | 0 | 0 | Gripper lower |
| 117 | 238 | 149 | 0 | 0 | Gripper lower |
| 118 | 381 | 190 | 0 | 0 | Gripper lower |
| 119 | 237 | 158 | 0 | 0 | Gripper lower |
| 120 | 449 | 233 | 0 | 0 | Gripper lower |
| 121 | 258 | 157 | 0 | 0 | Gripper lower |
| 122 | 430 | 232 | 5 | 0 | Joint3 lower, Gripper lower |
| 123 | 234 | 153 | 0 | 0 | Gripper lower |
| 124 | 235 | 157 | 0 | 0 | Gripper lower |
| 125 | 420 | 221 | 20 | 10 | Joint3 lower, Gripper lower |
| 126 | 239 | 151 | 0 | 0 | Gripper lower |
| 127 | 231 | 154 | 0 | 0 | Gripper lower |
| 128 | 240 | 148 | 0 | 0 | Gripper lower |
| 129 | 279 | 155 | 0 | 0 | Gripper lower |
| 130 | 234 | 151 | 0 | 0 | Gripper lower |
| 131 | 265 | 157 | 0 | 0 | Gripper lower |
| 132 | 396 | 195 | 0 | 0 | Gripper lower |
| 133 | 431 | 232 | 16 | 0 | Joint3 lower, Gripper lower |
| 134 | 255 | 151 | 0 | 0 | Gripper lower |
| 135 | 227 | 158 | 0 | 0 | Gripper lower |
| 136 | 256 | 154 | 0 | 0 | Gripper lower |
| 137 | 269 | 157 | 0 | 0 | Gripper lower |
| 138 | 231 | 150 | 0 | 0 | Gripper lower |
| 139 | 424 | 229 | 22 | 11 | Joint3 lower, Gripper lower |
| 140 | 232 | 150 | 0 | 0 | Gripper lower |
| 141 | 265 | 152 | 0 | 0 | Gripper lower |
| 142 | 232 | 156 | 0 | 0 | Gripper lower |
| 143 | 432 | 228 | 29 | 19 | Joint3 lower, Gripper lower |
| 144 | 248 | 155 | 0 | 0 | Gripper lower |
| 145 | 257 | 148 | 0 | 0 | Gripper lower |
| 146 | 271 | 154 | 0 | 0 | Gripper lower |
| 147 | 264 | 150 | 0 | 0 | Gripper lower |
| 148 | 242 | 154 | 0 | 0 | Gripper lower |
| 149 | 242 | 148 | 0 | 0 | Gripper lower |
