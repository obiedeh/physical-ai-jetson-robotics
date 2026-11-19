# Gemini pointing benchmark

Cloud Gemini models asked to point at the red cup in the 60 overhead frames of `reports/perception_bench` (224x224, truth in `truth.jsonl`). Scored by `physical_ai_lab/gemini_robotics/point_bench.py`. Perception only: nothing here commands a robot.

Run on 2026-11-18 from `rtx-5090-workstation`, same prompt and temperature 0.0 for both models.

| Model | Answered | Median error | p90 error | Max error | Within 20 mm | Median seconds per answer |
|---|---|---|---|---|---|---|
| `gemini-4-hydrogen` | 51 of 60 | 2.4 mm | 4.0 mm | 4.6 mm | 51 | 20.8 |
| `gemini-robotics-er-2-preview` | 59 of 60 | 2.3 mm | 3.8 mm | 187.6 mm | 58 | 2.96 |

What the numbers do and do not show:

- The 9 unanswered Hydrogen frames are service failures after 4 tries each (8 HTTP 503, 1 timeout), not wrong answers. The service was overloaded during the run, so the Hydrogen latency is a figure for that day only.
- ER 2 returned an empty list on one frame (54) and pointed 187.6 mm off on one frame (26).
- The two models were scored on different answered sets (51 and 59 frames), so the 0.1 mm difference in median is not a ranking.
- One run per model, one object, one camera view, simulation frames. No board-state reading, no move planning and no arm motion was tested.
- The earlier G1 figure for ER 2 (median 2.62 mm, `reports/perception_bench/g1_report.json`) used a different prompt.

Reproduce:

```bash
python -m physical_ai_lab.gemini_robotics.point_bench --model gemini-4-hydrogen \
  --bench reports/perception_bench --out reports/gemini_robotics/g1_point_gemini-4-hydrogen.json
```

The key is read from `GEMINI_API_KEY` or `~/.config/gemini/api_key`.

## Ludo board reading

Can a cloud Gemini model say where every token sits on a Ludo board from one overhead frame? The model points at each token and names its color; each point is mapped to a square of the 15x15 grid in `ludo_engine/board.py` with the known camera geometry and compared with the engine's truth. Scored by `physical_ai_lab/gemini_robotics/board_bench.py`.

Frames: 30 board states in `reports/ludo_board_bench`, rendered in Isaac Sim 5.1 on `rtx-5090-workstation` by `isaac/scripts/render_ludo_board_bench.py`. Each state is a random legal game position (seed 2026, 41 to 289 turns played) with all 16 tokens. 471 occupied color-and-square pairs in total. Rendered at 672x672; the 224 rows downsample the same frames to the size the arm's overhead camera records today.

Run on 2026-11-18, same prompt, temperature 0.0, one run per row.

| Model | Input width | Boards read exactly | Squares found | Wrong or extra | Median seconds per board |
|---|---|---|---|---|---|
| `gemini-4-hydrogen` | 672 | 30 of 30 | 471 of 471 | 0 | 8.19 |
| `gemini-4-hydrogen` | 224 | 30 of 30 | 471 of 471 | 0 | 8.5 |
| `gemini-robotics-er-2-preview` | 672 | 29 of 30 | 470 of 471 | 0 | 7.38 |
| `gemini-robotics-er-2-preview` | 224 | 27 of 30 | 469 of 471 | 2 | 7.02 |

The ER 2 errors: a green token under a red one missed at both sizes (board 8); at 224 also one token placed one square off (board 9) and one blue token called yellow (board 16).

What the numbers do and do not show:

- Rendered frames, not a camera. The board is drawn clean, the lighting is even, the camera is straight above the centre and its geometry is known exactly. No physical board has been read.
- No arm is in the frame. During play the arm covers part of the board.
- Tokens that share a square are stacked. The score checks which colors occupy a square, not how many tokens are in the stack; 10 of the 480 tokens sit above another token.
- The camera geometry was checked against the frames: predicted token centres match the rendered ones to a median of 0.9 px (max 1.3 px) on a 35 px square.
- One run per row and 30 boards: the difference between 30 and 29 boards is one answer.
- This is board reading only. Move choice, pick and place points on a real board, and arm motion were not tested.

Reproduce:

```bash
~/.venv/isaacsim5/bin/python isaac/scripts/render_ludo_board_bench.py --out reports/ludo_board_bench --n 30 --seed 2026
python -m physical_ai_lab.gemini_robotics.board_bench --model gemini-4-hydrogen \
  --out reports/gemini_robotics/ludo_board_gemini-4-hydrogen_672.json
python -m physical_ai_lab.gemini_robotics.board_bench --model gemini-4-hydrogen --size 224 \
  --out reports/gemini_robotics/ludo_board_gemini-4-hydrogen_224.json
```
