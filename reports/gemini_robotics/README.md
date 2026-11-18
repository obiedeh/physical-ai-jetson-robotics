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
