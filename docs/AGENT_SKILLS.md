# Agent Skills — sources & maintenance

This repo vendors NVIDIA agent skills into `.claude/skills/` (installed
2026-08-13: Jetson device/memory/LLM-serving, DALI, LeRobot viz,
Omniverse simready/USD tuning). Upstream catalogs update routinely —
check them before assuming a skill is current or missing.

## Where to search for skills

| Source | What it is |
|---|---|
| https://github.com/NVIDIA/skills | NVIDIA's official catalog (~335 skills), synced daily from product repos; signed with skill cards + eval datasets |
| https://skills.sh | The universal skills registry/directory behind the `skills` CLI — search across publishers |
| https://github.com/anthropics/skills | Anthropic's public skills collection |

## CLI cheatsheet (`npx skills`)

```bash
npx skills find <query>                 # interactive search across registries
npx skills find --owner nvidia          # limit to one publisher
npx skills add nvidia/skills -l         # list a repo's catalog without installing
npx skills add nvidia/skills --agent claude-code -y --copy \
    --skill <name> [--skill <name>...]  # install selected skills project-level
npx skills ls                           # what's installed here
npx skills update                       # refresh installed skills to latest
```

Convention for this repo: install with `--copy` (not symlinks) and commit
`.claude/skills/` so both machines (5090 + Thor) get identical skill
versions via git; run `npx skills update` deliberately, review the diff,
and commit the bump like any dependency.
