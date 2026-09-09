# Reports

This folder stores hardware bring-up evidence, screenshots, benchmark notes, and generated inventory reports.

Commit small Markdown reports and small images. Avoid committing large logs, bag files, datasets, model weights, simulation caches, or training outputs.

Suggested structure:

```text
reports/
|-- demo/
|-- inventory/
|-- isaac/
|-- jetson/
|-- robotics/
|-- simulation/
|-- synria/
|-- training/
`-- yahboom/
```

Use the per-track bring-up templates (`reports/yahboom/BRINGUP_TEMPLATE.md`, `reports/synria/BRINGUP_TEMPLATE.md`, `reports/simulation/BRINGUP_TEMPLATE.md`) when capturing evidence.
