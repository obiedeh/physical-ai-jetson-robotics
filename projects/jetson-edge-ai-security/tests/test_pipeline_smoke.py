from pathlib import Path

from jetson_edge_ai_security.detection import BaselineDetector, BaselineThresholds
from jetson_edge_ai_security.runtime import PipelineRunner
from jetson_edge_ai_security.sources import CsvReplaySource


def test_pipeline_smoke_emits_alert_from_labeled_replay(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "timestamp,source_ip,dest_ip,source_port,dest_port,protocol,packet_size,attack_label\n"
        "2026-01-01 00:00:00,10.0.0.1,10.0.0.2,1000,443,TCP,100,0\n"
        "2026-01-01 00:00:01,10.0.0.2,10.0.0.2,1001,443,TCP,120,1\n"
        "2026-01-01 00:00:02,10.0.0.3,10.0.0.2,1002,443,TCP,140,0\n",
        encoding="utf-8",
    )
    source = CsvReplaySource(path)
    detector = BaselineDetector(BaselineThresholds(packet_count_threshold=10, attack_count_threshold=1))

    with source:
        runner = PipelineRunner(source, window_size=3, step=1, detector=detector)
        alerts = runner.run()

    assert len(alerts) == 1
    assert alerts[0].severity == "medium"
    assert runner.metrics.events_seen == 3
    assert runner.metrics.windows_seen == 1

