from pathlib import Path

import pytest

from jetson_edge_ai_security.sources import CsvReplaySource


def test_csv_replay_maps_edge_iiot_columns(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "frame.time,ip.src_host,ip.dst_host,tcp.srcport,tcp.dstport,_ws.col.Protocol,"
        "frame.len,tcp.flags,Attack_label,Attack_type\n"
        "2026-01-01 00:00:00,10.0.0.1,10.0.0.2,1234,443,TCP,128,S,1,DDoS\n",
        encoding="utf-8",
    )

    events = list(CsvReplaySource(path).events())

    assert len(events) == 1
    assert events[0].source_ip == "10.0.0.1"
    assert events[0].dest_port == 443
    assert events[0].attack_label is True
    assert events[0].attack_type == "DDoS"


def test_csv_replay_skips_malformed_rows_by_default(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "timestamp,source_ip,dest_ip,source_port,dest_port,protocol,packet_size\n"
        "2026-01-01 00:00:00,10.0.0.1,10.0.0.2,bad,443,TCP,128\n"
        "2026-01-01 00:00:01,10.0.0.3,10.0.0.4,1235,443,TCP,256\n",
        encoding="utf-8",
    )
    source = CsvReplaySource(path)

    events = list(source.events())

    assert len(events) == 1
    assert source.rows_skipped == 1


def test_csv_replay_strict_mode_raises_on_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        "timestamp,source_ip,dest_ip,source_port,dest_port,protocol,packet_size\n"
        "2026-01-01 00:00:00,10.0.0.1,10.0.0.2,bad,443,TCP,128\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        list(CsvReplaySource(path, strict=True).events())

