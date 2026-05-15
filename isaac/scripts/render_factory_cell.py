#!/usr/bin/env python3
"""Open the repo-local factory-cell USD scene in Isaac Sim and render a screenshot."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "isaac" / "usd" / "factory_cell_v0.usda"
DEFAULT_OUTPUT = REPO_ROOT / "isaac" / "outputs" / "factory_cell_v0.png"
DEFAULT_CAMERA = "/World/FactoryCell/CameraViewpoints/OverviewCamera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a screenshot from the repo-local OpenUSD factory-cell scene."
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="USD scene to open.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Screenshot path.")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="USD camera prim path to render.")
    parser.add_argument("--width", type=int, default=1280, help="Viewport width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Viewport height in pixels.")
    parser.add_argument("--frames", type=int, default=30, help="Frames to advance before capture.")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show the Isaac Sim UI instead of running headless.",
    )
    return parser.parse_args()


def _capture_viewport_sync(output: Path, simulation_app) -> None:
    """Capture the active viewport by driving the Kit update loop explicitly."""
    import omni.kit.app
    import omni.kit.renderer_capture
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    capture = capture_viewport_to_file(viewport, file_path=str(output))
    capture_interface = omni.kit.renderer_capture.acquire_renderer_capture_interface()

    # Drive Kit's update loop so that next_update_async() futures can resolve.
    # wait_for_result(completion_frames=N) awaits N next_update_async() calls;
    # those futures only resolve when app.update() is called from the main thread.
    for _ in range(60):
        simulation_app.update()
        if output.exists() and output.stat().st_size > 0:
            break

    # Flush any remaining async capture work.
    for _ in range(3):
        capture_interface.wait_async_capture()
        simulation_app.update()


def main() -> None:
    args = parse_args()
    scene_path = args.scene.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not scene_path.exists():
        raise FileNotFoundError(f"USD scene does not exist: {scene_path}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": not args.no_headless,
            "active_gpu": 0,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "width": args.width,
            "height": args.height,
            "extra_args": [
                "--/renderer/multiGpu/autoEnable=false",
            ],
        }
    )

    try:
        import omni.kit.app
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport

        context = omni.usd.get_context()
        context.open_stage(str(scene_path))

        app = omni.kit.app.get_app()
        for _ in range(max(args.frames, 1)):
            app.update()

        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open USD scene: {scene_path}")

        camera = stage.GetPrimAtPath(args.camera)
        if not camera.IsValid():
            raise RuntimeError(f"Camera prim not found: {args.camera}")

        viewport = get_active_viewport()
        viewport.camera_path = args.camera
        viewport.resolution = (args.width, args.height)

        for _ in range(5):
            app.update()

        _capture_viewport_sync(output_path, simulation_app)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Screenshot was not written: {output_path}")

        print(f"Opened scene: {scene_path}")
        print(f"Rendered screenshot: {output_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
