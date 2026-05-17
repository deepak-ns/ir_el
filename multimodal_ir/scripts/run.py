#!/usr/bin/env python3
# scripts/run.py
"""
Step 5: Launch the FastAPI backend and Streamlit UI together.

Usage:
    python scripts/run.py
    python scripts/run.py --api-only
    python scripts/run.py --ui-only
"""

import sys
import subprocess
import time
import click
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import cfg


@click.command()
@click.option("--api-only", is_flag=True, help="Launch only the FastAPI backend")
@click.option("--ui-only",  is_flag=True, help="Launch only the Streamlit UI")
@click.option("--reload",   is_flag=True, default=False, help="Enable hot reload for API")
def main(api_only, ui_only, reload):
    procs = []

    if not ui_only:
        api_cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", cfg.api.host,
            "--port", str(cfg.api.port),
        ]
        if reload:
            api_cmd.append("--reload")
        print(f"[run] Starting API on http://{cfg.api.host}:{cfg.api.port}")
        procs.append(subprocess.Popen(api_cmd, cwd=str(Path(__file__).parent.parent)))
        time.sleep(3)  # Give API time to bind

    if not api_only:
        ui_cmd = [
            sys.executable, "-m", "streamlit", "run",
            "ui/app.py",
            "--server.port", str(cfg.ui.port),
            "--server.headless", "true",
        ]
        print(f"[run] Starting UI on http://localhost:{cfg.ui.port}")
        procs.append(subprocess.Popen(ui_cmd, cwd=str(Path(__file__).parent.parent)))

    print("\n[run] Both services started. Press Ctrl+C to stop.\n")

    def shutdown(sig, frame):
        print("\n[run] Shutting down...")
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()
