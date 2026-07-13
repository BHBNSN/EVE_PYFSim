from __future__ import annotations

import multiprocessing

from eve_sim.gui import run_gui

def main() -> None:
    run_gui()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
