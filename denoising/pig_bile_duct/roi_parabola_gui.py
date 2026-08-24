import argparse
import tkinter as tk
from pathlib import Path

from roi_gui.app import RoiParabolaGui

DEFAULT_GUI_INPUT = Path(__file__).resolve().parents[2] / "raw" / "pig_bile_duct.png"


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path

    fixed = Path(str(path).replace("\\_", "_").replace("/_", "_"))
    if fixed.exists():
        return fixed

    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        for candidate in (project_root / path, project_root / fixed):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"{path}\n"
        "If this path was copied from Markdown, remove backslashes before underscores. "
        "For example use raw\\pig_bile_duct.png, not raw\\pig\\_bile\\_duct.png."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ROI GUI for wavelet/parabola correction layers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_GUI_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "roi_gui_outputs")
    args = parser.parse_args()
    args.input = resolve_input_path(args.input)

    root = tk.Tk()
    RoiParabolaGui(root, args.input, args.output_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
