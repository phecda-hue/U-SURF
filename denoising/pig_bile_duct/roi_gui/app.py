import json
from dataclasses import asdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .backend import CorrectionBackend, direction_sign, is_horizontal_direction, luma_to_rgb, parse_floats
from .models import CorrectionLayer, Roi


class RoiParabolaGui:
    ruler = 42

    def __init__(self, root: tk.Tk, input_path: Path, output_dir: Path):
        self.root = root
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layers_dir = self.output_dir / "layers"
        self.layers_dir.mkdir(exist_ok=True)

        self.backend = CorrectionBackend(input_path)
        self.width = self.backend.width
        self.height = self.backend.height
        self.zoom = 0.55
        self.roi = Roi(1500, 0, min(1816, self.width), self.height).normalized(self.width, self.height)
        self.drag_start: tuple[int, int] | None = None
        self.preview: dict | None = None
        self.layers: list[CorrectionLayer] = []
        self.next_layer_id = 1
        self.photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.load_existing_project()
        self._sync_roi_entries()
        self.refresh_layer_list()
        self.refresh_canvas()

    def _build_ui(self) -> None:
        self.root.title("Pig bile duct ROI parabola corrector")
        self.root.geometry("1500x920")

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        self.main_pane = main

        image_frame = ttk.Frame(main)
        main.add(image_frame, weight=4)
        control_shell = ttk.Frame(main, width=260)
        control_shell.grid_propagate(False)
        control_shell.pack_propagate(False)
        main.add(control_shell, weight=0)

        self.canvas = tk.Canvas(image_frame, background="#222222", highlightthickness=0)
        xscroll = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        yscroll = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        control_canvas = tk.Canvas(control_shell, width=250, highlightthickness=0)
        control_scroll = ttk.Scrollbar(control_shell, orient=tk.VERTICAL, command=control_canvas.yview)
        control_canvas.configure(yscrollcommand=control_scroll.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame = ttk.Frame(control_canvas, padding=6)
        control_window = control_canvas.create_window((0, 0), window=control_frame, anchor="nw")

        def update_scroll_region(_event: tk.Event) -> None:
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def fit_control_width(event: tk.Event) -> None:
            control_canvas.itemconfigure(control_window, width=event.width)

        control_frame.bind("<Configure>", update_scroll_region)
        control_canvas.bind("<Configure>", fit_control_width)
        control_canvas.bind("<MouseWheel>", lambda event: control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        self._build_controls(control_frame)
        self.root.after(100, self._set_initial_pane_width)

    def _set_initial_pane_width(self) -> None:
        try:
            self.main_pane.sashpos(0, max(900, self.root.winfo_width() - 270))
        except tk.TclError:
            pass

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="ROI").pack(anchor="w")
        roi_grid = ttk.Frame(parent)
        roi_grid.pack(fill=tk.X, pady=(2, 8))
        self.roi_vars: dict[str, tk.StringVar] = {}
        for idx, name in enumerate(("min_x", "max_x", "min_y", "max_y")):
            ttk.Label(roi_grid, text=name).grid(row=idx // 2, column=(idx % 2) * 2, sticky="w", padx=(0, 4))
            var = tk.StringVar()
            self.roi_vars[name] = var
            ttk.Entry(roi_grid, textvariable=var, width=8).grid(row=idx // 2, column=(idx % 2) * 2 + 1, sticky="ew")
        ttk.Button(parent, text="Apply ROI values", command=self.apply_roi_entries).pack(fill=tk.X, pady=(0, 8))

        ttk.Separator(parent).pack(fill=tk.X, pady=6)
        ttk.Label(parent, text="Fit Parameters").pack(anchor="w")
        self.var_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Auto fit current ROI", variable=self.var_auto).pack(anchor="w")
        self.param_vars: dict[str, tk.StringVar] = {}
        defaults = {
            "family": "F1",
            "direction": "up",
            "center_x": "1659.6",
            "curvature": "0.016",
            "period": "30.542",
            "harmonics": "4",
            "alpha": "1.5",
            "y_shifts": "0",
        }
        for key, value in defaults.items():
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=key, width=10).pack(side=tk.LEFT)
            var = tk.StringVar(value=value)
            self.param_vars[key] = var
            if key == "direction":
                combo = ttk.Combobox(row, textvariable=var, values=("up", "down", "left", "right"), state="readonly", width=8)
                combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
            else:
                ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, fill=tk.X, expand=True)

        advanced = ttk.LabelFrame(parent, text="Advanced")
        advanced.pack(fill=tk.X, pady=8)
        adv_defaults = {
            "y_shift_weights": "1",
            "curv_min": "0.004",
            "curv_max": "0.04",
            "period_min": "8",
            "period_max": "42",
            "min_score": "0.0002",
            "clip_sigma": "2.5",
            "comp_mean": "1",
            "comp_sigma": "18",
            "comp_strength": "0.45",
            "center_steps": "7",
            "curv_steps": "9",
            "fine_steps": "7",
        }
        self.adv_vars: dict[str, tk.StringVar] = {}
        for key, value in adv_defaults.items():
            row = ttk.Frame(advanced)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=key, width=12).pack(side=tk.LEFT)
            var = tk.StringVar(value=value)
            self.adv_vars[key] = var
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT, fill=tk.X, expand=True)

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, pady=4)
        ttk.Button(buttons, text="Preview", command=self.preview_current).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(buttons, text="Save ROI", command=self.accept_preview).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(buttons, text="Discard", command=self.discard_preview).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Separator(parent).pack(fill=tk.X, pady=8)
        ttk.Label(parent, text="Saved ROI Layers").pack(anchor="w")
        self.layer_list = tk.Listbox(parent, height=8, exportselection=False)
        self.layer_list.pack(fill=tk.BOTH, expand=True, pady=(2, 4))
        layer_buttons = ttk.Frame(parent)
        layer_buttons.pack(fill=tk.X)
        ttk.Button(layer_buttons, text="Toggle", command=self.toggle_layer).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(layer_buttons, text="Delete", command=self.delete_layer).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(layer_buttons, text="Refresh", command=self.refresh_canvas).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        export_buttons = ttk.Frame(parent)
        export_buttons.pack(fill=tk.X, pady=8)
        ttk.Button(export_buttons, text="Save Project JSON", command=self.save_project).pack(fill=tk.X, pady=2)
        ttk.Button(export_buttons, text="Export Final Image", command=self.export_final).pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, wraplength=225).pack(anchor="w", pady=(8, 0))

    def canvas_to_image(self, cx: float, cy: float) -> tuple[int, int]:
        x = int(round((self.canvas.canvasx(cx) - self.ruler) / self.zoom))
        y = int(round((self.canvas.canvasy(cy) - self.ruler) / self.zoom))
        return int(np.clip(x, 0, self.width)), int(np.clip(y, 0, self.height))

    def on_press(self, event: tk.Event) -> None:
        self.drag_start = self.canvas_to_image(event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        self.roi = Roi(self.drag_start[0], self.drag_start[1], x, y).normalized(self.width, self.height)
        self._sync_roi_entries()
        self.refresh_canvas(redraw_image=False)

    def on_release(self, event: tk.Event) -> None:
        self.on_drag(event)
        self.drag_start = None

    def on_wheel(self, event: tk.Event) -> None:
        old = self.zoom
        self.zoom *= 1.1 if event.delta > 0 else 0.9
        self.zoom = float(np.clip(self.zoom, 0.2, 2.0))
        if abs(self.zoom - old) > 1e-6:
            self.refresh_canvas()

    def _sync_roi_entries(self) -> None:
        self.roi_vars["min_x"].set(str(self.roi.x0))
        self.roi_vars["max_x"].set(str(self.roi.x1))
        self.roi_vars["min_y"].set(str(self.roi.y0))
        self.roi_vars["max_y"].set(str(self.roi.y1))

    def apply_roi_entries(self) -> None:
        try:
            self.roi = Roi(
                int(self.roi_vars["min_x"].get()),
                int(self.roi_vars["min_y"].get()),
                int(self.roi_vars["max_x"].get()),
                int(self.roi_vars["max_y"].get()),
            ).normalized(self.width, self.height)
        except ValueError as exc:
            messagebox.showerror("ROI error", str(exc))
            return
        self._sync_roi_entries()
        self.refresh_canvas()

    def refresh_canvas(self, redraw_image: bool = True) -> None:
        if redraw_image:
            display_rgb = luma_to_rgb(self.backend.rgb, self.backend.combined_luma(self.layers, self.preview))
            image = Image.fromarray(display_rgb)
            scaled = image.resize((int(self.width * self.zoom), int(self.height * self.zoom)), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(scaled)

        self.canvas.delete("all")
        if self.photo is not None:
            self.canvas.create_image(self.ruler, self.ruler, anchor="nw", image=self.photo)
        self.draw_rulers()
        self.draw_roi()
        self.draw_preview_curves()
        self.draw_saved_rois()
        self.canvas.configure(scrollregion=(0, 0, self.ruler + self.width * self.zoom + 90, self.ruler + self.height * self.zoom + 90))

    def draw_rulers(self) -> None:
        w = self.width * self.zoom
        h = self.height * self.zoom
        self.canvas.create_rectangle(0, 0, self.ruler + w, self.ruler, fill="#303030", outline="")
        self.canvas.create_rectangle(0, 0, self.ruler, self.ruler + h, fill="#303030", outline="")
        step = 100 if self.zoom > 0.35 else 200
        for x in range(0, self.width + 1, step):
            cx = self.ruler + x * self.zoom
            self.canvas.create_line(cx, self.ruler - 8, cx, self.ruler, fill="#c9c9c9")
            self.canvas.create_text(cx + 2, self.ruler - 22, text=str(x), fill="#d6d6d6", anchor="w", font=("Segoe UI", 8))
        for y in range(0, self.height + 1, step):
            cy = self.ruler + y * self.zoom
            self.canvas.create_line(self.ruler - 8, cy, self.ruler, cy, fill="#c9c9c9")
            self.canvas.create_text(4, cy, text=str(y), fill="#d6d6d6", anchor="w", font=("Segoe UI", 8))

    def draw_roi(self) -> None:
        x0 = self.ruler + self.roi.x0 * self.zoom
        x1 = self.ruler + self.roi.x1 * self.zoom
        y0 = self.ruler + self.roi.y0 * self.zoom
        y1 = self.ruler + self.roi.y1 * self.zoom
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00e5ff", width=2)
        bottom = self.ruler + self.height * self.zoom + 18
        right = self.ruler + self.width * self.zoom + 18
        self.canvas.create_line(self.ruler, bottom, x0, bottom, fill="#e8e8e8", dash=(5, 4))
        self.canvas.create_line(x1, bottom, self.ruler + self.width * self.zoom, bottom, fill="#e8e8e8", dash=(5, 4))
        self.canvas.create_line(x0, bottom - 10, x0, bottom + 10, fill="#00e5ff")
        self.canvas.create_line(x1, bottom - 10, x1, bottom + 10, fill="#00e5ff")
        self.canvas.create_line(right, self.ruler, right, y0, fill="#e8e8e8", dash=(5, 4))
        self.canvas.create_line(right, y1, right, self.ruler + self.height * self.zoom, fill="#e8e8e8", dash=(5, 4))
        self.canvas.create_line(right - 10, y0, right + 10, y0, fill="#00e5ff")
        self.canvas.create_line(right - 10, y1, right + 10, y1, fill="#00e5ff")
        self.canvas.create_text((x0 + x1) / 2, bottom + 16, text=f"{self.roi.x0} : {self.roi.x1}", fill="#ffffff")
        self.canvas.create_text(right + 14, (y0 + y1) / 2, text=f"{self.roi.y0} : {self.roi.y1}", fill="#ffffff", anchor="w")

    def draw_preview_curves(self) -> None:
        if self.preview is not None:
            self.draw_component_on_canvas(self.preview["roi"], self.preview["component"], self.preview["direction"], "#00ff80")

    def draw_saved_rois(self) -> None:
        for layer in self.layers:
            if not layer.enabled:
                continue
            x0 = self.ruler + layer.roi.x0 * self.zoom
            x1 = self.ruler + layer.roi.x1 * self.zoom
            y0 = self.ruler + layer.roi.y0 * self.zoom
            y1 = self.ruler + layer.roi.y1 * self.zoom
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ffd54f", width=1)

    def draw_component_on_canvas(self, roi: Roi, comp, direction: str, color: str) -> None:
        if is_horizontal_direction(direction):
            ys = np.arange(roi.y0, roi.y1, dtype=np.float32)
            local_ys = ys - roi.y0
            for y_shift in comp.y_shifts:
                base = roi.x0 + comp.sign * comp.curvature * (local_ys - comp.center_x) ** 2 + y_shift
                offsets = np.arange(-2 * comp.period, roi.x1 - roi.x0 + 2 * comp.period, max(comp.period, 1.0), dtype=np.float32)
                for off in offsets:
                    pts = []
                    xs = base + off
                    for x, y in zip(xs, ys):
                        if roi.x0 <= x < roi.x1:
                            pts.append((self.ruler + x * self.zoom, self.ruler + y * self.zoom))
                    if len(pts) >= 2:
                        self.canvas.create_line(*[coord for pt in pts for coord in pt], fill=color, width=1)
        else:
            xs = np.arange(roi.x0, roi.x1, dtype=np.float32)
            local_xs = xs - roi.x0
            for y_shift in comp.y_shifts:
                base = roi.y0 + comp.sign * comp.curvature * (local_xs - comp.center_x) ** 2 + y_shift
                offsets = np.arange(-2 * comp.period, roi.y1 - roi.y0 + 2 * comp.period, max(comp.period, 1.0), dtype=np.float32)
                for off in offsets:
                    pts = []
                    ys = base + off
                    for x, y in zip(xs, ys):
                        if roi.y0 <= y < roi.y1:
                            pts.append((self.ruler + x * self.zoom, self.ruler + y * self.zoom))
                    if len(pts) >= 2:
                        self.canvas.create_line(*[coord for pt in pts for coord in pt], fill=color, width=1)

    def current_params(self) -> dict:
        y_shifts = parse_floats(self.param_vars["y_shifts"].get(), [0.0])
        y_weights = parse_floats(self.adv_vars["y_shift_weights"].get(), [1.0] * len(y_shifts))
        if len(y_weights) == 1 and len(y_shifts) > 1:
            y_weights = y_weights * len(y_shifts)
        if len(y_weights) != len(y_shifts):
            raise ValueError("y_shift_weights count must match y_shifts count")
        return {
            "family": self.param_vars["family"].get().strip() or "F1",
            "direction": self.param_vars["direction"].get().strip() or "up",
            "auto": bool(self.var_auto.get()),
            "center_x": float(self.param_vars["center_x"].get()),
            "curvature": float(self.param_vars["curvature"].get()),
            "period": float(self.param_vars["period"].get()),
            "sign": direction_sign(self.param_vars["direction"].get()),
            "harmonics": int(self.param_vars["harmonics"].get()),
            "alpha": float(self.param_vars["alpha"].get()),
            "y_shifts": y_shifts,
            "y_shift_weights": y_weights,
            "curvature_range": (float(self.adv_vars["curv_min"].get()), float(self.adv_vars["curv_max"].get())),
            "period_range": (float(self.adv_vars["period_min"].get()), float(self.adv_vars["period_max"].get())),
            "min_score": float(self.adv_vars["min_score"].get()),
            "clip_sigma": float(self.adv_vars["clip_sigma"].get()),
            "compensate_local_mean": self.adv_vars["comp_mean"].get().strip() not in ("0", "false", "False", "no"),
            "compensation_sigma": float(self.adv_vars["comp_sigma"].get()),
            "compensation_strength": float(self.adv_vars["comp_strength"].get()),
            "center_steps": int(self.adv_vars["center_steps"].get()),
            "curv_steps": int(self.adv_vars["curv_steps"].get()),
            "fine_steps": int(self.adv_vars["fine_steps"].get()),
        }

    def preview_current(self) -> None:
        try:
            params = self.current_params()
            preview = self.backend.compute_preview(self.roi, params)
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))
            return
        self.preview = preview
        comp = preview["component"]
        center_origin = self.roi.y0 if is_horizontal_direction(preview["direction"]) else self.roi.x0
        self.param_vars["center_x"].set(f"{center_origin + comp.center_x:.3f}")
        self.param_vars["curvature"].set(f"{comp.curvature:.7f}")
        self.param_vars["period"].set(f"{comp.period:.3f}")
        self.status_var.set(
            f"Preview: family={preview['family']} direction={preview['direction']} "
            f"roi=({self.roi.x0},{self.roi.y0})-({self.roi.x1},{self.roi.y1}) "
            f"score={comp.score:.6g} period={comp.period:.3f}"
        )
        self.refresh_canvas()

    def accept_preview(self) -> None:
        if self.preview is None:
            messagebox.showinfo("No preview", "Preview first, then save the ROI.")
            return
        layer_id = self.next_layer_id
        self.next_layer_id += 1
        stem = f"correction_{layer_id:03d}"
        correction_path = self.layers_dir / f"{stem}_correction.npy"
        noise_path = self.layers_dir / f"{stem}_noise.npy"
        weight_path = self.layers_dir / f"{stem}_weight.npy"
        np.save(correction_path, self.preview["correction"])
        np.save(noise_path, self.preview["noise"])
        np.save(weight_path, self.preview["weight"])

        roi = self.preview["roi"]
        comp = self.preview["component"]
        horizontal = is_horizontal_direction(self.preview["direction"])
        center_origin = roi.y0 if horizontal else roi.x0
        center_key = "center_y_global" if horizontal else "center_x_global"
        local_center_key = "center_y_local" if horizontal else "center_x_local"
        layer = CorrectionLayer(
            layer_id=layer_id,
            family=self.preview["family"],
            enabled=True,
            roi=roi,
            component={
                "direction": self.preview["direction"],
                center_key: center_origin + comp.center_x,
                local_center_key: comp.center_x,
                "curvature": comp.curvature,
                "sign": comp.sign,
                "period": comp.period,
                "score": comp.score,
                "y_shifts": list(comp.y_shifts),
                "y_shift_weights": list(comp.y_shift_weights),
            },
            params=self.preview["params"],
            score=comp.score,
            period=comp.period,
            correction_path=str(correction_path),
            noise_path=str(noise_path),
            weight_path=str(weight_path),
        )
        self.layers.append(layer)
        self.backend.save_layer_outputs(layer, self.layers_dir)
        with open(self.layers_dir / f"correction_{layer.layer_id:03d}_params.json", "w", encoding="utf-8") as f:
            json.dump(asdict(layer), f, ensure_ascii=False, indent=2)
        self.preview = None
        self.refresh_layer_list()
        self.refresh_canvas()
        self.status_var.set(f"Saved ROI layer {layer_id:03d}.")

    def discard_preview(self) -> None:
        self.preview = None
        self.status_var.set("Preview discarded. Unsaved ROI will stay original in final export.")
        self.refresh_canvas()

    def refresh_layer_list(self) -> None:
        self.layer_list.delete(0, tk.END)
        for layer in self.layers:
            mark = "on" if layer.enabled else "off"
            self.layer_list.insert(
                tk.END,
                f"{layer.layer_id:03d} [{mark}] {layer.family} x={layer.roi.x0}:{layer.roi.x1} y={layer.roi.y0}:{layer.roi.y1} "
                f"a={layer.component['curvature']:.5f} T={layer.period:.2f}",
            )

    def selected_layer_index(self) -> int | None:
        sel = self.layer_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def toggle_layer(self) -> None:
        idx = self.selected_layer_index()
        if idx is None:
            return
        self.layers[idx].enabled = not self.layers[idx].enabled
        self.refresh_layer_list()
        self.refresh_canvas()

    def delete_layer(self) -> None:
        idx = self.selected_layer_index()
        if idx is None:
            return
        del self.layers[idx]
        self.refresh_layer_list()
        self.refresh_canvas()

    def project_payload(self) -> dict:
        return {
            "input": str(self.input_path),
            "image_shape": [self.height, self.width],
            "layers": [asdict(layer) for layer in self.layers],
        }

    def _resolve_layer_array_path(self, layer_id: int, saved_path: str, suffix: str) -> str:
        path = Path(saved_path) if saved_path else None
        if path is not None and path.is_file():
            return str(path)

        fallback = self.layers_dir / f"correction_{layer_id:03d}_{suffix}.npy"
        if fallback.exists():
            return str(fallback)

        return saved_path

    def load_existing_project(self) -> None:
        project_path = self.output_dir / "roi_corrections_project.json"
        if not project_path.exists():
            return

        try:
            with open(project_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            shape = payload.get("image_shape")
            if shape and list(shape) != [self.height, self.width]:
                self.status_var.set("Existing project ignored: image size does not match.")
                return

            loaded_layers: list[CorrectionLayer] = []
            for item in payload.get("layers", []):
                layer_id = int(item["layer_id"])
                roi_data = item["roi"]
                roi = Roi(
                    int(roi_data["x0"]),
                    int(roi_data["y0"]),
                    int(roi_data["x1"]),
                    int(roi_data["y1"]),
                ).normalized(self.width, self.height)
                loaded_layers.append(
                    CorrectionLayer(
                        layer_id=layer_id,
                        family=item.get("family", "F1"),
                        enabled=bool(item.get("enabled", True)),
                        roi=roi,
                        component=item.get("component", {}),
                        params=item.get("params", {}),
                        score=float(item.get("score", 0.0)),
                        period=float(item.get("period", item.get("component", {}).get("period", 0.0))),
                        correction_path=self._resolve_layer_array_path(
                            layer_id, item.get("correction_path", ""), "correction"
                        ),
                        noise_path=self._resolve_layer_array_path(layer_id, item.get("noise_path", ""), "noise"),
                        weight_path=self._resolve_layer_array_path(layer_id, item.get("weight_path", ""), "weight"),
                    )
                )

            self.layers = loaded_layers
            if loaded_layers:
                self.next_layer_id = max(layer.layer_id for layer in loaded_layers) + 1
                self.status_var.set(f"Loaded {len(loaded_layers)} saved ROI layers.")
        except Exception as exc:
            self.layers = []
            self.status_var.set(f"Could not load existing project: {exc}")

    def save_project(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=str(self.output_dir),
            initialfile="roi_corrections_project.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.project_payload(), f, ensure_ascii=False, indent=2)
        self.status_var.set(f"Saved project: {path}")

    def export_final(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=str(self.output_dir),
            initialfile="pig_bile_duct_roi_corrected_final.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.backend.export_final(self.layers, output_path)
            with open(self.output_dir / "roi_corrections_project.json", "w", encoding="utf-8") as f:
                json.dump(self.project_payload(), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self.status_var.set(f"Export failed: {exc}")
            return
        self.status_var.set(f"Exported final image: {path}")
