from .common import INPUT_PATH, Path, argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Skeletonize a vessel probability map and estimate branch length, "
            "diameter, tortuosity, and curvature."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--multi-output-input",
        type=Path,
        default=None,
        help=(
            "Optional .npz containing multi-output vessel predictions. Expected "
            "keys are mask, centerline, and distance unless overridden by "
            "--mask-key, --centerline-key, and --distance-key."
        ),
    )
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--centerline-key", default="centerline")
    parser.add_argument("--distance-key", default="distance")
    parser.add_argument(
        "--centerline-input",
        type=Path,
        default=None,
        help="Optional centerline probability map used for skeleton extraction.",
    )
    parser.add_argument(
        "--distance-input",
        type=Path,
        default=None,
        help=(
            "Optional vessel distance map. Values are multiplied by "
            "--distance-to-diameter-scale to make the diameter map."
        ),
    )
    parser.add_argument(
        "--centerline-threshold",
        type=float,
        default=None,
        help=(
            "Threshold for --centerline-input or multi-output centerline map. "
            "When omitted, --threshold is used; if that is also omitted, Otsu is used."
        ),
    )
    parser.add_argument(
        "--distance-to-diameter-scale",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied to the distance output before diameter analysis. "
            "Use 2.0 when D(x,y) is in pixels; use 2*max_distance_px for normalized D."
        ),
    )

    # Skeleton mask preprocessing
    parser.add_argument(
        "--sigma-y",
        type=float,
        default=0.0,
        help="Gaussian sigma along rows (y direction) for skeleton extraction.",
    )
    parser.add_argument(
        "--sigma-x",
        type=float,
        default=0.0,
        help="Gaussian sigma along columns (x direction) for skeleton extraction.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Skeleton-mask threshold in [0, 1]. Omit to use Otsu.",
    )
    parser.add_argument(
        "--measurement-threshold",
        type=float,
        default=None,
        help=(
            "Threshold in [0, 1] for the unblurred diameter-measurement mask. "
            "Omit to use Otsu on the original probability map."
        ),
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=24,
        help="Remove foreground components smaller than this area in pixels.",
    )
    parser.add_argument(
        "--min-hole-size",
        type=int,
        default=0,
        help="Fill background holes smaller than this area in pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--mask-closing-radius",
        type=int,
        default=0,
        help=(
            "Morphologically close binary masks with this pixel radius before "
            "small-hole filling. Useful for tiny artifact gaps/pinholes; use 0 "
            "to disable. Keep small to avoid merging nearby vessels."
        ),
    )
    parser.add_argument(
        "--skeleton-min-hole-size",
        type=int,
        default=None,
        help=(
            "Hole area filled only in the skeleton mask. When omitted, "
            "--min-hole-size is used. Set 0 to preserve split-merge loops."
        ),
    )
    parser.add_argument(
        "--skeleton-loop-hole-size",
        type=int,
        default=0,
        help=(
            "After skeletonization, fill background regions smaller than this "
            "area when they are enclosed by the one-pixel skeleton, then "
            "skeletonize again. This suppresses tiny artifact loops/cycles. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--measurement-min-hole-size",
        type=int,
        default=None,
        help=(
            "Hole area filled only in the diameter-measurement mask. When omitted, "
            "--min-hole-size is used. Set 0 to keep the gap between parallel branches."
        ),
    )
    parser.add_argument(
        "--auto-density-roi",
        action="store_true",
        help=(
            "Automatically remove long low-vessel-density bands before geometry "
            "analysis and density normalization."
        ),
    )
    parser.add_argument(
        "--density-roi-visual-only",
        action="store_true",
        help=(
            "Detect low-density ROI bands and save the removed-ROI visualization, "
            "but do not apply the ROI to analysis masks or CSV summaries."
        ),
    )
    parser.add_argument(
        "--density-roi-axis",
        choices=("y", "x", "both"),
        default="y",
        help=(
            "Axis used for automatic density ROI. y removes horizontal bands "
            "based on row density; x removes vertical bands based on column density; "
            "both applies both filters."
        ),
    )
    parser.add_argument(
        "--density-roi-threshold",
        type=float,
        default=0.005,
        help="Rows/columns with vessel density at or below this value are low density.",
    )
    parser.add_argument(
        "--density-roi-min-run",
        type=int,
        default=25,
        help="Minimum consecutive low-density coordinates to remove from ROI.",
    )
    parser.add_argument(
        "--density-roi-bridge-gap",
        type=int,
        default=15,
        help=(
            "High-density gaps no wider than this between two low-density runs "
            "are also removed."
        ),
    )

    # Terminal-spur pruning
    parser.add_argument(
        "--prune-branch-length",
        type=float,
        default=0.0,
        help=(
            "Maximum terminal-branch path length in pixels. "
            "Use 0 to disable the fixed-length criterion."
        ),
    )
    parser.add_argument(
        "--prune-branch-length-physical",
        type=float,
        default=None,
        help=(
            "Maximum terminal-branch path length in physical units. Requires "
            "physical scaling and overrides --prune-branch-length."
        ),
    )
    parser.add_argument(
        "--prune-diameter-factor",
        type=float,
        default=0.0,
        help=(
            "Also regard a terminal branch as short when its length is at most "
            "this value times its local vessel diameter. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--prune-probability-ratio",
        type=float,
        default=0.75,
        help=(
            "A branch is weak when mean branch probability / mean parent probability "
            "is at or below this value."
        ),
    )
    parser.add_argument(
        "--prune-verticality-min",
        type=float,
        default=0.80,
        help=(
            "Minimum absolute y-component of the branch principal axis for it to be "
            "treated as y-direction noise. Range: 0 to 1."
        ),
    )
    parser.add_argument(
        "--prune-mode",
        choices=("conservative", "balanced", "aggressive"),
        default="balanced",
        help=(
            "conservative: remove very short branches or branches that are both weak "
            "and vertical; balanced: weak or vertical evidence is enough; "
            "aggressive: remove every short terminal branch."
        ),
    )
    parser.add_argument(
        "--prune-max-iterations",
        type=int,
        default=20,
        help="Maximum repeated terminal-pruning passes.",
    )
    parser.add_argument(
        "--parent-sample-length",
        type=float,
        default=20.0,
        help="Approximate path length sampled from the parent vessel near a junction.",
    )
    parser.add_argument(
        "--parent-sample-length-physical",
        type=float,
        default=None,
        help=(
            "Parent-vessel sample length in physical units. Requires physical "
            "scaling and overrides --parent-sample-length."
        ),
    )

    # Optional cleanup of tiny links/loops inside thick vessels.
    parser.add_argument(
        "--prune-junction-link-length",
        type=float,
        default=0.0,
        help=(
            "Remove junction-to-junction links at or below this path length. "
            "This can suppress small box-like loops, but may remove real anastomoses. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--prune-junction-link-length-physical",
        type=float,
        default=None,
        help=(
            "Junction-link pruning length in physical units. Requires physical "
            "scaling and overrides --prune-junction-link-length."
        ),
    )
    parser.add_argument(
        "--prune-junction-link-diameter-factor",
        type=float,
        default=0.0,
        help=(
            "Remove a junction-to-junction link when its path length is at most "
            "this value times its median local diameter. Use 0 to disable."
        ),
    )

    # Segment geometry
    parser.add_argument(
        "--smooth-skeleton-window",
        type=int,
        default=0,
        help=(
            "Odd Savitzky-Golay window for centerline-coordinate smoothing. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--smooth-skeleton-line-width",
        type=int,
        default=1,
        help="Line width used only for visualization of smoothed centerlines.",
    )
    parser.add_argument(
        "--resample-spacing",
        type=float,
        default=1.0,
        help="Arc-length spacing in pixels for resampled centerline points.",
    )
    parser.add_argument(
        "--resample-spacing-physical",
        type=float,
        default=None,
        help=(
            "Arc-length spacing in physical units. Requires physical scaling and "
            "overrides --resample-spacing."
        ),
    )
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=10.0,
        help="Exclude shorter segments from quantitative geometry CSV files.",
    )
    parser.add_argument(
        "--min-segment-length-physical",
        type=float,
        default=None,
        help=(
            "Minimum segment length in physical units. Requires physical scaling "
            "and overrides --min-segment-length."
        ),
    )
    parser.add_argument(
        "--curvature-window",
        type=int,
        default=21,
        help="Odd Savitzky-Golay derivative window for curvature.",
    )
    parser.add_argument(
        "--curvature-end-margin",
        type=float,
        default=5.0,
        help="Minimum distance in pixels excluded at both segment ends.",
    )
    parser.add_argument(
        "--curvature-end-margin-physical",
        type=float,
        default=None,
        help=(
            "End margin in physical units. Requires physical scaling and "
            "overrides --curvature-end-margin."
        ),
    )
    parser.add_argument(
        "--curvature-diameter-margin-factor",
        type=float,
        default=1.5,
        help=(
            "Also exclude this many median vessel diameters at both segment ends "
            "when calculating curvature."
        ),
    )
    parser.add_argument(
        "--curvature-margin-max-fraction",
        type=float,
        default=0.20,
        help=(
            "Cap the excluded curvature margin at this fraction of segment length "
            "per end. Range: 0 to <0.5."
        ),
    )
    parser.add_argument(
        "--junction-branch-min-length",
        type=float,
        default=5.0,
        help=(
            "Minimum path length in pixels for an outgoing junction branch to be "
            "reported in junction_branch_curvatures.csv."
        ),
    )
    parser.add_argument(
        "--junction-branch-min-length-physical",
        type=float,
        default=None,
        help=(
            "Minimum junction branch length in physical units. Requires physical "
            "scaling and overrides --junction-branch-min-length."
        ),
    )
    parser.add_argument(
        "--pixel-size-um",
        type=float,
        default=None,
        help=(
            "Optional physical pixel size in micrometres/pixel. "
            "Adds length_um, diameter_um, and curvature_per_um outputs."
        ),
    )
    parser.add_argument(
        "--physical-width",
        type=float,
        default=None,
        help=(
            "Physical width of the full input image. Use with --physical-height "
            "and --physical-unit to add calibrated outputs."
        ),
    )
    parser.add_argument(
        "--physical-height",
        type=float,
        default=None,
        help=(
            "Physical height of the full input image. Use with --physical-width "
            "and --physical-unit to support non-square pixel scaling."
        ),
    )
    parser.add_argument(
        "--physical-unit",
        choices=("um", "mm", "cm"),
        default="um",
        help="Physical unit for --physical-width and --physical-height.",
    )
    parser.add_argument(
        "--normal-measure-step",
        type=float,
        default=0.5,
        help="Subpixel step size for measuring vessel width along the skeleton normal.",
    )
    parser.add_argument(
        "--normal-max-half-width",
        type=float,
        default=200.0,
        help="Maximum one-sided normal search distance in pixels.",
    )
    parser.add_argument(
        "--thick-vessel-percentile",
        type=float,
        default=75.0,
        help=(
            "Percentile of segment median normal diameter used to define thick "
            "vessel trunks when --thick-vessel-min-diameter is omitted."
        ),
    )
    parser.add_argument(
        "--thick-vessel-min-diameter",
        type=float,
        default=None,
        help="Absolute minimum median normal diameter in pixels for thick vessel trunks.",
    )
    parser.add_argument(
        "--thick-vessel-min-length",
        type=float,
        default=50.0,
        help="Minimum total trunk length in pixels for a thick vessel group to be counted.",
    )
    parser.add_argument(
        "--thick-vessel-left-width",
        type=float,
        default=0.0,
        help=(
            "Only count thick vessel trunks that touch the leftmost N pixels. "
            "Use 0 to disable this left-edge criterion."
        ),
    )
    parser.add_argument(
        "--analyze-split-merge",
        action="store_true",
        help=(
            "Detect two or more segment edges sharing the same split and merge "
            "nodes, build a virtual centerline, and combine their diameters."
        ),
    )
    parser.add_argument(
        "--split-merge-min-branch-length",
        type=float,
        default=10.0,
        help="Minimum branch length in pixels for split-merge aggregation.",
    )
    parser.add_argument(
        "--split-merge-node-radius",
        type=float,
        default=3.0,
        help=(
            "Merge nearby graph endpoints within this radius before detecting "
            "parallel split-merge branches."
        ),
    )
    parser.add_argument(
        "--networkx-cycle-basis-limit",
        type=int,
        default=5000,
        help=(
            "Only materialize networkx.cycle_basis when the graph cycle rank is "
            "at or below this value. The cycle rank itself is always calculated."
        ),
    )
    parser.add_argument(
        "--microvessel-max-diameter",
        type=float,
        default=5.0,
        help=(
            "Maximum median normal diameter in pixels for a segment to be counted "
            "as microvessel length in vascular_metrics_summary.csv."
        ),
    )
    parser.add_argument(
        "--small-component-max-length",
        type=float,
        default=50.0,
        help=(
            "Connected skeleton components shorter than this pixel length are "
            "counted in small_component_length_fraction."
        ),
    )
    parser.add_argument(
        "--high-curvature-percentile",
        type=float,
        default=95.0,
        help="Percentile used as the high-curvature threshold for summary metrics.",
    )
    parser.add_argument(
        "--min-cycle-length",
        type=float,
        default=0.0,
        help=(
            "Minimum sknw cycle length to include in vascular_cycles.csv and "
            "overlay. Unit is controlled by --cycle-length-unit."
        ),
    )
    parser.add_argument(
        "--max-cycle-length",
        type=float,
        default=None,
        help=(
            "Maximum sknw cycle length to include in vascular_cycles.csv and "
            "overlay. Unit is controlled by --cycle-length-unit. Leave unset "
            "to include all larger cycles."
        ),
    )
    parser.add_argument(
        "--min-cycle-equivalent-diameter",
        type=float,
        default=0.0,
        help=(
            "Minimum equivalent loop diameter for enclosed-face cycles. Unit is "
            "controlled by --cycle-length-unit."
        ),
    )
    parser.add_argument(
        "--max-cycle-shape-index",
        type=float,
        default=None,
        help=(
            "Discard enclosed-face cycles whose perimeter^2/(4*pi*area) is "
            "larger than this value. Thin slit-like artifacts have high values."
        ),
    )
    parser.add_argument(
        "--cycle-length-unit",
        choices=("px", "physical"),
        default="px",
        help=(
            "Unit for --min-cycle-length and --max-cycle-length. Use physical "
            "with --physical-width/--physical-height or --pixel-size-um to "
            "filter cycles by the real image unit."
        ),
    )
    parser.add_argument(
        "--enable-cycle-orientation-filter",
        action="store_true",
        help=(
            "Use a BICROS-inspired orientation-score check only on detected "
            "cycle candidates. Cycles without enough multi-orientation "
            "junction-like support are removed. Disabled by default."
        ),
    )
    parser.add_argument(
        "--cycle-orientation-count",
        type=int,
        default=24,
        help="Number of orientations in the cycle orientation-score filter.",
    )
    parser.add_argument(
        "--cycle-orientation-sigma-parallel",
        type=float,
        default=4.0,
        help="Line-response sigma along the vessel direction for cycle filtering.",
    )
    parser.add_argument(
        "--cycle-orientation-sigma-perp",
        type=float,
        default=1.2,
        help="Line-response sigma across the vessel direction for cycle filtering.",
    )
    parser.add_argument(
        "--cycle-orientation-response-threshold",
        type=float,
        default=1.2,
        help=(
            "Dominant orientation threshold in global orientation-score standard "
            "deviations, matching the paper's t=1.2*sigma idea."
        ),
    )
    parser.add_argument(
        "--cycle-orientation-sample-radius",
        type=int,
        default=2,
        help=(
            "Pixel radius around each detected cycle boundary point used when "
            "sampling orientation-score support."
        ),
    )
    parser.add_argument(
        "--cycle-orientation-min-dominant",
        type=int,
        default=3,
        help="Minimum dominant orientations for a sampled point to be junction-like.",
    )
    parser.add_argument(
        "--cycle-orientation-min-junction-points",
        type=int,
        default=2,
        help=(
            "Minimum number of sampled junction-like points required to keep a "
            "cycle candidate."
        ),
    )
    parser.add_argument(
        "--cycle-orientation-min-junction-fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum fraction of sampled cycle-boundary points that must be "
            "junction-like to keep a cycle candidate."
        ),
    )

    # Optional multicontrast prediction/consensus input generation.
    parser.add_argument(
        "--multicontrast-image",
        type=Path,
        default=None,
        help=(
            "Run U-Net prediction on multiple contrast variants of this image, "
            "build one consensus probability map, then analyze that map. "
            "When omitted, --input is analyzed directly."
        ),
    )
    parser.add_argument(
        "--multicontrast-model",
        type=Path,
        default=None,
        help="Model checkpoint used with --multicontrast-image. Defaults to predict.DEFAULT_MODEL.",
    )
    parser.add_argument(
        "--multicontrast-device",
        default="auto",
        help='Device for multicontrast prediction, such as "auto", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--multicontrast-frame",
        type=int,
        default=0,
        help="Frame index when reading a multi-frame TIFF for multicontrast prediction.",
    )
    parser.add_argument("--multicontrast-sigma-min", type=int, default=2)
    parser.add_argument("--multicontrast-sigma-max", type=int, default=8)
    parser.add_argument("--multicontrast-frangi-input-weight", type=float, default=0.35)
    parser.add_argument("--multicontrast-post-frangi-weight", type=float, default=0.15)
    parser.add_argument(
        "--multicontrast-consensus",
        choices=("median", "mean", "weighted-mean"),
        default="median",
        help="Conservative probability map used for final graph analysis.",
    )
    parser.add_argument(
        "--multicontrast-support-threshold",
        type=float,
        default=0.45,
        help="A contrast variant supports a vessel pixel when probability is at least this value.",
    )
    parser.add_argument(
        "--multicontrast-min-support",
        type=int,
        default=2,
        help="Minimum number of contrast variants that must support a vessel pixel.",
    )
    parser.add_argument(
        "--multicontrast-final-threshold",
        type=float,
        default=0.45,
        help=(
            "Threshold used for the multicontrast consensus mask. If --threshold "
            "or --measurement-threshold are omitted, this value is used for them."
        ),
    )
    parser.add_argument(
        "--disable-multicontrast-orientation-filter",
        action="store_true",
        help=(
            "Disable cross-contrast orientation filtering. By default, overlapping "
            "detections with large tangent-angle disagreement are excluded before "
            "final graph/cycle analysis."
        ),
    )
    parser.add_argument(
        "--multicontrast-max-angle-diff-deg",
        type=float,
        default=45.0,
        help="Maximum allowed axial tangent-angle disagreement across contrast variants.",
    )
    parser.add_argument(
        "--multicontrast-orientation-match-radius",
        type=float,
        default=1.25,
        help="Pixel radius used to compare nearby skeleton orientations across variants.",
    )
    parser.add_argument(
        "--multicontrast-orientation-window-radius",
        type=int,
        default=3,
        help="Pixel radius around each skeleton point used to estimate its tangent.",
    )
    parser.add_argument(
        "--multicontrast-orientation-dilate-radius",
        type=int,
        default=0,
        help="Dilate ambiguous-orientation pixels by this radius before suppression.",
    )
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    if args.sigma_y < 0 or args.sigma_x < 0:
        raise ValueError("Gaussian sigma values must be >= 0.")
    if args.min_object_size < 0 or args.min_hole_size < 0:
        raise ValueError("Object/hole area thresholds must be >= 0.")
    if args.mask_closing_radius < 0:
        raise ValueError("--mask-closing-radius must be >= 0.")
    if args.skeleton_min_hole_size is not None and args.skeleton_min_hole_size < 0:
        raise ValueError("--skeleton-min-hole-size must be >= 0.")
    if args.skeleton_loop_hole_size < 0:
        raise ValueError("--skeleton-loop-hole-size must be >= 0.")
    if args.measurement_min_hole_size is not None and args.measurement_min_hole_size < 0:
        raise ValueError("--measurement-min-hole-size must be >= 0.")
    if not 0 <= args.density_roi_threshold <= 1:
        raise ValueError("--density-roi-threshold must be between 0 and 1.")
    if args.density_roi_min_run < 1:
        raise ValueError("--density-roi-min-run must be >= 1.")
    if args.density_roi_bridge_gap < 0:
        raise ValueError("--density-roi-bridge-gap must be >= 0.")
    if args.multi_output_input is not None and args.multicontrast_image is not None:
        raise ValueError("--multi-output-input and --multicontrast-image cannot be used together.")
    if args.distance_to_diameter_scale <= 0:
        raise ValueError("--distance-to-diameter-scale must be > 0.")
    if args.centerline_threshold is not None and not 0 <= args.centerline_threshold <= 1:
        raise ValueError("--centerline-threshold must be in [0, 1].")
    if args.prune_branch_length < 0 or args.prune_diameter_factor < 0:
        raise ValueError("Pruning length parameters must be >= 0.")
    for name in (
        "prune_branch_length_physical",
        "parent_sample_length_physical",
        "prune_junction_link_length_physical",
        "resample_spacing_physical",
        "min_segment_length_physical",
        "curvature_end_margin_physical",
        "junction_branch_min_length_physical",
    ):
        value = getattr(args, name, None)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0.")
    if not 0 <= args.prune_probability_ratio <= 2:
        raise ValueError("--prune-probability-ratio must be between 0 and 2.")
    if not 0 <= args.prune_verticality_min <= 1:
        raise ValueError("--prune-verticality-min must be between 0 and 1.")
    if args.prune_max_iterations < 1:
        raise ValueError("--prune-max-iterations must be >= 1.")
    if args.parent_sample_length <= 0:
        raise ValueError("--parent-sample-length must be > 0.")
    if args.resample_spacing <= 0:
        raise ValueError("--resample-spacing must be > 0.")
    if args.min_segment_length < 0:
        raise ValueError("--min-segment-length must be >= 0.")
    if args.junction_branch_min_length < 0:
        raise ValueError("--junction-branch-min-length must be >= 0.")
    if args.curvature_end_margin < 0:
        raise ValueError("--curvature-end-margin must be >= 0.")
    if args.curvature_diameter_margin_factor < 0:
        raise ValueError("--curvature-diameter-margin-factor must be >= 0.")
    if not 0 <= args.curvature_margin_max_fraction < 0.5:
        raise ValueError("--curvature-margin-max-fraction must be in [0, 0.5).")
    if args.pixel_size_um is not None and args.pixel_size_um <= 0:
        raise ValueError("--pixel-size-um must be > 0.")
    if (args.physical_width is None) != (args.physical_height is None):
        raise ValueError("--physical-width and --physical-height must be used together.")
    if args.physical_width is not None and args.physical_width <= 0:
        raise ValueError("--physical-width must be > 0.")
    if args.physical_height is not None and args.physical_height <= 0:
        raise ValueError("--physical-height must be > 0.")
    if args.normal_measure_step <= 0:
        raise ValueError("--normal-measure-step must be > 0.")
    if args.normal_max_half_width <= 0:
        raise ValueError("--normal-max-half-width must be > 0.")
    if not 0 <= args.thick_vessel_percentile <= 100:
        raise ValueError("--thick-vessel-percentile must be between 0 and 100.")
    if args.thick_vessel_min_diameter is not None and args.thick_vessel_min_diameter < 0:
        raise ValueError("--thick-vessel-min-diameter must be >= 0.")
    if args.thick_vessel_min_length < 0:
        raise ValueError("--thick-vessel-min-length must be >= 0.")
    if args.thick_vessel_left_width < 0:
        raise ValueError("--thick-vessel-left-width must be >= 0.")
    if args.split_merge_min_branch_length < 0:
        raise ValueError("--split-merge-min-branch-length must be >= 0.")
    if args.split_merge_node_radius < 0:
        raise ValueError("--split-merge-node-radius must be >= 0.")
    if args.networkx_cycle_basis_limit < 0:
        raise ValueError("--networkx-cycle-basis-limit must be >= 0.")
    if args.microvessel_max_diameter < 0:
        raise ValueError("--microvessel-max-diameter must be >= 0.")
    if args.small_component_max_length < 0:
        raise ValueError("--small-component-max-length must be >= 0.")
    if not 0 <= args.high_curvature_percentile <= 100:
        raise ValueError("--high-curvature-percentile must be between 0 and 100.")
    if args.min_cycle_length < 0:
        raise ValueError("--min-cycle-length must be >= 0.")
    if args.min_cycle_equivalent_diameter < 0:
        raise ValueError("--min-cycle-equivalent-diameter must be >= 0.")
    if args.max_cycle_shape_index is not None and args.max_cycle_shape_index < 1:
        raise ValueError("--max-cycle-shape-index must be >= 1.")
    if args.max_cycle_length is not None and args.max_cycle_length < 0:
        raise ValueError("--max-cycle-length must be >= 0.")
    if (
        args.max_cycle_length is not None
        and args.max_cycle_length < args.min_cycle_length
    ):
        raise ValueError("--max-cycle-length must be >= --min-cycle-length.")
    if args.cycle_orientation_count < 4:
        raise ValueError("--cycle-orientation-count must be >= 4.")
    if args.cycle_orientation_sigma_parallel <= 0:
        raise ValueError("--cycle-orientation-sigma-parallel must be > 0.")
    if args.cycle_orientation_sigma_perp <= 0:
        raise ValueError("--cycle-orientation-sigma-perp must be > 0.")
    if args.cycle_orientation_response_threshold < 0:
        raise ValueError("--cycle-orientation-response-threshold must be >= 0.")
    if args.cycle_orientation_sample_radius < 0:
        raise ValueError("--cycle-orientation-sample-radius must be >= 0.")
    if args.cycle_orientation_min_dominant < 1:
        raise ValueError("--cycle-orientation-min-dominant must be >= 1.")
    if args.cycle_orientation_min_junction_points < 0:
        raise ValueError("--cycle-orientation-min-junction-points must be >= 0.")
    if not 0 <= args.cycle_orientation_min_junction_fraction <= 1:
        raise ValueError("--cycle-orientation-min-junction-fraction must be in [0, 1].")
    if args.multicontrast_image is not None:
        if not 0 <= args.multicontrast_support_threshold <= 1:
            raise ValueError("--multicontrast-support-threshold must be in [0, 1].")
        if not 0 <= args.multicontrast_final_threshold <= 1:
            raise ValueError("--multicontrast-final-threshold must be in [0, 1].")
        if args.multicontrast_min_support < 1:
            raise ValueError("--multicontrast-min-support must be at least 1.")
        if not 0 <= args.multicontrast_max_angle_diff_deg <= 90:
            raise ValueError("--multicontrast-max-angle-diff-deg must be in [0, 90].")
        if args.multicontrast_orientation_match_radius < 0:
            raise ValueError("--multicontrast-orientation-match-radius must be >= 0.")
        if args.multicontrast_orientation_window_radius < 1:
            raise ValueError("--multicontrast-orientation-window-radius must be at least 1.")
        if args.multicontrast_orientation_dilate_radius < 0:
            raise ValueError("--multicontrast-orientation-dilate-radius must be >= 0.")
