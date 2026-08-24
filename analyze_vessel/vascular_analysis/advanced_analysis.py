from .common import (
    Path,
    argparse,
    cv2,
    np,
    plt,
)

from .physical_scale import path_length, path_length_scaled, physical_scale_from_args

from .skeleton_geometry import (
    neighbor_count,
    normal_diameters_for_segment,
    prepare_segments,
    resample_segment,
    sample_image,
    segment_curvature,
    skeleton_neighbors,
)

from .visualization import parse_id_list, safe_stat

def node_key(point_yx: np.ndarray) -> tuple[int, int]:
    point = np.rint(point_yx).astype(int)
    return int(point[0]), int(point[1])

def build_segment_graph(segment_rows: list[dict]) -> tuple[dict, dict]:
    nodes: dict[tuple[int, int], set[int]] = {}
    segments: dict[int, dict] = {}

    for row in segment_rows:
        segment_id = int(row["segment_id"])
        start = row["start_node"]
        end = row["end_node"]
        segments[segment_id] = row
        nodes.setdefault(start, set()).add(segment_id)
        nodes.setdefault(end, set()).add(segment_id)

    return nodes, segments

def connected_segment_groups(segment_ids: set[int], nodes: dict, segments: dict) -> list[set[int]]:
    groups: list[set[int]] = []
    unseen = set(segment_ids)

    while unseen:
        start_segment = unseen.pop()
        group = {start_segment}
        stack = [start_segment]

        while stack:
            segment_id = stack.pop()
            row = segments[segment_id]
            for node in (row["start_node"], row["end_node"]):
                for neighbor_id in nodes.get(node, set()):
                    if neighbor_id not in unseen:
                        continue
                    unseen.remove(neighbor_id)
                    group.add(neighbor_id)
                    stack.append(neighbor_id)

        groups.append(group)

    return groups

def longest_path_in_segment_tree(
    segment_ids: set[int],
    nodes: dict,
    segments: dict,
    length_key: str = "length_px",
) -> float:
    if not segment_ids:
        return 0.0
    adjacency: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    for segment_id in segment_ids:
        row = segments[segment_id]
        start = row["start_node"]
        end = row["end_node"]
        length = float(row.get(length_key, row["length_px"]))
        adjacency.setdefault(start, []).append((end, length))
        adjacency.setdefault(end, []).append((start, length))

    best = 0.0

    def dfs(node: tuple[int, int], visited_edges: set[int], length: float) -> None:
        nonlocal best
        best = max(best, length)
        for neighbor, edge_length in adjacency.get(node, []):
            edge = hash(frozenset((node, neighbor)))
            if edge in visited_edges:
                continue
            visited_edges.add(edge)
            dfs(neighbor, visited_edges, length + edge_length)
            visited_edges.remove(edge)

    # Exact DFS is fine for small trunks; cap very large groups by using two
    # Dijkstra-like sweeps below.
    if len(segment_ids) <= 80:
        for node in adjacency:
            dfs(node, set(), 0.0)
        return float(best)

    def farthest_from(source: tuple[int, int]) -> tuple[tuple[int, int], float]:
        distances = {source: 0.0}
        visited = set()
        while len(visited) < len(adjacency):
            current = min(
                (node for node in distances if node not in visited),
                key=lambda node: distances[node],
                default=None,
            )
            if current is None:
                break
            visited.add(current)
            for neighbor, edge_length in adjacency.get(current, []):
                candidate = distances[current] + edge_length
                if candidate > distances.get(neighbor, -np.inf):
                    distances[neighbor] = candidate
        farthest = max(distances, key=lambda node: distances[node])
        return farthest, float(distances[farthest])

    first_node = next(iter(adjacency))
    farthest, _ = farthest_from(first_node)
    _, distance = farthest_from(farthest)
    return float(distance)

def thick_vessel_analysis(segment_rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    if not segment_rows:
        return [], {
            "thick_vessel_threshold_px": np.nan,
            "thick_vessel_count": 0,
            "thick_vessel_branch_count": 0,
            "thick_vessel_max_path_px": 0.0,
        }

    values = np.asarray(
        [row["median_normal_diameter_px"] for row in segment_rows],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        threshold = np.nan
    elif args.thick_vessel_min_diameter is not None:
        threshold = float(args.thick_vessel_min_diameter)
    else:
        threshold = float(np.percentile(values, args.thick_vessel_percentile))

    nodes, segments = build_segment_graph(segment_rows)
    scale = physical_scale_from_args(args)
    unit = scale["unit"] if scale["enabled"] else ""
    length_key = f"length_{unit}" if scale["enabled"] else "length_px"
    thick_segment_ids = {
        int(row["segment_id"])
        for row in segment_rows
        if np.isfinite(row["median_normal_diameter_px"])
        and row["median_normal_diameter_px"] >= threshold
    }
    thick_groups = connected_segment_groups(thick_segment_ids, nodes, segments)

    trunk_rows = []
    total_branch_count = 0
    max_path = 0.0

    trunk_id = 0
    for group in thick_groups:
        length = float(sum(segments[segment_id]["length_px"] for segment_id in group))
        length_for_filter = length
        if scale["enabled"]:
            length_for_filter = float(
                sum(
                    segments[segment_id].get(length_key, segments[segment_id]["length_px"])
                    for segment_id in group
                )
            )
        if length < args.thick_vessel_min_length:
            continue
        group_min_x = min(float(segments[segment_id]["min_x"]) for segment_id in group)
        if args.thick_vessel_left_width > 0 and group_min_x > args.thick_vessel_left_width:
            continue

        trunk_id += 1
        group_nodes = set()
        for segment_id in group:
            row = segments[segment_id]
            group_nodes.add(row["start_node"])
            group_nodes.add(row["end_node"])

        branch_segment_ids = set()
        for node in group_nodes:
            for neighbor_id in nodes.get(node, set()):
                if neighbor_id not in group:
                    branch_segment_ids.add(neighbor_id)

        diameters = np.asarray(
            [segments[segment_id]["median_normal_diameter_px"] for segment_id in group],
            dtype=np.float64,
        )
        longest_path = longest_path_in_segment_tree(group, nodes, segments)
        longest_path_physical = (
            longest_path_in_segment_tree(group, nodes, segments, length_key=length_key)
            if scale["enabled"]
            else np.nan
        )
        branch_count = len(branch_segment_ids)
        total_branch_count += branch_count
        max_path = max(max_path, longest_path)

        trunk_rows.append(
            {
                "thick_vessel_id": trunk_id,
                "segment_count": len(group),
                "branch_count": branch_count,
                "total_length_px": length,
                "longest_path_px": longest_path,
                "min_x": group_min_x,
                "median_normal_diameter_px": safe_stat(diameters, np.median),
                "mean_normal_diameter_px": safe_stat(diameters, np.mean),
                "segment_ids": ";".join(str(segment_id) for segment_id in sorted(group)),
                "branch_segment_ids": ";".join(str(segment_id) for segment_id in sorted(branch_segment_ids)),
            }
        )
        if scale["enabled"]:
            trunk_rows[-1][f"total_length_{unit}"] = length_for_filter
            trunk_rows[-1][f"longest_path_{unit}"] = longest_path_physical

    summary = {
        "thick_vessel_threshold_px": threshold,
        "thick_vessel_min_length_px": float(args.thick_vessel_min_length),
        "thick_vessel_left_width_px": float(args.thick_vessel_left_width),
        "thick_vessel_count": len(trunk_rows),
        "thick_vessel_branch_count": total_branch_count,
        "thick_vessel_max_path_px": max_path,
    }
    return trunk_rows, summary

def resample_segment_and_profile(
    segment: np.ndarray,
    profile: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample one branch and its pointwise diameter on normalized arc length."""
    segment = np.asarray(segment, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)

    if len(segment) < 2:
        return segment.astype(np.float32), profile.astype(np.float32)

    steps = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-8])
    cumulative = cumulative[keep]
    segment = segment[keep]
    profile = profile[keep]

    if len(segment) < 2 or cumulative[-1] <= 1e-8:
        return segment.astype(np.float32), profile.astype(np.float32)

    source_t = cumulative / cumulative[-1]
    target_t = np.linspace(0.0, 1.0, max(2, int(sample_count)))
    y = np.interp(target_t, source_t, segment[:, 0])
    x = np.interp(target_t, source_t, segment[:, 1])

    finite = np.isfinite(profile) & (profile > 0)
    if finite.sum() >= 2:
        diameter = np.interp(target_t, source_t[finite], profile[finite])
    elif finite.sum() == 1:
        diameter = np.full_like(target_t, profile[finite][0], dtype=np.float64)
    else:
        diameter = np.full_like(target_t, np.nan, dtype=np.float64)

    return (
        np.column_stack([y, x]).astype(np.float32),
        diameter.astype(np.float32),
    )

def split_merge_analysis(
    segment_rows: list[dict],
    segment_geometries: dict[int, np.ndarray],
    normal_diameter_profiles: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict[int, np.ndarray]]:
    """
    Detect direct split-merge structures.

    Two or more graph edges that share the same unordered endpoint pair are
    interpreted as parallel branches between one split node and one merge node.
    """
    if not args.analyze_split_merge:
        return [], [], {}

    rows_by_id = {
        int(row["segment_id"]): row
        for row in segment_rows
    }

    # Skeleton junctions often occupy several adjacent pixels. The segment
    # tracer then gives the two parallel branches endpoints such as
    # (59, 66) and (61, 66), although they belong to the same split node.
    # Cluster nearby endpoints inside each connected component first.
    endpoint_items: list[tuple[int, tuple[int, int]]] = []
    for row in segment_rows:
        component_id = int(row.get("component_id", 0))
        endpoint_items.append((component_id, tuple(row["start_node"])))
        endpoint_items.append((component_id, tuple(row["end_node"])))
    endpoint_items = sorted(set(endpoint_items))

    parent = list(range(len(endpoint_items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    radius = float(args.split_merge_node_radius)
    if radius > 0:
        for first in range(len(endpoint_items)):
            component_first, point_first = endpoint_items[first]
            for second in range(first + 1, len(endpoint_items)):
                component_second, point_second = endpoint_items[second]
                if component_first != component_second:
                    continue
                if np.hypot(
                    point_first[0] - point_second[0],
                    point_first[1] - point_second[1],
                ) <= radius:
                    union(first, second)

    cluster_members: dict[int, list[tuple[int, int]]] = {}
    for index, (_, point) in enumerate(endpoint_items):
        cluster_members.setdefault(find(index), []).append(point)

    cluster_representative: dict[int, tuple[int, int]] = {}
    for root, points in cluster_members.items():
        array = np.asarray(points, dtype=np.float64)
        representative = tuple(np.rint(array.mean(axis=0)).astype(int))
        cluster_representative[root] = representative

    canonical_node: dict[tuple[int, tuple[int, int]], tuple[int, int]] = {}
    for index, item in enumerate(endpoint_items):
        canonical_node[item] = cluster_representative[find(index)]

    groups: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
    canonical_endpoints_by_id: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    for row in segment_rows:
        segment_id = int(row["segment_id"])
        if float(row["length_px"]) < args.split_merge_min_branch_length:
            continue
        component_id = int(row.get("component_id", 0))
        start = canonical_node[(component_id, tuple(row["start_node"]))]
        end = canonical_node[(component_id, tuple(row["end_node"]))]
        if start == end:
            continue
        canonical_endpoints_by_id[segment_id] = (start, end)
        key = tuple(sorted((start, end)))
        groups.setdefault(key, []).append(segment_id)

    group_rows: list[dict] = []
    point_rows: list[dict] = []
    virtual_geometries: dict[int, np.ndarray] = {}
    group_id = 0

    for (node_a, node_b), candidate_ids in sorted(groups.items()):
        if len(candidate_ids) < 2:
            continue

        branch_ids: list[int] = []
        oriented_segments: list[np.ndarray] = []
        oriented_diameters: list[np.ndarray] = []
        branch_lengths: list[float] = []

        for segment_id in sorted(candidate_ids):
            row = rows_by_id.get(segment_id)
            segment = segment_geometries.get(segment_id)
            diameter = normal_diameter_profiles.get(segment_id)
            if row is None or segment is None or diameter is None:
                continue

            canonical_start, _ = canonical_endpoints_by_id[segment_id]
            if canonical_start != node_a:
                segment = segment[::-1].copy()
                diameter = diameter[::-1].copy()

            branch_ids.append(segment_id)
            oriented_segments.append(segment)
            oriented_diameters.append(diameter)
            branch_lengths.append(float(row["length_px"]))

        if len(branch_ids) < 2:
            continue

        sample_count = max(
            7,
            int(np.ceil(max(branch_lengths) / args.resample_spacing)) + 1,
        )
        sampled_segments: list[np.ndarray] = []
        sampled_diameters: list[np.ndarray] = []

        for segment, diameter in zip(oriented_segments, oriented_diameters):
            sampled_segment, sampled_diameter = resample_segment_and_profile(
                segment,
                diameter,
                sample_count,
            )
            sampled_segments.append(sampled_segment)
            sampled_diameters.append(sampled_diameter)

        coordinates = np.stack(sampled_segments, axis=0).astype(np.float64)
        diameters = np.stack(sampled_diameters, axis=0).astype(np.float64)

        # Replace isolated invalid values with that branch's median.
        for branch_index in range(diameters.shape[0]):
            finite = np.isfinite(diameters[branch_index]) & (
                diameters[branch_index] > 0
            )
            if finite.any():
                diameters[branch_index, ~finite] = np.median(
                    diameters[branch_index, finite]
                )

        valid = np.isfinite(diameters) & (diameters > 0)
        area_weights = np.where(valid, diameters ** 2, 0.0)
        weight_sum = area_weights.sum(axis=0)

        # Area-weighted midpoint becomes the representative centerline.
        virtual = np.empty((sample_count, 2), dtype=np.float64)
        for point_index in range(sample_count):
            if weight_sum[point_index] > 0:
                virtual[point_index] = (
                    coordinates[:, point_index, :]
                    * area_weights[:, point_index, None]
                ).sum(axis=0) / weight_sum[point_index]
            else:
                virtual[point_index] = coordinates[:, point_index, :].mean(axis=0)

        # Two useful definitions:
        # 1) summed diameter = d1 + d2 + ...
        # 2) area-equivalent diameter = sqrt(d1^2 + d2^2 + ...)
        summed_diameter = np.nansum(
            np.where(valid, diameters, np.nan),
            axis=0,
        )
        area_equivalent_diameter = np.sqrt(
            np.nansum(
                np.where(valid, diameters ** 2, np.nan),
                axis=0,
            )
        )

        virtual = resample_segment(
            virtual.astype(np.float32),
            args.resample_spacing,
        )
        source_t = np.linspace(0.0, 1.0, len(summed_diameter))
        target_t = np.linspace(0.0, 1.0, len(virtual))
        summed_diameter = np.interp(
            target_t,
            source_t,
            summed_diameter,
        ).astype(np.float32)
        area_equivalent_diameter = np.interp(
            target_t,
            source_t,
            area_equivalent_diameter,
        ).astype(np.float32)

        virtual_length = path_length(virtual)
        representative_diameter = safe_stat(
            area_equivalent_diameter,
            np.median,
            default=0.0,
        )
        raw_margin = max(
            args.curvature_end_margin,
            args.curvature_diameter_margin_factor * representative_diameter,
        )
        curvature_margin = min(
            raw_margin,
            args.curvature_margin_max_fraction * virtual_length,
        )
        curvature = segment_curvature(
            virtual,
            spacing=args.resample_spacing,
            requested_window=args.curvature_window,
            end_margin_px=curvature_margin,
        )
        scale = physical_scale_from_args(args)
        curvature_physical = None
        if scale["enabled"]:
            mean_scale = float(scale["mean_per_px"])
            curvature_physical = segment_curvature(
                virtual,
                spacing=args.resample_spacing * mean_scale,
                requested_window=args.curvature_window,
                end_margin_px=curvature_margin * mean_scale,
                scale_y=float(scale["y_per_px"]),
                scale_x=float(scale["x_per_px"]),
            )
        valid_curvature = curvature[np.isfinite(curvature)]
        valid_curvature_physical = (
            curvature_physical[np.isfinite(curvature_physical)]
            if curvature_physical is not None
            else np.asarray([], dtype=np.float32)
        )

        group_id += 1
        virtual_geometries[group_id] = virtual
        group_row = {
            "split_merge_id": group_id,
            "node_a_y": int(node_a[0]),
            "node_a_x": int(node_a[1]),
            "node_b_y": int(node_b[0]),
            "node_b_x": int(node_b[1]),
            "branch_count": len(branch_ids),
            "branch_segment_ids": ";".join(map(str, branch_ids)),
            "branch_lengths_px": ";".join(
                f"{value:.6f}" for value in branch_lengths
            ),
            "virtual_length_px": float(virtual_length),
            "median_summed_diameter_px": safe_stat(
                summed_diameter,
                np.median,
            ),
            "mean_summed_diameter_px": safe_stat(
                summed_diameter,
                np.mean,
            ),
            "median_area_equivalent_diameter_px": safe_stat(
                area_equivalent_diameter,
                np.median,
            ),
            "mean_area_equivalent_diameter_px": safe_stat(
                area_equivalent_diameter,
                np.mean,
            ),
            "mean_curvature_per_px": safe_stat(
                valid_curvature,
                np.mean,
            ),
            "median_curvature_per_px": safe_stat(
                valid_curvature,
                np.median,
            ),
            "p95_curvature_per_px": safe_stat(
                valid_curvature,
                lambda values: np.percentile(values, 95),
            ),
            "curvature_end_margin_px": float(curvature_margin),
        }
        if scale["enabled"]:
            unit = scale["unit"]
            mean_scale = float(scale["mean_per_px"])
            group_row.update(
                {
                    f"virtual_length_{unit}": path_length_scaled(
                        virtual,
                        scale_y=float(scale["y_per_px"]),
                        scale_x=float(scale["x_per_px"]),
                    ),
                    f"median_summed_diameter_{unit}": group_row[
                        "median_summed_diameter_px"
                    ]
                    * mean_scale,
                    f"mean_summed_diameter_{unit}": group_row[
                        "mean_summed_diameter_px"
                    ]
                    * mean_scale,
                    f"median_area_equivalent_diameter_{unit}": group_row[
                        "median_area_equivalent_diameter_px"
                    ]
                    * mean_scale,
                    f"mean_area_equivalent_diameter_{unit}": group_row[
                        "mean_area_equivalent_diameter_px"
                    ]
                    * mean_scale,
                    f"mean_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        np.mean,
                    ),
                    f"median_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        np.median,
                    ),
                    f"p95_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        lambda values: np.percentile(values, 95),
                    ),
                }
            )
        group_rows.append(group_row)

        for point_index, (
            (y, x),
            diameter_sum,
            diameter_equivalent,
            curvature_value,
        ) in enumerate(
            zip(
                virtual,
                summed_diameter,
                area_equivalent_diameter,
                curvature,
            )
        ):
            point_rows.append(
                {
                    "split_merge_id": group_id,
                    "point_index": point_index,
                    "y": float(y),
                    "x": float(x),
                    "summed_diameter_px": float(diameter_sum),
                    "area_equivalent_diameter_px": float(
                        diameter_equivalent
                    ),
                    "curvature_per_px": float(curvature_value),
                }
            )
            if scale["enabled"]:
                unit = scale["unit"]
                mean_scale = float(scale["mean_per_px"])
                point_rows[-1].update(
                    {
                        f"summed_diameter_{unit}": float(diameter_sum * mean_scale),
                        f"area_equivalent_diameter_{unit}": float(
                            diameter_equivalent * mean_scale
                        ),
                        f"curvature_per_{unit}": float(
                            curvature_physical[point_index]
                            if curvature_physical is not None
                            else np.nan
                        ),
                    }
                )

    return group_rows, point_rows, virtual_geometries

def save_split_merge_overlay(
    probability: np.ndarray,
    segment_geometries: dict[int, np.ndarray],
    split_merge_rows: list[dict],
    virtual_geometries: dict[int, np.ndarray],
    path: Path,
) -> None:
    height, width = probability.shape
    fig_width = 14
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    ax.imshow(probability, cmap="gray")

    for row in split_merge_rows:
        group_id = int(row["split_merge_id"])
        for segment_id in parse_id_list(row["branch_segment_ids"]):
            segment = segment_geometries.get(segment_id)
            if segment is not None and len(segment) >= 2:
                ax.plot(
                    segment[:, 1],
                    segment[:, 0],
                    linewidth=1.0,
                    alpha=0.65,
                )

        virtual = virtual_geometries.get(group_id)
        if virtual is not None and len(virtual) >= 2:
            ax.plot(
                virtual[:, 1],
                virtual[:, 0],
                color="white",
                linewidth=2.0,
                linestyle="--",
            )
            middle = virtual[len(virtual) // 2]
            ax.text(
                middle[1],
                middle[0],
                f"SM{group_id}",
                color="white",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": (0, 0, 0, 0.65),
                    "edgecolor": "white",
                    "linewidth": 0.5,
                },
            )

    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def trace_branch_from_junction(
    skeleton: np.ndarray,
    junction_labels: np.ndarray,
    start_junction_label: int,
    junction_pixel: tuple[int, int],
    first_outside: tuple[int, int],
) -> tuple[np.ndarray, int, str]:
    path = [junction_pixel, first_outside]
    prev = junction_pixel
    current = first_outside

    while True:
        current_label = int(junction_labels[current])
        if current_label > 0 and current_label != start_junction_label:
            return np.asarray(path, dtype=np.float32), current_label, "junction"

        neighbors = [
            point
            for point in skeleton_neighbors(current, skeleton)
            if point != prev and int(junction_labels[point]) != start_junction_label
        ]
        if not neighbors:
            return np.asarray(path, dtype=np.float32), 0, "endpoint"

        non_start_junction_neighbors = [
            point
            for point in neighbors
            if int(junction_labels[point]) > 0
            and int(junction_labels[point]) != start_junction_label
        ]
        if non_start_junction_neighbors:
            next_point = non_start_junction_neighbors[0]
            path.append(next_point)
            return (
                np.asarray(path, dtype=np.float32),
                int(junction_labels[next_point]),
                "junction",
            )

        if len(neighbors) > 1:
            return np.asarray(path, dtype=np.float32), 0, "branching_nonjunction"

        next_point = neighbors[0]
        path.append(next_point)
        prev, current = current, next_point

def junction_branch_curvature_analysis(
    skeleton: np.ndarray,
    probability: np.ndarray,
    diameter_map: np.ndarray,
    measurement_binary: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    degrees = neighbor_count(skeleton)
    junction_mask = skeleton & (degrees >= 3)
    junction_count, junction_labels = cv2.connectedComponents(
        junction_mask.astype(np.uint8),
        connectivity=8,
    )
    branch_rows: list[dict] = []
    point_rows: list[dict] = []
    cluster_rows: list[dict] = []
    branch_id = 0

    for junction_id in range(1, junction_count):
        cluster_pixels = [tuple(point) for point in np.argwhere(junction_labels == junction_id)]
        if not cluster_pixels:
            continue

        cluster_array = np.asarray(cluster_pixels, dtype=np.float32)
        centroid_y, centroid_x = np.mean(cluster_array, axis=0)
        outgoing: list[tuple[tuple[int, int], tuple[int, int]]] = []
        seen_first_pixels: set[tuple[int, int]] = set()
        for pixel in cluster_pixels:
            for neighbor in skeleton_neighbors(pixel, skeleton):
                if int(junction_labels[neighbor]) == junction_id:
                    continue
                if neighbor in seen_first_pixels:
                    continue
                seen_first_pixels.add(neighbor)
                outgoing.append((pixel, neighbor))

        if len(outgoing) < 2:
            continue

        cluster_branch_ids: list[int] = []
        cluster_lengths: list[float] = []
        cluster_probabilities: list[float] = []
        cluster_diameters: list[float] = []
        cluster_normal_diameters: list[float] = []
        cluster_curvatures: list[float] = []
        cluster_curvatures_physical: list[float] = []

        for branch_index, (junction_pixel, first_outside) in enumerate(outgoing, start=1):
            raw_branch, end_junction_id, end_type = trace_branch_from_junction(
                skeleton,
                junction_labels,
                junction_id,
                junction_pixel,
                first_outside,
            )
            prepared = prepare_segments(
                [raw_branch],
                spacing=args.resample_spacing,
                smooth_window=args.smooth_skeleton_window,
            )
            if not prepared:
                continue

            branch = prepared[0]
            length_px = path_length(branch)
            if length_px < args.junction_branch_min_length:
                continue

            diameters_px = sample_image(diameter_map, branch, order=1)
            normal_diameters_px = normal_diameters_for_segment(
                branch,
                measurement_binary,
                step=args.normal_measure_step,
                max_half_width=args.normal_max_half_width,
                diameter_hints_px=diameters_px,
            )
            probabilities = sample_image(probability, branch, order=1)
            positive_diameters = diameters_px[diameters_px > 0]
            positive_normal_diameters = normal_diameters_px[
                np.isfinite(normal_diameters_px) & (normal_diameters_px > 0)
            ]
            median_diameter_px = (
                float(np.median(positive_diameters))
                if positive_diameters.size
                else 0.0
            )
            raw_margin_px = max(
                args.curvature_end_margin,
                args.curvature_diameter_margin_factor * median_diameter_px,
            )
            margin_px = min(
                raw_margin_px,
                args.curvature_margin_max_fraction * length_px,
            )
            curvature_per_px = segment_curvature(
                branch,
                spacing=args.resample_spacing,
                requested_window=args.curvature_window,
                end_margin_px=margin_px,
            )
            scale = physical_scale_from_args(args)
            curvature_per_physical = None
            if scale["enabled"]:
                mean_scale = float(scale["mean_per_px"])
                curvature_per_physical = segment_curvature(
                    branch,
                    spacing=args.resample_spacing * mean_scale,
                    requested_window=args.curvature_window,
                    end_margin_px=margin_px * mean_scale,
                    scale_y=float(scale["y_per_px"]),
                    scale_x=float(scale["x_per_px"]),
                )
            valid_curvature = curvature_per_px[np.isfinite(curvature_per_px)]
            valid_curvature_physical = (
                curvature_per_physical[np.isfinite(curvature_per_physical)]
                if curvature_per_physical is not None
                else np.asarray([], dtype=np.float32)
            )

            branch_id += 1
            branch_row = {
                "junction_id": junction_id,
                "branch_index": branch_index,
                "branch_id": branch_id,
                "junction_degree": len(outgoing),
                "junction_center_y": float(centroid_y),
                "junction_center_x": float(centroid_x),
                "start_y": float(branch[0, 0]),
                "start_x": float(branch[0, 1]),
                "end_y": float(branch[-1, 0]),
                "end_x": float(branch[-1, 1]),
                "end_type": end_type,
                "end_junction_id": end_junction_id,
                "n_points": int(len(branch)),
                "length_px": float(length_px),
                "mean_probability": safe_stat(probabilities, np.mean),
                "mean_diameter_px": safe_stat(positive_diameters, np.mean),
                "median_diameter_px": safe_stat(positive_diameters, np.median),
                "mean_normal_diameter_px": safe_stat(
                    positive_normal_diameters,
                    np.mean,
                ),
                "median_normal_diameter_px": safe_stat(
                    positive_normal_diameters,
                    np.median,
                ),
                "mean_curvature_per_px": safe_stat(valid_curvature, np.mean),
                "median_curvature_per_px": safe_stat(valid_curvature, np.median),
                "p95_curvature_per_px": safe_stat(
                    valid_curvature,
                    lambda values: np.percentile(values, 95),
                ),
                "max_curvature_per_px": safe_stat(valid_curvature, np.max),
                "curvature_end_margin_px": float(margin_px),
            }
            if scale["enabled"]:
                unit = scale["unit"]
                mean_scale = float(scale["mean_per_px"])
                length_physical = path_length_scaled(
                    branch,
                    scale_y=float(scale["y_per_px"]),
                    scale_x=float(scale["x_per_px"]),
                )
                branch_row.update(
                    {
                        f"length_{unit}": float(length_physical),
                        f"mean_diameter_{unit}": float(
                            branch_row["mean_diameter_px"] * mean_scale
                        ),
                        f"median_diameter_{unit}": float(
                            branch_row["median_diameter_px"] * mean_scale
                        ),
                        f"mean_normal_diameter_{unit}": float(
                            branch_row["mean_normal_diameter_px"] * mean_scale
                        ),
                        f"median_normal_diameter_{unit}": float(
                            branch_row["median_normal_diameter_px"] * mean_scale
                        ),
                        f"mean_curvature_per_{unit}": safe_stat(
                            valid_curvature_physical,
                            np.mean,
                        ),
                        f"median_curvature_per_{unit}": safe_stat(
                            valid_curvature_physical,
                            np.median,
                        ),
                        f"p95_curvature_per_{unit}": safe_stat(
                            valid_curvature_physical,
                            lambda values: np.percentile(values, 95),
                        ),
                        f"max_curvature_per_{unit}": safe_stat(
                            valid_curvature_physical,
                            np.max,
                        ),
                    }
                )
            branch_rows.append(branch_row)
            cluster_branch_ids.append(branch_id)
            cluster_lengths.append(float(length_px))
            cluster_probabilities.extend(
                probabilities[np.isfinite(probabilities)].tolist()
            )
            cluster_diameters.extend(positive_diameters.tolist())
            cluster_normal_diameters.extend(positive_normal_diameters.tolist())
            cluster_curvatures.extend(valid_curvature.tolist())
            cluster_curvatures_physical.extend(valid_curvature_physical.tolist())

            for point_index, (
                point,
                probability_value,
                diameter_value,
                normal_value,
                curvature_value,
                curvature_physical,
            ) in enumerate(
                zip(
                    branch,
                    probabilities,
                    diameters_px,
                    normal_diameters_px,
                    curvature_per_px,
                    curvature_per_physical
                    if curvature_per_physical is not None
                    else np.full_like(curvature_per_px, np.nan),
                )
            ):
                row = {
                    "junction_id": junction_id,
                    "branch_id": branch_id,
                    "point_index": point_index,
                    "y": float(point[0]),
                    "x": float(point[1]),
                    "probability": float(probability_value),
                    "diameter_px": float(diameter_value),
                    "normal_diameter_px": float(normal_value),
                    "curvature_per_px": float(curvature_value),
                }
                if scale["enabled"]:
                    unit = scale["unit"]
                    mean_scale = float(scale["mean_per_px"])
                    row.update(
                        {
                            f"diameter_{unit}": float(diameter_value * mean_scale),
                            f"normal_diameter_{unit}": float(
                                normal_value * mean_scale
                            ),
                            f"curvature_per_{unit}": float(curvature_physical),
                        }
                    )
                point_rows.append(row)

        if cluster_branch_ids:
            cluster_length_values = np.asarray(cluster_lengths, dtype=np.float32)
            cluster_probability_values = np.asarray(
                cluster_probabilities,
                dtype=np.float32,
            )
            cluster_diameter_values = np.asarray(cluster_diameters, dtype=np.float32)
            cluster_normal_diameter_values = np.asarray(
                cluster_normal_diameters,
                dtype=np.float32,
            )
            cluster_curvature_values = np.asarray(
                cluster_curvatures,
                dtype=np.float32,
            )
            cluster_curvature_values_physical = np.asarray(
                cluster_curvatures_physical,
                dtype=np.float32,
            )
            cluster_row = {
                "junction_id": junction_id,
                "junction_degree": len(outgoing),
                "cluster_pixel_count": len(cluster_pixels),
                "junction_center_y": float(centroid_y),
                "junction_center_x": float(centroid_x),
                "branch_count": len(cluster_branch_ids),
                "branch_ids": ";".join(str(value) for value in cluster_branch_ids),
                "total_branch_length_px": float(np.sum(cluster_length_values)),
                "mean_branch_length_px": safe_stat(cluster_length_values, np.mean),
                "median_branch_length_px": safe_stat(
                    cluster_length_values,
                    np.median,
                ),
                "mean_probability": safe_stat(cluster_probability_values, np.mean),
                "mean_diameter_px": safe_stat(cluster_diameter_values, np.mean),
                "median_diameter_px": safe_stat(cluster_diameter_values, np.median),
                "mean_normal_diameter_px": safe_stat(
                    cluster_normal_diameter_values,
                    np.mean,
                ),
                "median_normal_diameter_px": safe_stat(
                    cluster_normal_diameter_values,
                    np.median,
                ),
                "mean_curvature_per_px": safe_stat(
                    cluster_curvature_values,
                    np.mean,
                ),
                "median_curvature_per_px": safe_stat(
                    cluster_curvature_values,
                    np.median,
                ),
                "p95_curvature_per_px": safe_stat(
                    cluster_curvature_values,
                    lambda values: np.percentile(values, 95),
                ),
                "max_curvature_per_px": safe_stat(
                    cluster_curvature_values,
                    np.max,
                ),
            }
            if scale["enabled"]:
                unit = scale["unit"]
                mean_scale = float(scale["mean_per_px"])
                cluster_row.update(
                    {
                        f"total_branch_length_{unit}": float(
                            cluster_row["total_branch_length_px"] * mean_scale
                        ),
                        f"mean_branch_length_{unit}": float(
                            cluster_row["mean_branch_length_px"] * mean_scale
                        ),
                        f"median_branch_length_{unit}": float(
                            cluster_row["median_branch_length_px"] * mean_scale
                        ),
                        f"mean_diameter_{unit}": float(
                            cluster_row["mean_diameter_px"] * mean_scale
                        ),
                        f"median_diameter_{unit}": float(
                            cluster_row["median_diameter_px"] * mean_scale
                        ),
                        f"mean_normal_diameter_{unit}": float(
                            cluster_row["mean_normal_diameter_px"] * mean_scale
                        ),
                        f"median_normal_diameter_{unit}": float(
                            cluster_row["median_normal_diameter_px"] * mean_scale
                        ),
                        f"mean_curvature_per_{unit}": safe_stat(
                            cluster_curvature_values_physical,
                            np.mean,
                        ),
                        f"median_curvature_per_{unit}": safe_stat(
                            cluster_curvature_values_physical,
                            np.median,
                        ),
                        f"p95_curvature_per_{unit}": safe_stat(
                            cluster_curvature_values_physical,
                            lambda values: np.percentile(values, 95),
                        ),
                        f"max_curvature_per_{unit}": safe_stat(
                            cluster_curvature_values_physical,
                            np.max,
                        ),
                    }
                )
            cluster_rows.append(cluster_row)

    return branch_rows, point_rows, cluster_rows
