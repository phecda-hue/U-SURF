from .common import (
    argparse,
    convolve,
    cv2,
    map_coordinates,
    np,
    savgol_filter,
)

from .physical_scale import path_length

def segment_component_id(segment: np.ndarray, labels: np.ndarray) -> int:
    coords = np.rint(segment).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, labels.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, labels.shape[1] - 1)
    component_ids = labels[coords[:, 0], coords[:, 1]]
    component_ids = component_ids[component_ids > 0]
    if component_ids.size == 0:
        return 0
    values, counts = np.unique(component_ids, return_counts=True)
    return int(values[np.argmax(counts)])

def neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    )

def skeleton_neighbors(
    point: tuple[int, int],
    skeleton: np.ndarray,
) -> list[tuple[int, int]]:
    y, x = point
    neighbors: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if (
                0 <= yy < skeleton.shape[0]
                and 0 <= xx < skeleton.shape[1]
                and skeleton[yy, xx]
            ):
                neighbors.append((yy, xx))
    return neighbors

def sample_image(
    image: np.ndarray,
    points_yx: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    if len(points_yx) == 0:
        return np.empty(0, dtype=np.float32)
    values = map_coordinates(
        image.astype(np.float32),
        [points_yx[:, 0], points_yx[:, 1]],
        order=order,
        mode="nearest",
    )
    return values.astype(np.float32)

def inside_mask_at(mask: np.ndarray, y: float, x: float) -> bool:
    yy = int(round(y))
    xx = int(round(x))
    return (
        0 <= yy < mask.shape[0]
        and 0 <= xx < mask.shape[1]
        and bool(mask[yy, xx])
    )

def ray_distance_to_background(
    mask: np.ndarray,
    y: float,
    x: float,
    direction_yx: np.ndarray,
    step: float,
    max_distance: float,
) -> float:
    distance = 0.0
    last_inside = 0.0

    while distance + step <= max_distance:
        distance += step
        yy = y + direction_yx[0] * distance
        xx = x + direction_yx[1] * distance
        if not inside_mask_at(mask, yy, xx):
            break
        last_inside = distance

    return float(last_inside)

def normal_diameters_for_segment(
    segment: np.ndarray,
    mask: np.ndarray,
    step: float,
    max_half_width: float,
    diameter_hints_px: np.ndarray | None = None,
) -> np.ndarray:
    diameters = np.full(len(segment), np.nan, dtype=np.float32)
    if len(segment) < 2:
        return diameters

    for index, point in enumerate(segment):
        prev_index = max(0, index - 2)
        next_index = min(len(segment) - 1, index + 2)
        tangent = segment[next_index] - segment[prev_index]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-8:
            continue

        tangent = tangent / norm
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        y, x = float(point[0]), float(point[1])
        if not inside_mask_at(mask, y, x):
            continue
        local_max_half_width = max_half_width
        if diameter_hints_px is not None and index < len(diameter_hints_px):
            hint = float(diameter_hints_px[index])
            if np.isfinite(hint) and hint > 0:
                local_max_half_width = min(
                    max_half_width,
                    max(4.0 * step, 3.0 * hint),
                )

        positive = ray_distance_to_background(
            mask,
            y,
            x,
            normal,
            step,
            local_max_half_width,
        )
        negative = ray_distance_to_background(
            mask,
            y,
            x,
            -normal,
            step,
            local_max_half_width,
        )
        diameters[index] = positive + negative + step

    return diameters

def branch_verticality(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    centered = points.astype(np.float64) - np.mean(points, axis=0, keepdims=True)
    if np.allclose(centered, 0):
        return 0.0
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    principal_yx = vh[0]
    return float(abs(principal_yx[0]))

def trace_until_node(
    start: tuple[int, int],
    nxt: tuple[int, int],
    skeleton: np.ndarray,
    degrees: np.ndarray,
    max_length: float | None = None,
) -> list[tuple[int, int]]:
    path = [start, nxt]
    prev, current = start, nxt
    accumulated = float(np.hypot(current[0] - prev[0], current[1] - prev[1]))

    while degrees[current] == 2:
        candidates = [
            p for p in skeleton_neighbors(current, skeleton)
            if p != prev
        ]
        if not candidates:
            break

        next_point = candidates[0]
        step = float(
            np.hypot(
                next_point[0] - current[0],
                next_point[1] - current[1],
            )
        )
        if max_length is not None and accumulated + step > max_length:
            break

        path.append(next_point)
        accumulated += step
        prev, current = current, next_point

    return path

def parent_pixels_near_junction(
    junction: tuple[int, int],
    terminal_path: list[tuple[int, int]],
    skeleton: np.ndarray,
    degrees: np.ndarray,
    sample_length: float,
) -> np.ndarray:
    forbidden = set(terminal_path[:-1])
    collected: list[tuple[int, int]] = []

    for neighbor in skeleton_neighbors(junction, skeleton):
        if neighbor in forbidden:
            continue
        parent_path = trace_until_node(
            junction,
            neighbor,
            skeleton,
            degrees,
            max_length=sample_length,
        )
        collected.extend(parent_path[1:])

    if not collected:
        return np.asarray([junction], dtype=np.float32)

    return np.asarray(sorted(set(collected)), dtype=np.float32)

def pruning_decision(
    branch_length_px: float,
    local_diameter_px: float,
    probability_ratio: float,
    verticality: float,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    fixed_short = (
        args.prune_branch_length > 0
        and branch_length_px <= args.prune_branch_length
    )
    adaptive_short = (
        args.prune_diameter_factor > 0
        and local_diameter_px > 0
        and branch_length_px <= args.prune_diameter_factor * local_diameter_px
    )
    short = fixed_short or adaptive_short
    if not short:
        return False, ""

    weak = np.isfinite(probability_ratio) and (
        probability_ratio <= args.prune_probability_ratio
    )
    vertical = verticality >= args.prune_verticality_min

    # A branch shorter than roughly one local diameter is usually a skeleton spur
    # caused by boundary roughness rather than a resolvable vessel.
    very_short = (
        local_diameter_px > 0
        and branch_length_px <= max(3.0, 1.0 * local_diameter_px)
    )

    if very_short:
        return True, "very_short_vs_diameter"
    if args.prune_mode == "aggressive":
        return True, "short"
    if args.prune_mode == "balanced" and (weak or vertical):
        return True, "short_and_" + ("weak" if weak else "vertical")
    if args.prune_mode == "conservative" and weak and vertical:
        return True, "short_weak_and_vertical"
    return False, ""

def prune_terminal_branches(
    skeleton: np.ndarray,
    probability: np.ndarray,
    diameter_map: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if (
        args.prune_branch_length <= 0
        and args.prune_diameter_factor <= 0
    ):
        return skeleton.copy(), np.zeros_like(skeleton, dtype=bool), []

    pruned = skeleton.copy()
    removed_mask = np.zeros_like(skeleton, dtype=bool)
    records: list[dict] = []
    positive_diameters = diameter_map[diameter_map > 0]
    diameter_trace_hint = (
        float(np.percentile(positive_diameters, 99))
        if positive_diameters.size
        else 0.0
    )
    trace_limit = max(
        args.prune_branch_length,
        args.prune_diameter_factor * diameter_trace_hint * 1.25,
        2.0,
    )

    for iteration in range(1, args.prune_max_iterations + 1):
        degrees = neighbor_count(pruned)
        endpoints = [
            tuple(point)
            for point in np.argwhere(pruned & (degrees == 1))
        ]
        candidates_to_remove: list[tuple[list[tuple[int, int]], dict]] = []

        for endpoint in endpoints:
            neighbors = skeleton_neighbors(endpoint, pruned)
            if not neighbors:
                continue

            path = trace_until_node(
                endpoint,
                neighbors[0],
                pruned,
                degrees,
                max_length=trace_limit,
            )
            if len(path) < 2:
                continue

            is_terminal = (
                degrees[path[0]] == 1
                and degrees[path[-1]] >= 3
            )
            if not is_terminal:
                continue

            branch_without_junction = path[:-1]
            branch_points = np.asarray(branch_without_junction, dtype=np.float32)
            junction = path[-1]
            length_px = path_length(path)

            branch_probabilities = sample_image(
                probability,
                branch_points,
                order=1,
            )
            branch_diameters = sample_image(
                diameter_map,
                branch_points,
                order=1,
            )
            positive_diameters = branch_diameters[branch_diameters > 0]
            local_diameter = float(
                np.median(positive_diameters)
                if positive_diameters.size
                else diameter_map[junction]
            )

            parent_points = parent_pixels_near_junction(
                junction,
                path,
                pruned,
                degrees,
                args.parent_sample_length,
            )
            parent_probabilities = sample_image(
                probability,
                parent_points,
                order=1,
            )

            branch_probability = float(
                np.mean(branch_probabilities)
                if branch_probabilities.size
                else 0.0
            )
            parent_probability = float(
                np.mean(parent_probabilities)
                if parent_probabilities.size
                else 0.0
            )
            probability_ratio = (
                branch_probability / parent_probability
                if parent_probability > 1e-8
                else np.nan
            )
            verticality = branch_verticality(
                np.asarray(path, dtype=np.float32)
            )

            remove, reason = pruning_decision(
                length_px,
                local_diameter,
                probability_ratio,
                verticality,
                args,
            )
            if not remove:
                continue

            record = {
                "iteration": iteration,
                "endpoint_y": int(endpoint[0]),
                "endpoint_x": int(endpoint[1]),
                "junction_y": int(junction[0]),
                "junction_x": int(junction[1]),
                "length_px": float(length_px),
                "local_diameter_px": float(local_diameter),
                "branch_probability": branch_probability,
                "parent_probability": parent_probability,
                "probability_ratio": float(probability_ratio),
                "verticality": float(verticality),
                "reason": reason,
            }
            candidates_to_remove.append((branch_without_junction, record))

        if not candidates_to_remove:
            break

        # Collect first, remove second, so endpoint order cannot change decisions
        # inside one pruning pass.
        for branch, record in candidates_to_remove:
            for y, x in branch:
                pruned[y, x] = False
                removed_mask[y, x] = True
            records.append(record)

    return pruned, removed_mask, records

def trace_segments_with_nodes(
    skeleton: np.ndarray,
) -> list[np.ndarray]:
    degrees = neighbor_count(skeleton)
    pixels = {tuple(point) for point in np.argwhere(skeleton)}
    nodes = {point for point in pixels if degrees[point] != 2}
    visited_edges: set[frozenset] = set()
    segments: list[np.ndarray] = []

    def edge_key(
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> frozenset:
        return frozenset((first, second))

    def trace_from(
        start: tuple[int, int],
        nxt: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [start, nxt]
        prev, current = start, nxt
        visited_edges.add(edge_key(prev, current))

        while current not in nodes:
            candidates = [
                point
                for point in skeleton_neighbors(current, skeleton)
                if point != prev
            ]
            if not candidates:
                break

            next_point = candidates[0]
            key = edge_key(current, next_point)
            if key in visited_edges:
                break

            visited_edges.add(key)
            path.append(next_point)
            prev, current = current, next_point

        return path

    # Open segments between endpoints/junctions.
    for node in sorted(nodes):
        for neighbor in skeleton_neighbors(node, skeleton):
            if edge_key(node, neighbor) in visited_edges:
                continue
            path = trace_from(node, neighbor)
            if len(path) >= 2:
                segments.append(np.asarray(path, dtype=np.float32))

    # Closed cycles have no nodes, so trace all remaining unvisited edges.
    for pixel in sorted(pixels):
        for neighbor in skeleton_neighbors(pixel, skeleton):
            if edge_key(pixel, neighbor) in visited_edges:
                continue
            path = trace_from(pixel, neighbor)
            if len(path) >= 2:
                segments.append(np.asarray(path, dtype=np.float32))

    return segments

def prune_short_junction_links(
    skeleton: np.ndarray,
    diameter_map: np.ndarray,
    max_length_px: float,
    diameter_factor: float,
    max_iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if max_length_px <= 0 and diameter_factor <= 0:
        return skeleton.copy(), np.zeros_like(skeleton, dtype=bool), []

    pruned = skeleton.copy()
    removed_mask = np.zeros_like(skeleton, dtype=bool)
    records: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        degrees = neighbor_count(pruned)
        positive_diameters = diameter_map[diameter_map > 0]
        diameter_trace_hint = (
            float(np.percentile(positive_diameters, 99))
            if positive_diameters.size
            else 0.0
        )
        trace_limit = max(
            max_length_px,
            diameter_factor * diameter_trace_hint * 1.25,
            2.0,
        )
        junctions = [
            tuple(point)
            for point in np.argwhere(pruned & (degrees >= 3))
        ]
        visited_edges: set[frozenset] = set()
        segments: list[np.ndarray] = []

        for junction in junctions:
            for neighbor in skeleton_neighbors(junction, pruned):
                key = frozenset((junction, neighbor))
                if key in visited_edges:
                    continue
                path = trace_until_node(
                    junction,
                    neighbor,
                    pruned,
                    degrees,
                    max_length=trace_limit,
                )
                for first, second in zip(path[:-1], path[1:]):
                    visited_edges.add(frozenset((first, second)))
                if len(path) >= 2 and degrees[path[-1]] >= 3:
                    segments.append(np.asarray(path, dtype=np.float32))
        removed_in_iteration = 0

        for segment in segments:
            start = tuple(np.rint(segment[0]).astype(int))
            end = tuple(np.rint(segment[-1]).astype(int))
            if start == end:
                continue
            if degrees[start] < 3 or degrees[end] < 3:
                continue

            length_px = path_length(segment)
            diameters = sample_image(diameter_map, segment, order=1)
            positive = diameters[diameters > 0]
            median_diameter = float(np.median(positive)) if positive.size else 0.0

            fixed_short = max_length_px > 0 and length_px <= max_length_px
            adaptive_short = (
                diameter_factor > 0
                and median_diameter > 0
                and length_px <= diameter_factor * median_diameter
            )
            if not (fixed_short or adaptive_short):
                continue

            # Preserve junction pixels and remove only the link interior.
            interior = np.rint(segment[1:-1]).astype(int)
            if len(interior) == 0:
                continue
            for y, x in interior:
                pruned[y, x] = False
                removed_mask[y, x] = True

            records.append(
                {
                    "iteration": iteration,
                    "start_y": int(start[0]),
                    "start_x": int(start[1]),
                    "end_y": int(end[0]),
                    "end_x": int(end[1]),
                    "length_px": float(length_px),
                    "median_diameter_px": float(median_diameter),
                }
            )
            removed_in_iteration += 1

        if removed_in_iteration == 0:
            break

    return pruned, removed_mask, records

def valid_savgol_window(
    segment_length: int,
    requested_window: int,
    polyorder: int = 3,
) -> int | None:
    minimum = polyorder + 2
    if minimum % 2 == 0:
        minimum += 1

    if requested_window <= 0 or segment_length < minimum:
        return None

    window = (
        requested_window
        if requested_window % 2 == 1
        else requested_window + 1
    )
    if window > segment_length:
        window = (
            segment_length
            if segment_length % 2 == 1
            else segment_length - 1
        )
    if window < minimum:
        return None
    return int(window)

def resample_segment(
    segment: np.ndarray,
    spacing: float,
) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float64)
    if len(segment) < 2:
        return segment.astype(np.float32)

    step_lengths = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step_lengths)])
    keep = np.concatenate([[True], np.diff(cumulative) > 1e-8])
    cumulative = cumulative[keep]
    segment = segment[keep]

    total = cumulative[-1]
    if total <= 1e-8 or len(segment) < 2:
        return segment.astype(np.float32)

    sample_count = max(2, int(np.floor(total / spacing)) + 1)
    new_distance = np.linspace(0.0, total, sample_count)
    y = np.interp(new_distance, cumulative, segment[:, 0])
    x = np.interp(new_distance, cumulative, segment[:, 1])
    return np.column_stack([y, x]).astype(np.float32)

def smooth_segment_coordinates(
    segment: np.ndarray,
    requested_window: int,
) -> np.ndarray:
    window = valid_savgol_window(len(segment), requested_window)
    if window is None:
        return segment.copy()

    y = savgol_filter(
        segment[:, 0],
        window_length=window,
        polyorder=3,
        mode="interp",
    )
    x = savgol_filter(
        segment[:, 1],
        window_length=window,
        polyorder=3,
        mode="interp",
    )

    # Keep graph endpoints fixed so adjacent segments still meet at the
    # original endpoint or junction.
    y[0], x[0] = segment[0, 0], segment[0, 1]
    y[-1], x[-1] = segment[-1, 0], segment[-1, 1]
    return np.column_stack([y, x]).astype(np.float32)

def prepare_segments(
    raw_segments: list[np.ndarray],
    spacing: float,
    smooth_window: int,
) -> list[np.ndarray]:
    prepared: list[np.ndarray] = []
    for segment in raw_segments:
        uniform = resample_segment(segment, spacing)
        smoothed = smooth_segment_coordinates(uniform, smooth_window)
        uniform_again = resample_segment(smoothed, spacing)
        if len(uniform_again) >= 2:
            prepared.append(uniform_again)
    return prepared

def segment_curvature(
    segment: np.ndarray,
    spacing: float,
    requested_window: int,
    end_margin_px: float,
    scale_y: float = 1.0,
    scale_x: float = 1.0,
) -> np.ndarray:
    if len(segment) < 7:
        return np.full(len(segment), np.nan, dtype=np.float32)

    window = valid_savgol_window(
        len(segment),
        requested_window,
        polyorder=3,
    )
    if window is None:
        return np.full(len(segment), np.nan, dtype=np.float32)

    y = segment[:, 0].astype(np.float64) * float(scale_y)
    x = segment[:, 1].astype(np.float64) * float(scale_x)

    dx = savgol_filter(
        x,
        window_length=window,
        polyorder=3,
        deriv=1,
        delta=spacing,
        mode="interp",
    )
    dy = savgol_filter(
        y,
        window_length=window,
        polyorder=3,
        deriv=1,
        delta=spacing,
        mode="interp",
    )
    ddx = savgol_filter(
        x,
        window_length=window,
        polyorder=3,
        deriv=2,
        delta=spacing,
        mode="interp",
    )
    ddy = savgol_filter(
        y,
        window_length=window,
        polyorder=3,
        deriv=2,
        delta=spacing,
        mode="interp",
    )

    denominator = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        np.abs(dx * ddy - dy * ddx),
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-8,
    )

    margin_points = int(np.ceil(end_margin_px / spacing))
    if margin_points > 0:
        curvature[:margin_points] = np.nan
        curvature[-margin_points:] = np.nan

    return curvature.astype(np.float32)

def rasterize_segments(
    segments: list[np.ndarray],
    shape: tuple[int, int],
    line_width: int,
) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    thickness = max(1, int(line_width))

    for segment in segments:
        if len(segment) < 2:
            continue
        points = np.rint(segment[:, [1, 0]]).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, shape[0] - 1)
        cv2.polylines(
            image,
            [points],
            isClosed=False,
            color=1,
            thickness=thickness,
        )
    return image.astype(bool)
