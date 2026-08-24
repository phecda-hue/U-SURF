from .common import cv2, np, nx, sknw

from .visualization import safe_stat

from .physical_scale import path_length, path_length_scaled

def skeleton_component_labels(skeleton: np.ndarray) -> tuple[int, np.ndarray]:
    count, labels = cv2.connectedComponents(
        skeleton.astype(np.uint8),
        connectivity=8,
    )
    return count - 1, labels.astype(np.int32)

def skeleton_component_edge_lengths(
    skeleton: np.ndarray,
    labels: np.ndarray,
    component_count: int,
) -> dict[int, float]:
    lengths = {component_id: 0.0 for component_id in range(1, component_count + 1)}
    height, width = skeleton.shape
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]

    for y, x in np.argwhere(skeleton):
        component_id = int(labels[y, x])
        if component_id <= 0:
            continue
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if yy >= height or xx < 0 or xx >= width:
                continue
            if skeleton[yy, xx] and labels[yy, xx] == component_id:
                lengths[component_id] += float(np.hypot(dy, dx))

    return lengths

def build_networkx_pixel_graph(
    skeleton: np.ndarray,
    scale_y: float = 1.0,
    scale_x: float = 1.0,
) -> nx.Graph:
    graph = nx.Graph()
    coords = [tuple(point) for point in np.argwhere(skeleton)]
    graph.add_nodes_from(coords)

    height, width = skeleton.shape
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    for y, x in coords:
        for dy, dx in offsets:
            yy, xx = y + dy, x + dx
            if yy >= height or xx < 0 or xx >= width:
                continue
            if skeleton[yy, xx]:
                graph.add_edge(
                    (y, x),
                    (yy, xx),
                    weight=float(np.hypot(dy, dx)),
                    physical_weight=float(np.hypot(dy * scale_y, dx * scale_x)),
                )
    return graph

def dfs_total_edge_length(graph: nx.Graph, weight_key: str = "weight") -> float:
    total = 0.0
    visited_nodes = set()
    visited_edges = set()

    for source in graph.nodes:
        if source in visited_nodes:
            continue
        stack = [source]
        visited_nodes.add(source)
        while stack:
            node = stack.pop()
            for neighbor, edge_data in graph[node].items():
                edge_key = tuple(sorted((node, neighbor)))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    total += float(edge_data.get(weight_key, 1.0))
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    stack.append(neighbor)

    return float(total)

def networkx_graph_analysis(
    skeleton: np.ndarray,
    cycle_basis_limit: int,
    scale: dict | None = None,
) -> dict:
    scale = scale or {"enabled": False, "x_per_px": 1.0, "y_per_px": 1.0, "unit": ""}
    graph = build_networkx_pixel_graph(
        skeleton,
        scale_y=float(scale.get("y_per_px", 1.0)) if scale.get("enabled") else 1.0,
        scale_x=float(scale.get("x_per_px", 1.0)) if scale.get("enabled") else 1.0,
    )
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    component_count = nx.number_connected_components(graph) if node_count else 0
    cycle_rank = max(0, edge_count - node_count + component_count)
    total_length = dfs_total_edge_length(graph)
    total_length_physical = (
        dfs_total_edge_length(graph, weight_key="physical_weight")
        if scale.get("enabled")
        else np.nan
    )

    cycle_basis_count = np.nan
    cycle_basis_max_length_px = np.nan
    cycle_basis_mean_length_px = np.nan
    if cycle_rank <= cycle_basis_limit:
        cycles = nx.cycle_basis(graph)
        cycle_lengths = []
        for cycle in cycles:
            length = 0.0
            for index, node in enumerate(cycle):
                neighbor = cycle[(index + 1) % len(cycle)]
                length += float(graph[node][neighbor].get("weight", 1.0))
            cycle_lengths.append(length)
        cycle_values = np.asarray(cycle_lengths, dtype=np.float32)
        cycle_basis_count = len(cycles)
        cycle_basis_max_length_px = safe_stat(cycle_values, np.max)
        cycle_basis_mean_length_px = safe_stat(cycle_values, np.mean)

    summary = {
        "networkx_node_count": node_count,
        "networkx_edge_count": edge_count,
        "networkx_component_count": component_count,
        "networkx_cycle_rank": cycle_rank,
        "networkx_cycle_basis_count": cycle_basis_count,
        "networkx_cycle_basis_limit": cycle_basis_limit,
        "networkx_cycle_basis_mean_length_px": cycle_basis_mean_length_px,
        "networkx_cycle_basis_max_length_px": cycle_basis_max_length_px,
        "networkx_dfs_total_length_px": total_length,
    }
    if scale.get("enabled"):
        summary[f"networkx_dfs_total_length_{scale['unit']}"] = total_length_physical
    return summary

def sknw_graph_analysis(skeleton: np.ndarray, scale: dict | None = None) -> dict:
    scale = scale or {"enabled": False, "x_per_px": 1.0, "y_per_px": 1.0, "unit": ""}
    if sknw is None:
        return {
            "sknw_available": False,
            "sknw_node_count": np.nan,
            "sknw_edge_count": np.nan,
            "sknw_component_count": np.nan,
            "sknw_cycle_rank": np.nan,
            "sknw_total_edge_length_px": np.nan,
        }

    try:
        graph = sknw.build_sknw(skeleton.astype(np.uint16), multi=False)
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        component_count = nx.number_connected_components(graph) if node_count else 0
        cycle_rank = max(0, edge_count - node_count + component_count)
        total_length = 0.0
        total_length_physical = 0.0
        for _, _, edge_data in graph.edges(data=True):
            if "weight" in edge_data:
                total_length += float(edge_data["weight"])
            elif "pts" in edge_data:
                total_length += path_length(edge_data["pts"])
            else:
                total_length += 1.0
            if scale.get("enabled") and "pts" in edge_data:
                total_length_physical += path_length_scaled(
                    edge_data["pts"],
                    scale_y=float(scale["y_per_px"]),
                    scale_x=float(scale["x_per_px"]),
                )
            elif scale.get("enabled"):
                total_length_physical += float(edge_data.get("weight", 1.0)) * float(
                    scale["mean_per_px"]
                )
        summary = {
            "sknw_available": True,
            "sknw_node_count": node_count,
            "sknw_edge_count": edge_count,
            "sknw_component_count": component_count,
            "sknw_cycle_rank": cycle_rank,
            "sknw_total_edge_length_px": float(total_length),
        }
        if scale.get("enabled"):
            summary[f"sknw_total_edge_length_{scale['unit']}"] = float(total_length_physical)
        return summary
    except Exception as error:
        return {
            "sknw_available": True,
            "sknw_node_count": np.nan,
            "sknw_edge_count": np.nan,
            "sknw_component_count": np.nan,
            "sknw_cycle_rank": np.nan,
            "sknw_total_edge_length_px": np.nan,
            "sknw_error": str(error),
        }

def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentiles: list[float],
) -> dict[float, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return {percentile: np.nan for percentile in percentiles}

    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    return {
        percentile: float(np.interp(percentile / 100.0 * total, cumulative, values))
        for percentile in percentiles
    }
