from .common import np


HIDDEN_SUMMARY_OUTPUT_FIELDS = {
    "sknw_available",
    "sknw_error",
    "networkx_cycle_basis_limit",
}


def public_summary_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in HIDDEN_SUMMARY_OUTPUT_FIELDS
        and not key.startswith("networkx_dfs_total_length_")
        and not key.startswith("sknw_total_edge_length_")
        and not key.startswith("vessel_pixel_density_")
        and not key.startswith("density_roi_")
        and not key.startswith("vascular_cycle_density_per_length_")
    }

def public_metrics_summary_row(row: dict) -> dict:
    unit = row.get("physical_unit", "")
    fields = [
        "physical_unit",
        f"vessel_area_density_{unit}2_per_{unit}2" if unit else "vessel_area_density",
        f"skeleton_length_density_{unit}_per_{unit}2" if unit else "skeleton_length_density_per_px",
        f"diameter_median_{unit}" if unit else "diameter_median_px",
        f"diameter_iqr_{unit}" if unit else "diameter_iqr_px",
        f"diameter_p90_{unit}" if unit else "diameter_p90_px",
        f"length_weighted_mean_diameter_{unit}" if unit else "length_weighted_mean_diameter_px",
        "microvessel_length_fraction",
        "tortuosity_length_weighted_mean",
        "tortuosity_p90",
        f"median_curvature_per_{unit}" if unit else "median_curvature_per_px",
        f"high_curvature_length_density_{unit}_per_{unit}2"
        if unit
        else "high_curvature_length_density_per_px",
        f"branch_density_per_{unit}2" if unit else "branch_density_per_px",
        "largest_component_length_fraction",
        f"internal_endpoint_density_per_{unit}2" if unit else "internal_endpoint_density_per_px",
        f"vascular_cycle_density_per_{unit}2" if unit else "vascular_cycle_density_per_px",
        f"vascular_cycle_density_per_{unit}_vessel"
        if unit
        else "vascular_cycle_density_per_vessel_length_px",
        "loop_area_fraction",
        f"loop_area_median_{unit}2" if unit else "loop_area_median_px2",
        f"loop_perimeter_median_{unit}" if unit else "loop_perimeter_median_px",
        f"loop_equivalent_diameter_median_{unit}"
        if unit
        else "loop_equivalent_diameter_median_px",
        f"cycle_length_median_{unit}" if unit else "cycle_length_median_px",
        f"cycle_length_p90_{unit}" if unit else "cycle_length_p90_px",
        "orientation_anisotropy",
        "principal_orientation_deg",
        "horizontal_length_fraction",
        "vertical_length_fraction",
        "diagonal_length_fraction",
    ]
    return {field: row.get(field, np.nan) for field in fields if field in row}
