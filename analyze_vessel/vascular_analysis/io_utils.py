from .common import (
    Image,
    Path,
    argparse,
    cv2,
    np,
)

def normalize_probability_array(array: np.ndarray, source: Path | str) -> np.ndarray:
    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array in {source}, got shape {array.shape}")
    probability = array.astype(np.float32)
    finite = probability[np.isfinite(probability)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        low, high = np.percentile(finite, [0.1, 99.9])
        if high > low:
            probability = (probability - low) / (high - low)
    return np.clip(probability, 0.0, 1.0)

def load_npz_channel(
    path: Path,
    key: str,
    fallback_keys: tuple[str, ...],
) -> np.ndarray:
    with np.load(path) as data:
        candidate_keys = (key, *fallback_keys)
        for candidate in candidate_keys:
            if candidate in data:
                return np.asarray(data[candidate]).copy()
        raise KeyError(
            f"None of {candidate_keys} were found in {path}. "
            f"Available keys: {', '.join(data.files)}"
        )

def load_probability_channel(
    path: Path,
    key: str | None = None,
    fallback_keys: tuple[str, ...] = (),
) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        if key is None:
            raise ValueError(f"A key is required when reading .npz: {path}")
        return normalize_probability_array(
            load_npz_channel(path, key, fallback_keys),
            f"{path}:{key}",
        )
    return load_grayscale(path)

def load_numeric_channel(
    path: Path,
    key: str | None = None,
    fallback_keys: tuple[str, ...] = (),
) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        if key is None:
            raise ValueError(f"A key is required when reading .npz: {path}")
        array = load_npz_channel(path, key, fallback_keys)
    elif path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        array = image

    array = np.squeeze(np.asarray(array)).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D numeric map in {path}, got shape {array.shape}")
    return array

def load_grayscale(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        return normalize_probability_array(array, path)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        probability = image.astype(np.float32) / max_value
    else:
        probability = image.astype(np.float32)
        finite = probability[np.isfinite(probability)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            low, high = np.percentile(finite, [0.1, 99.9])
            if high > low:
                probability = (probability - low) / (high - low)

    return np.clip(probability, 0.0, 1.0)

def resolve_multi_output_maps(args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    mask_probability = None
    centerline_probability = None
    distance_map = None

    if args.multi_output_input is not None:
        path = args.multi_output_input
        mask_probability = load_probability_channel(
            path,
            key=args.mask_key,
            fallback_keys=("P_mask", "probability", "vessel_probability"),
        )
        centerline_probability = load_probability_channel(
            path,
            key=args.centerline_key,
            fallback_keys=("P_centerline", "skeleton", "centerline_probability"),
        )
        distance_map = load_numeric_channel(
            path,
            key=args.distance_key,
            fallback_keys=("D", "distance_map", "distance_transform"),
        )

    if args.centerline_input is not None:
        centerline_probability = load_probability_channel(args.centerline_input)
    if args.distance_input is not None:
        distance_map = load_numeric_channel(args.distance_input)

    return mask_probability, centerline_probability, distance_map

def require_same_shape(reference: np.ndarray, candidate: np.ndarray | None, name: str) -> None:
    if candidate is not None and candidate.shape != reference.shape:
        raise ValueError(
            f"{name} shape {candidate.shape} does not match mask shape {reference.shape}."
        )

def multicontrast_save_probability_png(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.clip(np.rint(probability * 65535.0), 0, 65535).astype(np.uint16)
    ).save(path)
