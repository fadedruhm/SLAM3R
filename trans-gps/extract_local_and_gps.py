from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import exifread
import numpy as np
from pyproj import Transformer


ImageGPS = Tuple[Optional[float], Optional[float], Optional[float]]


def parse_scene_traj(traj_path: Path) -> np.ndarray:
    """Parse scene_traj.txt and extract Tx, Ty, Tz from each 4x4 matrix row."""
    points: List[List[float]] = []

    with traj_path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            values = [float(x) for x in line.split()]
            if len(values) != 16:
                raise ValueError(
                    f"{traj_path} line {line_idx} should contain 16 floats, got {len(values)}"
                )

            points.append([values[3], values[7], values[11]])

    return np.asarray(points, dtype=np.float64)


def _ratio_to_float(value) -> float:
    if hasattr(value, "num") and hasattr(value, "den"):
        return float(value.num) / float(value.den)
    return float(value)


def _dms_to_decimal(values: Sequence, ref: str) -> float:
    degrees = _ratio_to_float(values[0])
    minutes = _ratio_to_float(values[1])
    seconds = _ratio_to_float(values[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if ref in ("S", "W"):
        decimal = -decimal

    return decimal


def _parse_gps_altitude(tag_value, altitude_ref_tag) -> float:
    altitude = _ratio_to_float(tag_value.values[0])
    altitude_ref = 0

    if altitude_ref_tag is not None and altitude_ref_tag.values:
        altitude_ref = int(altitude_ref_tag.values[0])

    if altitude_ref == 1:
        altitude = -altitude

    return altitude


def extract_gps_from_image(image_path: Path) -> ImageGPS:
    """Read GPS longitude, latitude and altitude from one image."""
    with image_path.open("rb") as f:
        tags = exifread.process_file(f, details=False)

    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref_tag = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref_tag = tags.get("GPS GPSLongitudeRef")
    alt_tag = tags.get("GPS GPSAltitude")
    alt_ref_tag = tags.get("GPS GPSAltitudeRef")

    if not all([lat_tag, lat_ref_tag, lon_tag, lon_ref_tag]):
        return None, None, None

    latitude = _dms_to_decimal(lat_tag.values, str(lat_ref_tag.values[0]))
    longitude = _dms_to_decimal(lon_tag.values, str(lon_ref_tag.values[0]))
    altitude = _parse_gps_altitude(alt_tag, alt_ref_tag) if alt_tag else None

    return longitude, latitude, altitude


def list_image_files(images_dir: Path) -> List[Path]:
    supported_suffixes = {".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]
    )


def extract_all_gps(images_dir: Path) -> List[ImageGPS]:
    image_paths = list_image_files(images_dir)
    return [extract_gps_from_image(path) for path in image_paths]


def geodetic_to_enu(
    gps_points: Sequence[Tuple[float, float, float]]
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Convert GPS points (lon, lat, alt) to ENU coordinates.
    The first point is used as the ENU origin.
    """
    if not gps_points:
        raise ValueError("gps_points is empty")

    origin_lon, origin_lat, origin_alt = gps_points[0]

    ecef_transformer = Transformer.from_crs(
        "EPSG:4979", "EPSG:4978", always_xy=True
    )

    x0, y0, z0 = ecef_transformer.transform(origin_lon, origin_lat, origin_alt)

    lat0_rad = np.deg2rad(origin_lat)
    lon0_rad = np.deg2rad(origin_lon)

    sin_lat = np.sin(lat0_rad)
    cos_lat = np.cos(lat0_rad)
    sin_lon = np.sin(lon0_rad)
    cos_lon = np.cos(lon0_rad)

    ecef_to_enu = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )

    enu_points = []
    for lon, lat, alt in gps_points:
        x, y, z = ecef_transformer.transform(lon, lat, alt)
        delta = np.array([x - x0, y - y0, z - z0], dtype=np.float64)
        enu = ecef_to_enu @ delta
        enu_points.append(enu)

    return np.asarray(enu_points, dtype=np.float64), (origin_lon, origin_lat, origin_alt)


def umeyama_alignment(
    src_points: np.ndarray, dst_points: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate similarity transform from src_points to dst_points using Umeyama SVD.
    Returns scale, rotation R, translation T, and a 4x4 transform matrix.
    """
    if src_points.shape != dst_points.shape:
        raise ValueError(
            f"Shape mismatch: src_points {src_points.shape}, dst_points {dst_points.shape}"
        )

    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError("Input points must have shape (N, 3)")

    num_points = src_points.shape[0]
    if num_points < 3:
        raise ValueError("At least 3 points are required for 3D similarity alignment")

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    covariance = (dst_centered.T @ src_centered) / num_points
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt

    src_variance = np.sum(src_centered**2) / num_points
    if np.isclose(src_variance, 0.0):
        raise ValueError("Source points variance is zero; cannot estimate scale")

    scale = np.sum(singular_values * np.diag(correction)) / src_variance
    translation = dst_mean - scale * (rotation @ src_mean)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation

    return scale, rotation, translation, transform


def main() -> None:
    root = Path(__file__).resolve().parent
    traj_path = root / "scene_traj.txt"
    images_dir = root / "images"

    local_points_all = parse_scene_traj(traj_path)
    gps_list_all = extract_all_gps(images_dir)

    local_points = local_points_all[3:]
    gps_list = gps_list_all[3:]

    if len(local_points) != len(gps_list):
        raise ValueError(
            f"Count mismatch after dropping first 3 items: "
            f"{len(local_points)} local points vs {len(gps_list)} GPS entries"
        )

    valid_local_points = []
    valid_gps_points = []
    for local_point, gps in zip(local_points, gps_list):
        lon, lat, alt = gps
        if lon is None or lat is None or alt is None:
            continue
        valid_local_points.append(local_point)
        valid_gps_points.append((lon, lat, alt))

    if not valid_gps_points:
        raise ValueError("No valid GPS points found after dropping the first 3 items")

    local_points_filtered = np.asarray(valid_local_points, dtype=np.float64)
    enu_points, origin_gps = geodetic_to_enu(valid_gps_points)

    _, _, _, transform = umeyama_alignment(local_points_filtered, enu_points)

    print("Origin GPS:", origin_gps)
    print("Transform Matrix 4x4:")
    print(transform)
    print("Flattened (column-major / Fortran order):")
    print(transform.flatten("F"))


if __name__ == "__main__":
    main()
