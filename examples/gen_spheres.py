import argparse
import csv
import math
from pathlib import Path

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0

_BASE_VERTICES = np.array([
    [-1,  PHI,  0],
    [ 1,  PHI,  0],
    [-1, -PHI,  0],
    [ 1, -PHI,  0],
    [ 0, -1,  PHI],
    [ 0,  1,  PHI],
    [ 0, -1, -PHI],
    [ 0,  1, -PHI],
    [ PHI,  0, -1],
    [ PHI,  0,  1],
    [-PHI,  0, -1],
    [-PHI,  0,  1],
], dtype=float)

_BASE_FACES = [
    ( 0, 11,  5), ( 0,  5,  1), ( 0,  1,  7), ( 0,  7, 10), ( 0, 10, 11),
    ( 1,  5,  9), ( 5, 11,  4), (11, 10,  2), (10,  7,  6), ( 7,  1,  8),
    ( 3,  9,  4), ( 3,  4,  2), ( 3,  2,  6), ( 3,  6,  8), ( 3,  8,  9),
    ( 4,  9,  5), ( 2,  4, 11), ( 6,  2, 10), ( 8,  6,  7), ( 9,  8,  1),
]

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def _edge_key(i: int, j: int) -> tuple:
    return (min(i, j), max(i, j))


def subdivide(vertices: list, faces: list) -> tuple[list, list]:
    cache: dict[tuple, int] = {}
    new_faces: list[tuple] = []

    def midpoint(i: int, j: int) -> int:
        key = _edge_key(i, j)
        if key not in cache:
            mid = _normalize(np.asarray(vertices[i]) + np.asarray(vertices[j]))
            cache[key] = len(vertices)
            vertices.append(mid)
        return cache[key]

    for (a, b, c) in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]

    return vertices, new_faces


def rotate_to_north(verts: np.ndarray, src: np.ndarray) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(src, target)
    sin_a = np.linalg.norm(axis)
    if sin_a < 1e-12: 
        if np.dot(src, target) < 0:
            verts = verts * np.array([1.0, -1.0, -1.0])
        return verts
    axis /= sin_a
    cos_a = float(np.dot(src, target))
    dot   = verts @ axis  
    cross = np.cross(verts, axis)
    return cos_a * verts + sin_a * cross + (1.0 - cos_a) * dot[:, None] * axis


def build_sphere(level: int) -> tuple[np.ndarray, list, int]:
    verts = [_normalize(v) for v in _BASE_VERTICES]
    faces = list(_BASE_FACES)

    for _ in range(level):
        verts, faces = subdivide(verts, faces)

    verts = np.array(verts)

    np_idx = int(np.argmin(np.linalg.norm(verts - [0., 0., 1.], axis=1)))
    north  = verts[np_idx].copy()
    verts  = rotate_to_north(verts, north)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    np_idx = int(np.argmin(np.linalg.norm(verts - [0., 0., 1.], axis=1)))
    assert np.linalg.norm(verts[np_idx] - [0., 0., 1.]) < 1e-8, \
        f"North pole not at (0,0,1) after rotation, residual = {np.linalg.norm(verts[np_idx] - [0.,0.,1.])}"

    verts -= verts[np_idx]  

    return verts, faces, np_idx

def mean_edge_length(verts: np.ndarray, faces: list) -> float:
    edges: set[tuple] = set()
    for (a, b, c) in faces:
        edges.update([_edge_key(a, b), _edge_key(b, c), _edge_key(a, c)])
    lengths = [np.linalg.norm(verts[i] - verts[j]) for (i, j) in edges]
    return float(np.mean(lengths))


def ground_truth_distances(verts: np.ndarray) -> np.ndarray:
    vz_unit = verts[:, 2] + 1.0         
    return np.arccos(np.clip(vz_unit, -1.0, 1.0))

def write_obj(path: Path, verts: np.ndarray, faces: list,
              source_idx: int, h: float) -> None:
    with open(path, "w") as f:
        f.write("# Geodesic sphere (subdivided icosahedron)\n")
        f.write(f"# {len(verts)} vertices, {len(faces)} faces\n")
        f.write(f"# mean edge length h = {h:.6f}\n")
        f.write(f"# source vertex (north pole, at origin): {source_idx}\n")
        f.write("# ground-truth distance: arccos(v_z + 1)  "
                "  (v_z is the z-coord as stored in this file)\n\n")
        for v in verts:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        f.write("\n")
        for (a, b, c) in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")   # OBJ is 1-indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5],
                        help="Subdivision levels to generate (default: 0–5)")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory (default: current dir)")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path = args.outdir / "sphere_summary.csv"

    rows = []
    for level in sorted(args.levels):
        verts, faces, src_idx = build_sphere(level)
        h   = mean_edge_length(verts, faces)
        n_v = len(verts)
        n_f = len(faces)

        # Sanity check: source vertex should be at the origin
        assert np.linalg.norm(verts[src_idx]) < 1e-10, \
            f"Source vertex not at origin! level={level}"

        # Sanity check: ground truth at source = 0, at south pole ≈ π
        gt = ground_truth_distances(verts)
        assert gt[src_idx] < 1e-10, "Ground truth distance at source is not 0"

        fname = args.outdir / f"sphere_{n_v}.obj"
        write_obj(fname, verts, faces, src_idx, h)

        rows.append({"level": level, "n_vertices": n_v, "n_faces": n_f,
                     "h_mean": h, "source_vertex": src_idx, "file": fname.name})

        print(f"level {level:2d} | {n_v:6d} vertices | {n_f:6d} faces "
              f"| h = {h:.5f} | source = vertex {src_idx} | → {fname.name}")

    # Write summary CSV
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to {summary_path}")

if __name__ == "__main__":
    main()
