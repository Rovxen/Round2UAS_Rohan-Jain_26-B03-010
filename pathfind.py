"""
pathfind.py - shortest routes across the traversable mask.

Two costs matter and they are different numbers:
  distance - plain pixels, used by the score formula
  time     - pixels divided by terrain speed, used for rankings

We plan on a shrunken copy of the mask for speed, then scale the
resulting coordinates back to full size.
"""

import heapq
import cv2
import numpy as np

from palette import CLASS_INDEX, SPEED

SCALE = 4          # plan at quarter size
INFLATE = 3        # pixels to grow obstacles by, so paths avoid corners

NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, 1.4142), (-1, 1, 1.4142),
              (1, -1, 1.4142), (1, 1, 1.4142)]


def build_grid(labels, mask):
    """Shrink the mask and build a matching cost-per-pixel grid.

    Returns (small_mask, distance_cost, time_cost).
    """
    # Grow obstacles first, at full resolution, so narrow gaps close
    # honestly rather than by accident of resizing.
    blocked = (mask == 0).astype(np.uint8)
    blocked = cv2.dilate(blocked, np.ones((INFLATE * 2 + 1,) * 2, np.uint8))
    free = (blocked == 0).astype(np.uint8)

    h, w = free.shape
    small = (w // SCALE, h // SCALE)

    # INTER_NEAREST keeps values at exactly 0 or 1 - no invented greys.
    small_free = cv2.resize(free, small, interpolation=cv2.INTER_NEAREST) > 0

    # Cost of moving one pixel at each cell. For distance it is always 1.
    dist_cost = np.ones(small_free.shape, dtype=np.float64)

    # For time it is 1/speed, taken from whichever terrain is underneath.
    time_cost = np.full(small_free.shape, 1.0 / 20.0)
    for name, speed in SPEED.items():
        layer = (labels == CLASS_INDEX[name]).astype(np.uint8)
        layer = cv2.resize(layer, small, interpolation=cv2.INTER_NEAREST)
        time_cost[layer > 0] = 1.0 / speed

    return small_free, dist_cost, time_cost


def to_small(xy):
    """Full-size (x, y) -> small-grid (row, col)."""
    x, y = xy
    return (int(y) // SCALE, int(x) // SCALE)


def to_full(rc):
    """Small-grid (row, col) -> full-size (x, y), aimed at the cell centre."""
    r, c = rc
    return (int(c * SCALE + SCALE // 2), int(r * SCALE + SCALE // 2))


def nearest_free(free, rc, radius=12):
    """Snap a point onto traversable ground if inflation swallowed it.

    Casualty centres sometimes end up inside an inflated obstacle.
    """
    r, c = rc
    h, w = free.shape
    r, c = min(max(r, 0), h - 1), min(max(c, 0), w - 1)
    if free[r, c]:
        return (r, c)

    for rad in range(1, radius + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and free[nr, nc]:
                    return (nr, nc)
    raise ValueError(f"no traversable cell near {rc}")


def dijkstra(free, cost, start_rc):
    """Cheapest cost from start_rc to every reachable cell.

    Returns (cost_to_each_cell, previous_cell) so routes can be rebuilt.
    """
    h, w = free.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w, 2), -1, dtype=np.int32)

    dist[start_rc] = 0.0
    queue = [(0.0, start_rc[0], start_rc[1])]

    while queue:
        d, r, c = heapq.heappop(queue)
        if d > dist[r, c]:          # stale entry, already improved
            continue
        for dr, dc, step in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                continue
            nd = d + step * cost[nr, nc]
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[nr, nc] = (r, c)
                heapq.heappush(queue, (nd, nr, nc))

    return dist, prev


def trace(prev, end_rc):
    """Walk the previous-cell trail backwards to rebuild a route."""
    route = []
    r, c = end_rc
    while r >= 0:
        route.append((r, c))
        r, c = prev[r, c]
    route.reverse()
    return route

def all_pairs(free, dist_cost, time_cost, points_rc):
    """Travel distance, travel time, and routes between every pair of points.

    points_rc : list of (row, col) on the small grid

    Returns (D, T, routes) where D[i][j] is pixel distance from point i
    to point j, T[i][j] is seconds, and routes[i][j] is the cell path.
    """
    n = len(points_rc)
    D = np.full((n, n), np.inf)
    T = np.full((n, n), np.inf)
    routes = [[None] * n for _ in range(n)]

    for i, src in enumerate(points_rc):
        # One run gives the cost from src to every other point at once.
        d_map, prev = dijkstra(free, dist_cost, src)
        t_map, _ = dijkstra(free, time_cost, src)

        for j, dst in enumerate(points_rc):
            if i == j:
                D[i, j] = T[i, j] = 0.0
                routes[i][j] = [src]
                continue
            if np.isinf(d_map[dst]):
                continue
            D[i, j] = d_map[dst] * SCALE
            T[i, j] = t_map[dst] * SCALE
            routes[i][j] = trace(prev, dst)

    return D, T, routes

def score_order(order, D, straight, priority):
    """Total path score for visiting casualties in the given order.

    order    : list of casualty indices, in visiting sequence
    straight : straight-line distance from start to each casualty
    priority : priority score of each casualty
    """
    total = 0.0
    travelled = 0.0
    here = 0                      # start point

    for c in order:
        travelled += D[here, c]
        if travelled > 0:
            total += (straight[c] / travelled) * priority[c]
        here = c

    return total