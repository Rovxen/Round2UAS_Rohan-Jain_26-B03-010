"""
optimise.py - choose the order in which the rover visits casualties.

The score for a casualty is:

    (straight-line distance from start / distance travelled so far) x priority

Distance travelled only grows, so a casualty visited late scores less
than the same casualty visited early.

Note we visit every casualty, never a subset. Appending a stop to the
end of the order leaves every earlier score untouched and adds a
positive term, so skipping anyone is never an improvement. The problem
is purely about ordering.
"""

import numpy as np


def score_order(order, D, straight, priority):
    """Total path score for one visiting order.

    order    : list of point indices, in visiting sequence
    D        : D[i][j] = travel distance from point i to point j
    straight : straight-line distance from the start to each point
    priority : priority score of each point
    """
    total = 0.0
    travelled = 0.0
    here = 0                                  # index 0 is the start

    for c in order:
        travelled += D[here, c]
        if travelled > 0:
            total += (straight[c] / travelled) * priority[c]
        here = c

    return total


def greedy_build(candidates, D, straight, priority):
    """Build an order by repeatedly taking the best single insertion."""
    order = []
    remaining = set(candidates)

    while remaining:
        best_gain, best_order, best_pick = -np.inf, None, None
        current = score_order(order, D, straight, priority)

        for c in remaining:
            for pos in range(len(order) + 1):
                trial = order[:pos] + [c] + order[pos:]
                gain = score_order(trial, D, straight, priority) - current
                if gain > best_gain:
                    best_gain, best_order, best_pick = gain, trial, c

        # Every casualty gets placed eventually, even if its gain is tiny.
        order = best_order
        remaining.discard(best_pick)

    return order


def improve(order, D, straight, priority, rounds=200, verbose=False):
    """Polish the order with two kinds of local move.

    swap     : exchange the positions of two stops
    relocate : lift one stop out and reinsert it elsewhere, leaving the
               rest in sequence. A swap cannot do this, because it drags
               a second stop along with it.

    Repeats until a full round finds no improvement.
    """
    best = list(order)
    best_score = score_order(best, D, straight, priority)

    if verbose:
        print(f"  improve() starting from {best_score:.4f}")

    for r in range(rounds):
        changed = False

        # --- swap ---
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                trial = list(best)
                trial[i], trial[j] = trial[j], trial[i]
                s = score_order(trial, D, straight, priority)
                if s > best_score + 1e-9:
                    best, best_score, changed = trial, s, True

        # --- relocate ---
        for i in range(len(best)):
            stop = best[i]
            without = best[:i] + best[i + 1:]
            for pos in range(len(without) + 1):
                if pos == i:
                    continue                  # same position, no change
                trial = without[:pos] + [stop] + without[pos:]
                s = score_order(trial, D, straight, priority)
                if s > best_score + 1e-9:
                    best, best_score, changed = trial, s, True
                    break
            if changed:
                break

        if verbose:
            print(f"  round {r}: {best_score:.4f}")

        if not changed:
            break

    return best, best_score


def random_restarts(candidates, D, straight, priority, tries=40, seed=0):
    """Polish several random starting orders and keep the best result.

    Local search can stall in a poor local optimum. Starting from
    different random orders and keeping the winner guards against that.
    """
    rng = np.random.default_rng(seed)

    best_order = None
    best_score = -np.inf

    for _ in range(tries):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        order, score = improve(shuffled, D, straight, priority)
        if score > best_score:
            best_order, best_score = order, score

    return best_order, best_score


def plan(points_xy, people, D, verbose=True):
    """Full ordering step.

    points_xy : [start, casualty1, ..., casualtyN, goal] as (x, y)
    people    : the casualty dicts, in the same order as points 1..N
    D         : all-pairs travel distance matrix

    Returns (order, score, straight, priority).
    """
    n = len(points_xy)
    sx, sy = points_xy[0]

    straight = np.zeros(n)
    priority = np.zeros(n)
    for i in range(1, n - 1):
        x, y = points_xy[i]
        straight[i] = np.hypot(x - sx, y - sy)
        priority[i] = people[i - 1]["priority"]

    # Skip anything walled off from the start.
    candidates = [i for i in range(1, n - 1) if not np.isinf(D[0, i])]
    if len(candidates) < n - 2:
        print(f"  warning: {n - 2 - len(candidates)} casualties unreachable")

    greedy = greedy_build(candidates, D, straight, priority)
    greedy_score = score_order(greedy, D, straight, priority)

    polished, polished_score = improve(greedy, D, straight, priority)
    random_order, random_score = random_restarts(candidates, D, straight, priority)

    if verbose:
        print(f"  greedy          {greedy_score:.4f}")
        print(f"  after polish    {polished_score:.4f}")
        print(f"  random restarts {random_score:.4f}")

    if random_score > polished_score:
        return random_order, random_score, straight, priority
    return polished, polished_score, straight, priority