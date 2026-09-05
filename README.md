# Round2UAS_Rohan-Jain_26-B03-010

# UAS-DTU Round 2 — Casualty Analysis using a Rover Guided by a UAV

Given an aerial segmentation map of a triage site, this project finds the
traversable area, locates and classifies every casualty, plans a rover route
from the start marker to the destination, scores that route, times it, and
ranks all five input images.

---

## Running it

```bash
pip install opencv-python numpy
python main.py
```

Input images go in `Images/`. Everything is written to `outputs/`:

```
outputs/
├── masks/<name>_mask.png    traversable (white) vs blocked (black)
├── paths/<name>_path.png    the route drawn on the original image
├── data/<name>.json         casualties, coordinates, scores, timing
├── rankings.json            both global rankings
└── all_results.json         every image in one file
```

The committed `outputs/` folder already contains results for all five images,
so the work can be reviewed without running anything.

---

## Viewing the result

The documented result can be found in result.md it has been purpossed to provide the results to the viewer straight away, the masks and the paths can be found in the outputs folder, where there have been laid all the results.

## Repository layout

| File | Responsibility |
|---|---|
| `palette.py` | Clean JPEG colour noise into flat class labels |
| `mask.py` | Build the traversable / non-traversable mask |
| `casualties.py` | Locate casualties, classify shape and colour, find elevation |
| `pathfind.py` | Dijkstra on the mask; distances between every pair of points |
| `optimise.py` | Choose the order in which casualties are visited |
| `output.py` | Reconstruct the pixel path, time it, draw it |
| `pipeline.py` | Run every stage on one image |
| `main.py` | Loop over the folder and produce the rankings |

Each file does one job. `pipeline.process()` takes one image path and returns
one dictionary, so it can be called from a script, a notebook or a test
without dragging folder logic along.

---

## Approach

### 1. Colour cleanup

The input images are drawings saved as **JPEG**. That matters more than it
sounds. A drawing of this kind should contain about ten flat colours; the
actual files contain **8,437 distinct colours** in a single 1280×720 image.
JPEG compression approximates colour to save space, so every flat region is
smeared into hundreds of near-identical variants.

Exact colour matching (`img == [255, 0, 0]`) therefore fails almost
completely.

The fix is nearest-colour snapping. A colour is a point in 3D space with axes
R, G and B. For every pixel, measure the distance to each known reference
colour and relabel the pixel as the nearest one. Edge and compression noise
gets absorbed into whichever region it sits closest to, and the image
collapses back to ten clean classes.

Two implementation notes:

- The image is cast to `int32` before subtracting. In `uint8`, `84 - 215`
  wraps around to 125 instead of going negative, silently corrupting every
  distance.
- The square root is skipped. Whichever colour has the smallest squared
  distance also has the smallest true distance, so computing it would be
  wasted work.

### 2. Traversable mask

Black walls and blue regions become 0; all three greens plus casualties and
the two triangle markers become 255. Casualties are traversable because the
rover has to reach them.

### 3. Casualty detection

Blobs are found per colour with `cv2.connectedComponentsWithStats`, which
returns each blob's area, bounding box and centroid in one call. The centroid
is the reported coordinate.

Shape classification uses two measurements rather than one:

- **Solidity** — blob area divided by the area of its convex hull. A circle or
  square fills its hull almost completely (near 1.0); a star has deep notches
  between its points and fills roughly half. This separates stars.
- **Circularity** — `4π × area / perimeter²`. Exactly 1.0 for a perfect
  circle, about 0.785 for a square. This separates the remaining two.

Elevation is read from the terrain *around* a casualty, not under it: the blob
is dilated, the original blob subtracted to leave a ring, and the most common
terrain class in that ring wins.

### 4. Path planning

Dijkstra rather than BFS, because terrain speeds differ and steps are not
equal cost. Two costs are tracked separately:

- **Distance** — plain pixels, used by the score formula
- **Time** — pixels ÷ terrain speed, used by the ranking

Three practical decisions:

- **Obstacles are inflated** by a few pixels before planning, so routes do not
  scrape along walls or clip corners.
- **Planning happens at quarter resolution** (320×180 instead of 1280×720).
  Dijkstra over 921,600 pixels in Python is slow; over 57,000 cells it is
  instant. Coordinates are scaled back afterwards.
- **Diagonal steps cost √2**, not 1. Getting this wrong quietly inflates every
  distance in the project.

One Dijkstra run from a point gives the cost to every other point at once, so
the full distance matrix needs only one run per point of interest.

### 5. Choosing the visiting order

```
Casualty score = (straight-line distance from start / distance travelled) × priority
```

That fraction can never exceed 1, and shrinks as the rover accumulates
distance. So a casualty visited first may be worth 0.93 × its priority, while
the same casualty visited eighth is worth 0.10 × its priority.

Ordering is done on the distance matrix alone — about a dozen points — not on
the image. The search is:

1. **Greedy build** — repeatedly insert the casualty and position that raises
   the total score most.
2. **Local search** — two move types. *Swap* exchanges two positions.
   *Relocate* lifts one stop out and reinserts it elsewhere, leaving the rest
   in sequence. Relocate matters because a swap cannot move a single stop
   without dragging a second one along, so the useful move never gets tried.
3. **Random restarts** — 40 random starting orders, each polished, best kept.

---

## Findings and error analysis

### JPEG compression nearly hid the red casualties

Counting solid colour regions, orange had 2,524 pixels, purple 2,961, white
1,677, yellow 1,243 — and red only **616**. Red had been shattered into
hundreds of one-pixel variants (`[255,0,16]`, `[253,0,21]`, `[250,0,25]`…),
none common enough to appear in a frequency ranking.

This is expected behaviour: JPEG stores colour at lower resolution than
brightness, and red suffers worst under that scheme. It was only found by
bucketing similar colours together and re-counting. Since red is the highest
severity class, missing it would have corrupted every score in the project.

**Consequence:** all outputs are saved as PNG, never JPEG. Saving results as
JPEG would re-introduce exactly the noise the pipeline just removed, and would
fray the mask's black/white boundary into grey.

### Three greens or four?

Colour analysis found four distinct greens, all large enough to be real
regions rather than edge noise:

| RGB | Pixels | Share |
|---|---|---|
| 84, 215, 0 | 673,663 | 68.9% |
| 97, 194, 55 | 74,831 | 6.1% |
| 76, 137, 0 | 20,798 | 1.7% |
| 47, 82, 0 | 14,523 | 1.2% |

But the brief specifies **three** terrain speeds, which leaves no room for a
fourth class. The two are reconciled by perceived brightness: `84,215,0` gives
151 and `97,194,55` gives 149 — visually the same shade. The second also
carries 55 in the blue channel while every other green has ~0, which is the
signature of white blended in, i.e. a pale halo or a compression artefact
around the large high-contrast ellipses, not a separate terrain.

**Decision:** the two are merged into light green. The palette maps each
terrain class to a *list* of raw colours, so a new shade in a future image is
one line to add rather than a logic change.

Terrain levels are ordered by the green channel (215 → 194 → 137 → 82), not by
brightness. Brightness would rank the top two at 151 and 149 and could flip
them arbitrarily between images.

### The brief contradicts its own figures — twice

1. **Blue regions.** The text names only black as non-traversable, but the
   sample mask (page 5) clearly renders the blue ellipses as black blobs. The
   figure is followed: blue is treated as an obstacle.
2. **Elevation levels.** The example table (page 5) shows a casualty at level
   3, which requires four terrains. The speed table allows only three, so the
   deepest level is 2. The speed table is followed, since the example table is
   evidently illustrative — the same table has `x,y` where coordinates belong.

### Skipping casualties is never worth it

An early assumption was that adding a stop could lower the total, since it
inflates the travelled distance for everyone after it. Experiment disproved
this, and the reason is simple:

> Appending a casualty to the **end** of the order leaves every earlier
> casualty's score unchanged — each was already divided by the same travelled
> distance — and adds a strictly positive term. So no casualty is ever worth
> skipping.

The problem is therefore purely one of ordering, not selection. The code was
simplified accordingly: the greedy build now places every casualty, and the
removal move was deleted from the local search.

### Random restarts earned their place on one image out of five

| Image | Greedy | + local search | + restarts |
|---|---|---|---|
| WA0025 | 7.1582 | 7.1582 | 7.1582 |
| WA0026 | 11.6520 | 11.6520 | 11.6520 |
| **WA0027** | **16.2210** | **16.2210** | **16.7705** |
| WA0028 | 16.8932 | 16.8932 | 16.8932 |
| WA0029 | 24.6004 | 24.6661 | 24.6661 |

On four images all three methods agree, which is reasonable evidence the
answer is optimal or near it. On WA0027 both greedy and local search stalled
in the same local optimum, and only restarting from random orders escaped it —
a 3.4% improvement. Convergence of independent methods is used here as the
correctness check, since the true optimum is unknown.

### Straight-line closeness does not make a casualty cheap

On WA0029 the yellow star sits 311 px from the start in a straight line, yet
the optimiser places it **eighth**, earning only 0.20. It is behind an
obstacle: reaching it actually costs 603 px of travel. Spending that early
would divide every subsequent casualty's score.

The formula rewards routes that flow, not routes that grab whatever is
nearest.

---

## Results

| Image | Casualties | Path score | Time (s) | Distance (px) | Avg speed |
|---|---|---|---|---|---|
| WA0025 | 3 | 7.16 | 118.6 | 2,371 | 20.0 px/s |
| WA0026 | 6 | 11.65 | 227.3 | 4,388 | 19.3 px/s |
| WA0027 | 9 | 16.77 | 293.4 | 5,489 | 18.7 px/s |
| WA0028 | 7 | 16.89 | 232.6 | 4,609 | 19.8 px/s |
| WA0029 | 9 | 24.67 | 241.8 | 4,764 | 19.7 px/s |

**Ranking by path score** (highest first):
`[WA0029, WA0028, WA0027, WA0026, WA0025]`

**Ranking by time** (fastest first):
`[WA0025, WA0026, WA0028, WA0029, WA0027]`

Two observations. Score broadly tracks casualty count, but not strictly —
WA0028 has 7 casualties and outscores WA0027's 9, because WA0027's casualties
are spread further apart and the travelled distance climbs faster, collapsing
the later ratios. And every average speed sits just under the light-green
maximum of 20 px/s, confirming Dijkstra is routing around slow terrain rather
than through it; WA0025 at exactly 20.0 never touches dark green at all.

---

## Assumptions

- Blue regions are non-traversable, per the sample figure rather than the text.
- Terrain is flat; slope is ignored and 1 pixel = 1 unit of distance, as
  specified.
- A casualty is covered once the path reaches its centroid.
- Every casualty is visited, for the reason proved above.
- Blobs under 80 px are treated as compression speckle, not shapes.







The first programme 
"""Colour scheme we have extracted:
84, 215, 0	green level 0
97, 194, 55	green level 1
76, 137, 0	green level 2
47, 82, 0	green level 3
93, 23, 235	blue region
0, 0, 0	black obstacle
255, 116, 31	orange triangle — start
203, 107, 230	purple triangle — destination
255, 222, 89	yellow casualty
255, 255, 255	white casualty"""


**Appending a casualty to the end of the order leaves every earlier casualty's score unchanged and adds a strictly positive term. So no casualty is ever worth skipping, and the problem reduces to choosing an order.
**



IMG-20260831-WA0025.jpg
  greedy          7.1582
  after polish    7.1582
  random restarts 7.1582
  3 casualties | score 7.16 | time 118.6s | distance 2371px

IMG-20260831-WA0026.jpg
  greedy          11.6520
  after polish    11.6520
  random restarts 11.6520
  6 casualties | score 11.65 | time 227.3s | distance 4388px

IMG-20260831-WA0027.jpg
  greedy          16.2210
  after polish    16.2210
  random restarts 16.7705
  9 casualties | score 16.77 | time 293.4s | distance 5489px

IMG-20260831-WA0028.jpg
  greedy          16.8932
  after polish    16.8932
  random restarts 16.8932
  7 casualties | score 16.89 | time 232.6s | distance 4609px

IMG-20260831-WA0029.jpg
  greedy          24.6004
  after polish    24.6661
  random restarts 24.6661
  9 casualties | score 24.67 | time 241.8s | distance 4764px

==========================================================
RANKING BY PATH SCORE (highest first)
  1. IMG-20260831-WA0029.jpg             24.67
  2. IMG-20260831-WA0028.jpg             16.89
  3. IMG-20260831-WA0027.jpg             16.77
  4. IMG-20260831-WA0026.jpg             11.65
  5. IMG-20260831-WA0025.jpg              7.16

RANKING BY TIME (fastest first)
  1. IMG-20260831-WA0025.jpg             118.6s
  2. IMG-20260831-WA0026.jpg             227.3s
  3. IMG-20260831-WA0028.jpg             232.6s
  4. IMG-20260831-WA0029.jpg             241.8s
  5. IMG-20260831-WA0027.jpg             293.4s
==========================================================

processed 5 of 5 images
