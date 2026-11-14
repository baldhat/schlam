# Monocular SLAM evaluation

`SchlamValidator().validate(system.run())` evaluates the online camera trajectory
without reopening a dataset, using the ground truth captured in `SchlamResult`.
The application prints the resulting RMSEs and coverage automatically.

## Method and rationale

Monocular SLAM has an unknown global coordinate frame and scale. The default
evaluation associates timestamps, fits **one Sim(3) transform to the whole matched
trajectory**, and measures camera position error after alignment. This follows
the [evo metric and alignment documentation](https://github.com/MichaelGrupp/evo/wiki/Metrics).
Repeated alignment of individual segments would hide drift and is not used.

* **Absolute trajectory error (ATE)** measures global trajectory consistency:
  `||p_gt_i - (s R p_est_i + t)||`. RMSE is the primary score.
* **Relative pose error (RPE)** measures local motion error. For camera-to-world
  poses, `E_ij = (T_gt_i^-1 T_gt_j)^-1 (T_est_i^-1 T_est_j)`.
  Translation error is `||trans(E_ij)||`; rotation error is the SO(3) angle in
  radians. Estimated translations receive the same global scale fitted for ATE.
  RPE uses all overlapping pairs separated by `relativePoseDelta` estimated poses
  (one by default). It is not divided by distance or elapsed time.
* Each metric returns count, RMSE, mean, median, population standard deviation,
  minimum and maximum. Translation units follow ground truth: metres for the
  supported EuRoC MAV and KITTI loaders.
* Association coverage is matched poses / estimated poses. Tracking coverage is
  estimated poses / processed frames, and is absent when frame count is unknown.
  FPS and final tracking state are carried into the report.

The [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
instead aggregates drift over 100–800 metre segments, reporting translation in
percent and rotation in degrees/metre. That is valuable for driving sequences,
but ATE plus frame-interval RPE also works for shorter indoor MAV sequences.
This validator does not claim to reproduce the official KITTI score.

## Data and association

Both paths contain camera-to-world rotations and camera centres. Ground truth is
converted from body poses with the camera-to-body calibration. Every consumed GT
sample is retained, including samples skipped during synchronization and frames
without an accepted prediction. GT retains its sensor timestamp; it is never
relabelled with an image timestamp.

The MAV loader normalizes CSV ground-truth quaternions before converting them
to rotation matrices. Quaternion norm drift would otherwise introduce scale/shear
into body poses and their composed camera poses, affecting tracking and evaluation.

Association visits predictions in time order and selects the nearest available
reference timestamp within `maximumTimestampDifference` (inclusive, default
20 ms). References cannot be reused, matches preserve chronological order, and
ties select the earlier sample. This is a greedy association, not interpolation
or a globally optimal assignment. Integer timestamp differences preserve precision
at epoch-scale timestamps. Paths must have strictly increasing timestamps.
Unmatched predictions never collapse a requested RPE interval: if either endpoint
of an index pair lacks ground truth, that pair is skipped.

## Configuration and failures

```cpp
#include "schlam/SchlamValidator.h"

SchlamValidator::Options options;
options.alignment = SchlamValidator::Alignment::SE3; // Keep metric scale.
options.maximumTimestampDifference = std::chrono::milliseconds(5);
options.relativePoseDelta = 10;
auto report = SchlamValidator(options).validate(result);
if (report.status == SchlamValidator::Status::SUCCESS) {
    const double ate = report.absoluteTranslation->rmse;
}
```

`SIM3` is the default; `SE3` fits rotation and translation with scale fixed to one;
`NONE` evaluates the existing coordinate frame directly. Alignment needs at least
three associated, non-collinear positions. Stationary/collinear motion returns
`DEGENERATE_ALIGNMENT`; planar motion is valid. `NONE` can evaluate a single pose.
Missing data or too few matches returns `INSUFFICIENT_MATCHES`. Failed evaluations
have no error statistics. A successful ATE evaluation may have no RPE if no pair
meets the requested interval. Rotations with Frobenius orthogonality error
`||R^T R - I|| <= 1e-4` and determinant error `|det(R) - 1| <= 1e-4` are projected
onto SO(3) in a private copy. This accommodates rounding in dataset calibration
without modifying the input. Reflections and larger rotation defects are rejected.
Invalid options, nonfinite poses, invalid rotations, or duplicate/out-of-order
timestamps throw `std::invalid_argument`; pose errors identify the path, index
and timestamp. The application reports validation errors and exits unsuccessfully
after viewer cleanup, without aborting on an uncaught validation exception.

## Interpretation in Schlam

Schlam currently seeds its initial world pose and baseline scale from ground
truth. Its results therefore describe **GT-initialized monocular tracking**.
Sim(3) ATE measures trajectory shape; SE(3) also exposes scale error after this
initialization. Neither score establishes independent recovery of metric scale.

`predictedPath` contains accepted online estimates, not a reconstructed trajectory
after final bundle adjustment. Map points are not evaluated because no reference
map is supplied. Ground truth covers only the consumed part of the dataset, and
tracking coverage uses only processed frames. In particular, stopping on `LOST`
does not evaluate the unread remainder of a sequence. Compare coverage, final
tracking state, dataset percentage read, FPS, and errors together. Use the same
timestamp tolerance, alignment and RPE interval across runs; frame-based RPE
changes its temporal meaning when accepted estimates have different spacing.

Synthetic unit tests cover known similarity transforms, metric scale error,
translation and rotation drift, timestamp tolerance and missing matches,
reflections and degenerate alignment. System tests cover capture of camera-frame
ground truth, its original timestamps, frame counts and tracking state.
