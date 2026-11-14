#!/usr/bin/env bash
set -euo pipefail

# Run from schlam/cpp after activating the Conan runtime environment.
build_dir="${SCHLAM_BUILD_DIR:-build/Release}"
output_dir="${SCHLAM_VALIDATION_DIR:-build/validation}"
mkdir -p "$output_dir"
summary="${GITHUB_STEP_SUMMARY:-$output_dir/summary.md}"
printf '## Release trajectory validation\n\n' >> "$summary"

failed=0
for config in config/mav_headless.yaml config/kitti_headless.yaml; do
    name="$(basename "$config" .yaml)"
    log="$output_dir/$name.log"
    cp "$config" "$output_dir/$name.yaml"

    # The conditional retains pipefail's exit code and allows the next dataset
    # to run even after an application error, timeout, or failed validation.
    if timeout --kill-after=30s "${SCHLAM_HEADLESS_TIMEOUT:-30m}" \
        "$build_dir/app" "$config" 2>&1 | tee "$log"; then
        run_exit=0
    else
        run_exit=$?
    fi

    outcome=passed
    if (( run_exit != 0 )) || ! awk '
        /^Trajectory validation: Trajectory evaluation succeeded \(/ { found = 1 }
        END { exit !found }
    ' "$log"; then
        outcome=failed
        failed=1
        printf '::error::%s failed or produced no successful trajectory validation (exit code %s).\n' "$name" "$run_exit"
    fi

    {
        printf '### %s\n\n' "$name"
        printf 'Configuration: `%s`\n\n' "$config"
        printf 'Result: **%s**. Process exit code: %s.\n\n' "$outcome" "$run_exit"
        printf '```text\n'
        awk '
            /^(Finished with |Created |Trajectory validation[: ]|Tracking coverage |Sim\(3\) alignment scale: |RPE RMSE: )/ {
                print
                found = 1
            }
            END { if (!found) print "No validation output was produced. See the run log." }
        ' "$log"
        printf '```\n\n'
    } >> "$summary"
done

exit "$failed"
