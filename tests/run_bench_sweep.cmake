# Runs tools/zarr-bench/sweep.py end to end on a committed fixture: every stage, then the same sweep
# again, which must find everything done, and a dry run, which must write nothing. Timings are not
# checked -- the fixture is far too small to mean anything -- only that the pipeline runs, writes a
# row for every operation without an error, and reports in every section.

foreach(variable UV BENCH SOURCE_DIR OUTPUT_DIR)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "${variable} must be set")
    endif()
endforeach()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(sweep "${SOURCE_DIR}/tools/zarr-bench/sweep.py")
set(results "${OUTPUT_DIR}/results")
set(config "${OUTPUT_DIR}/sweep.toml")

# The fixture is 512 x 520 x 4 channels x 2 polarizations. Two grid layouts and the fixture's own,
# one that is refused, two settings in stage 2, and validation over all four channels. The RAM check
# is off, since nothing here is larger than RAM, and with it validation runs although caches stay warm.
# Two channels are far too few for 16 planes a user to each read chunks of their own, so the report
# must say so: that warning is checked for below. The warm stage compares the bench with itself, as a
# variant that overrides nothing, which the report must also notice.
file(WRITE "${config}" "
[paths]
bench = \"${BENCH}\"
work = \"${OUTPUT_DIR}/work\"
[source]
path = \"${SOURCE_DIR}/tests/data/images/zarr/xradio/pixels_wide\"
crop = \"frequency=0:2\"
validate_crop = \"frequency=0:4\"
[users]
target = 2
typical = 1
[measure]
trials = 1
cold = \"off\"
animation_fps = 100
histogram_reference = [\"binned\"]
generator_workers = 2
[stage1]
chunk = [\"l=128,m=128,frequency=1\", \"l=512,m=520,frequency=2\"]
[[stage1.layout]]
name = \"refused\"
chunk = \"l=128,m=128\"
shard = \"l=200\"
[stage2]
top = 1
io_threads = [2, 4]
[checks]
ram = false
[warm]
variants = [{ name = \"same\", bench = \"${BENCH}\" }]
trials = 1
")

function(sweep_run)
    execute_process(
        COMMAND "${UV}" run --quiet --script "${sweep}" "${config}" --output "${results}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE out
        ERROR_VARIABLE log)
    set(sweep_result "${result}" PARENT_SCOPE)
    set(sweep_log "${out}${log}" PARENT_SCOPE)
endfunction()

# A dry run says what would happen and writes nothing at all.
sweep_run(--dry-run)
if(sweep_result OR EXISTS "${results}" OR EXISTS "${OUTPUT_DIR}/work")
    message(FATAL_ERROR "the dry run failed or wrote something (${sweep_result}):\n${sweep_log}")
endif()
if(NOT sweep_log MATCHES "skipped refused")
    message(FATAL_ERROR "the dry run did not list the refused layout:\n${sweep_log}")
endif()

sweep_run()
if(sweep_result)
    message(FATAL_ERROR "the sweep failed (${sweep_result}):\n${sweep_log}")
endif()

file(READ "${results}/results.csv" rows)
foreach(stage stage1 stage2 confirm validate warm-default warm-same)
    if(NOT rows MATCHES ",${stage},")
        message(FATAL_ERROR "results.csv has no ${stage} rows:\n${sweep_log}")
    endif()
endforeach()
# The header names the status and error columns, so it is left out of the search.
string(FIND "${rows}" "\n" header_end)
math(EXPR body_start "${header_end} + 1")
string(SUBSTRING "${rows}" ${body_start} -1 body)
if(body MATCHES "[^\n]*,(error|timeout),[^\n]*")
    message(FATAL_ERROR "an operation did not succeed:\n${CMAKE_MATCH_0}")
endif()

file(READ "${results}/summary.md" summary)
foreach(section "## Conclusion" "## Machine and environment" "## Stage 1: layouts" "## Stage 2: reader settings"
        "## Sensitivity to the number of users" "## Validation" "## Reference: one-pass cube histograms"
        "## Configuration" "--zarr_file_io_threads" "Skipped layout refused" "| binned |"
        "### animation" "### Plane against spectrum" "animation per frame" "Too few first touches for plane"
        "first touches (2)" "## Warm stage: tuning constants" "Tuning variant same has no overrides")
    string(FIND "${summary}" "${section}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "summary.md has no \"${section}\":\n${summary}")
    endif()
endforeach()
if(summary MATCHES "Not run yet|No recommendation yet|Skipped: ")
    message(FATAL_ERROR "a stage of the sweep did not run:\n${summary}")
endif()

# The same sweep again finds every run done: nothing is written and no row is added.
string(LENGTH "${rows}" before)
sweep_run()
file(READ "${results}/results.csv" rows)
string(LENGTH "${rows}" after)
if(sweep_result OR NOT before EQUAL after OR sweep_log MATCHES "generating")
    message(FATAL_ERROR "resuming a finished sweep did more work (${sweep_result}):\n${sweep_log}")
endif()
