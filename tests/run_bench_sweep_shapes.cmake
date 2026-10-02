# Runs tools/zarr-bench/sweep.py over two synthetic shapes: a sweep each in a directory of its own, and
# a summary across them that says what one layout and one setting would serve both. As with
# run_bench_sweep.cmake, timings are not checked, only that every part is there.

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

# Two layouts, one of them twice: as it is and in shards of two of its chunks along the spectrum, said
# relative to the chunk. "wide" has four channels, too few for the eight-deep chunk, which it must
# skip, and stands for a cube of sixteen, which the report must scale to.
file(WRITE "${config}" "
[paths]
bench = \"${BENCH}\"
work = \"${OUTPUT_DIR}/work\"
[source]
seed = 3
[[source.shape]]
name = \"small\"
synthetic = \"frequency=8,polarization=1,l=64,m=64\"
[[source.shape]]
name = \"wide\"
synthetic = \"frequency=4,polarization=1,l=128,m=128\"
channels = 16
[users]
target = 1
[measure]
trials = 1
cold = \"off\"
animation_fps = 100
generator_workers = 2
ops = { plane = 2, spectrum = 4 }
[stage1]
chunk = [\"l=32,m=32,frequency=2\", \"l=64,m=64,frequency=8\"]
shard = [\"\", \"frequency*2\"]
[stage2]
top = 1
io_threads = [2, 4]
[checks]
ram = false
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

sweep_run(--dry-run)
if(sweep_result OR EXISTS "${results}" OR EXISTS "${OUTPUT_DIR}/work")
    message(FATAL_ERROR "the dry run failed or wrote something (${sweep_result}):\n${sweep_log}")
endif()
foreach(expected "== shape small" "== shape wide" "shard frequency=4"
        "chunk frequency=8 is larger than the cube's 4")
    string(FIND "${sweep_log}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "the dry run did not say \"${expected}\":\n${sweep_log}")
    endif()
endforeach()

sweep_run()
if(sweep_result)
    message(FATAL_ERROR "the sweep failed (${sweep_result}):\n${sweep_log}")
endif()
foreach(shape small wide)
    if(NOT EXISTS "${results}/${shape}/summary.md" OR NOT EXISTS "${results}/${shape}/results.csv")
        message(FATAL_ERROR "shape ${shape} has no report of its own:\n${sweep_log}")
    endif()
endforeach()

file(READ "${results}/summary.md" summary)
foreach(section "## Each shape" "[small](small/summary.md)" "## One layout for every shape"
        "**One layout for every shape:**" "## One setting for every shape" "--zarr_file_io_threads"
        "## At full depth" "| wide |" "## Configuration" "[[source.shape]]")
    string(FIND "${summary}" "${section}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "the summary across shapes has no \"${section}\":\n${summary}")
    endif()
endforeach()

# Again: every run of both shapes is done, so nothing is written.
sweep_run()
if(sweep_result OR sweep_log MATCHES "generating")
    message(FATAL_ERROR "resuming a finished sweep over shapes did more work (${sweep_result}):\n${sweep_log}")
endif()
