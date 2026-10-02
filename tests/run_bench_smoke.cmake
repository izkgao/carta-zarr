# Runs carta-zarr-bench as tools/zarr-bench/sweep.py will: probe a dataset, run every mode in two
# processes, and resume. tests/bench_test.cc covers what it computes; this covers that the executable
# forks, releases, collects and writes it -- one row per operation, a trial at a time, never twice.

if(NOT DEFINED BENCH OR NOT DEFINED FIXTURE OR NOT DEFINED OUTPUT_DIR)
    message(FATAL_ERROR "BENCH, FIXTURE and OUTPUT_DIR must be set")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(csv "${OUTPUT_DIR}/bench.csv")

execute_process(COMMAND "${BENCH}" probe "${FIXTURE}" RESULT_VARIABLE result OUTPUT_VARIABLE report)
if(result)
    message(FATAL_ERROR "probe failed on the fixture: ${report}")
endif()
string(JSON ok GET "${report}" ok)
string(JSON sky GET "${report}" image id)
if(NOT ok OR NOT sky STREQUAL "SKY")
    message(FATAL_ERROR "probe did not report the fixture's image: ${report}")
endif()

execute_process(COMMAND "${BENCH}" probe "${OUTPUT_DIR}/missing" RESULT_VARIABLE result OUTPUT_VARIABLE report)
if(NOT result)
    message(FATAL_ERROR "probe succeeded on a dataset that does not exist: ${report}")
endif()

# Rows in one trial of every mode for two processes: plane 16, animation 2, spectrum 32, region 1,
# cube-histogram 1 and open 8 operations each.
math(EXPR per_trial "2 * (16 + 2 + 32 + 1 + 1 + 8)")

function(run_and_count trials expected)
    execute_process(
        COMMAND "${BENCH}" run "${FIXTURE}" --processes 2 --trials ${trials} --cold off --csv "${csv}"
                --animation-fps 200 ${ARGN}
        RESULT_VARIABLE result
        ERROR_VARIABLE log)
    if(result)
        message(FATAL_ERROR "run failed (${result}):\n${log}")
    endif()
    # A position is semicolon-separated, which a CMake list would split, so the lines are cut by hand.
    file(READ "${csv}" content)
    string(REPLACE ";" "|" content "${content}")
    string(REGEX REPLACE "\n$" "" content "${content}")
    string(REPLACE "\n" ";" lines "${content}")
    list(LENGTH lines count)
    if(NOT count EQUAL expected)
        message(FATAL_ERROR "expected ${expected} lines in the CSV after ${trials} trials, found ${count}:\n${log}")
    endif()
    list(POP_FRONT lines header)
    foreach(line IN LISTS lines)
        if(NOT line MATCHES ",ok,")
            message(FATAL_ERROR "an operation did not succeed: ${line}")
        endif()
    endforeach()
endfunction()

math(EXPR two "1 + 2 * ${per_trial}")
math(EXPR three "1 + 3 * ${per_trial}")
run_and_count(2 ${two})
# Resuming runs only the trial that is not there yet, and then nothing at all.
run_and_count(3 ${three} --resume)
run_and_count(3 ${three} --resume)
