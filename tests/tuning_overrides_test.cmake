# cmake/TuningOverrides.cmake: which overrides a configure accepts, the definitions they become, and
# the description carta-zarr-bench writes into every row of a build made with them.

# A script sets no policies of its own, and before CMake 4 IN_LIST needs one.
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR must be set")
endif()
include("${SOURCE_DIR}/cmake/TuningOverrides.cmake")

carta_zarr_tuning_definitions("" definitions description)
if(definitions OR NOT description STREQUAL "default")
    message(FATAL_ERROR "no overrides gave definitions \"${definitions}\" and description \"${description}\"")
endif()

carta_zarr_tuning_definitions("LEAST_PIXELS_PER_TASK=16384;CUBE_ACCUMULATOR_CACHE_BYTES=4194304"
                              definitions description)
if(NOT definitions STREQUAL
   "CARTA_ZARR_TUNING_LEAST_PIXELS_PER_TASK=16384ull;CARTA_ZARR_TUNING_CUBE_ACCUMULATOR_CACHE_BYTES=4194304ull")
    message(FATAL_ERROR "two overrides gave the definitions \"${definitions}\"")
endif()
# Sorted, so that the same build is described the same way whatever order it was asked for in.
if(NOT description STREQUAL "CUBE_ACCUMULATOR_CACHE_BYTES=4194304,LEAST_PIXELS_PER_TASK=16384")
    message(FATAL_ERROR "two overrides were described as \"${description}\"")
endif()

# Each of these must stop a configure. A refusal is a FATAL_ERROR, which ends a script, so each is
# tried in a script of its own.
foreach(bad "PROVISIONAL_BINS_PER_BIN=8" "LEAST_PIXELS_PER_TASK=0" "LEAST_PIXELS_PER_TASK=-1"
        "LEAST_PIXELS_PER_TASK=1.5" "LEAST_PIXELS_PER_TASK" "LEAST_PIXELS_PER_TASK=1|LEAST_PIXELS_PER_TASK=2")
    # "|" separates the entries of one list, which set() in the script makes a list of.
    string(REPLACE "|" "\" \"" entries "${bad}")
    set(script "${CMAKE_CURRENT_BINARY_DIR}/tuning_overrides_refused.cmake")
    file(WRITE "${script}" "cmake_minimum_required(VERSION 3.24)\n"
                           "include(\"${SOURCE_DIR}/cmake/TuningOverrides.cmake\")\n"
                           "set(overrides \"${entries}\")\n"
                           "carta_zarr_tuning_definitions(\"\${overrides}\" d s)\n")
    execute_process(COMMAND "${CMAKE_COMMAND}" -P "${script}" RESULT_VARIABLE result OUTPUT_QUIET
                    ERROR_VARIABLE refusal)
    file(REMOVE "${script}")
    if(result EQUAL 0)
        message(FATAL_ERROR "the override \"${bad}\" was accepted")
    endif()
    if(bad MATCHES "\\|" AND NOT refusal MATCHES "given twice")
        message(FATAL_ERROR "a name given twice was refused for something else:\n${refusal}")
    endif()
endforeach()
