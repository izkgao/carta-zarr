# Replacements for the machine-dependent constants of src/reduce/tuning.h, for measuring them.
#
# CARTA_ZARR_TUNING_OVERRIDES is a list of NAME=VALUE, NAME one of the constants below without its
# CARTA_ZARR_TUNING_ prefix and VALUE a positive whole number:
#
#   -DCARTA_ZARR_TUNING_OVERRIDES="CUBE_ACCUMULATOR_CACHE_BYTES=4194304;LEAST_PIXELS_PER_TASK=16384"
#
# A build with any is a measurement: carta-zarr-bench writes the list into every row, and nothing
# else should be built from it. See ADR 0014.

set(CARTA_ZARR_TUNABLE_CONSTANTS
    LEAST_PIXELS_PER_TASK
    SPECTRAL_PARTIAL_BUDGET_BYTES
    HISTOGRAM_PARTIAL_BUDGET_BYTES
    CUBE_ACCUMULATOR_CACHE_BYTES)

#[[
  carta_zarr_tuning_definitions(<overrides> <definitions-var> <description-var>)

  Checks a list of overrides and sets <definitions-var> to the compile definitions that apply them,
  and <description-var> to the list as carta-zarr-bench records it: sorted by name, so that the same
  overrides given in another order describe the same build, and "default" when there are none. A
  name that is not one of CARTA_ZARR_TUNABLE_CONSTANTS, a value that is not a positive whole number,
  or a name given twice stops the configure.
]]
function(carta_zarr_tuning_definitions overrides definitions_var description_var)
    set(definitions)
    set(seen)
    set(described)
    foreach(entry IN LISTS overrides)
        if(NOT entry MATCHES "^([A-Z_]+)=([0-9]+)$")
            message(FATAL_ERROR "CARTA_ZARR_TUNING_OVERRIDES: \"${entry}\" is not NAME=VALUE with a whole number")
        endif()
        set(name "${CMAKE_MATCH_1}")
        set(value "${CMAKE_MATCH_2}")
        if(NOT name IN_LIST CARTA_ZARR_TUNABLE_CONSTANTS)
            string(REPLACE ";" ", " known "${CARTA_ZARR_TUNABLE_CONSTANTS}")
            message(FATAL_ERROR "CARTA_ZARR_TUNING_OVERRIDES: ${name} is not one of ${known}")
        endif()
        if(name IN_LIST seen)
            message(FATAL_ERROR "CARTA_ZARR_TUNING_OVERRIDES: ${name} is given twice")
        endif()
        if(value MATCHES "^0+$")
            message(FATAL_ERROR "CARTA_ZARR_TUNING_OVERRIDES: ${name} must be positive")
        endif()
        list(APPEND seen "${name}")
        # As unsigned long long, so that a value past 2^32 means what it says on every platform.
        list(APPEND definitions "CARTA_ZARR_TUNING_${name}=${value}ull")
        list(APPEND described "${name}=${value}")
    endforeach()
    list(SORT described)
    if(NOT described)
        set(described "default")
    endif()
    string(REPLACE ";" "," described "${described}")
    set(${definitions_var} "${definitions}" PARENT_SCOPE)
    set(${description_var} "${described}" PARENT_SCOPE)
endfunction()
