#[[
  carta-zarr-bench, which measures how fast this library reads a dataset the way CARTA reads it.

  It links the installed interface only -- CARTA::zarr and the public headers -- so that it measures
  what carta-backend linked against the same build would. Everything but main.cc is a library of its
  own so that the tests can reach the planning and the CSV without running the executable.

  The commit it reports is the one at configure time: reconfigure before measuring a change.
]]

set(CARTA_ZARR_BENCH_COMMIT "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE commit
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE result)
    if(result EQUAL 0)
        set(CARTA_ZARR_BENCH_COMMIT "${commit}")
    endif()
endif()

add_library(carta_zarr_bench_core STATIC
    cold.cc
    options.cc
    record.cc
    trial.cc
    workload.cc)
target_compile_features(carta_zarr_bench_core PUBLIC cxx_std_17)
target_include_directories(carta_zarr_bench_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(carta_zarr_bench_core
    PUBLIC CARTA::zarr
    PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(carta_zarr_bench_core PUBLIC
    CARTA_ZARR_BENCH_COMMIT="${CARTA_ZARR_BENCH_COMMIT}"
    CARTA_ZARR_BENCH_BUILD_TYPE="$<IF:$<STREQUAL:$<CONFIG>,>,none,$<CONFIG>>"
    # The library's tuning overrides, which the bench cannot ask the library for: they are internal,
    # and the bench reads it through the public interface alone. Right as long as the bench runs
    # against the library it was built with, which its RPATH sees to once installed.
    CARTA_ZARR_BENCH_TUNING="${CARTA_ZARR_TUNING}")

add_executable(carta_zarr_bench main.cc)
target_link_libraries(carta_zarr_bench PRIVATE carta_zarr_bench_core nlohmann_json::nlohmann_json)
if(APPLE)
    set(origin "@loader_path")
else()
    set(origin "$ORIGIN")
endif()
set_target_properties(carta_zarr_bench PROPERTIES
    OUTPUT_NAME carta-zarr-bench
    INSTALL_RPATH "${origin}/../${CMAKE_INSTALL_LIBDIR}")

install(TARGETS carta_zarr_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
