cmake_minimum_required(VERSION 3.24)

project(carta_zarr VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DOCDIR "${CMAKE_INSTALL_DATADIR}/doc/${PROJECT_NAME}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CARTA_ZARR_BUILD_TESTS "Build carta-zarr tests" ON)
option(CARTA_ZARR_BUILD_PACKAGE_TEST "Build the install-tree CMake consumer test" ON)
option(CARTA_ZARR_BUILD_BENCH "Build the storage benchmark tools and their tests" OFF)
set(CARTA_ZARR_TUNING_OVERRIDES "" CACHE STRING
    "For measuring only: NAME=VALUE replacements for the machine-dependent constants of src/reduce/tuning.h (ADR 0014)")

# TensorStore is a required private dependency: it is the storage backend of this library,
# not an optional feature. Building without it would produce a library that cannot read
# pixel data, so there is no supported configuration that omits it.
include(FetchContent)

# Prefer the copies supplied by TensorStore so that TensorStore and carta-zarr always use the
# same dependency versions and build configuration.
set(TENSORSTORE_USE_SYSTEM_BLOSC OFF CACHE BOOL "" FORCE)
set(TENSORSTORE_USE_SYSTEM_NLOHMANN_JSON OFF CACHE BOOL "" FORCE)
set(TENSORSTORE_USE_SYSTEM_PROTOBUF OFF CACHE BOOL "" FORCE)
set(TENSORSTORE_USE_SYSTEM_ZLIB OFF CACHE BOOL "" FORCE)
set(TENSORSTORE_USE_SYSTEM_ZSTD OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    tensorstore
    URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.84.tar.gz"
    URL_HASH SHA256=d86fe9dca4b69e5c8b488562351dc654bce1400f5981f922e4be9ba4fafc95ce)
FetchContent_MakeAvailable(tensorstore)

# The custom fixed_length_utf32 decoder needs the same compression libraries as the Zarr v3 codec
# chains that XRADIO emits. These targets are supplied by TensorStore's bundled dependencies.
# nlohmann/json is used only by the implementation and is intentionally discovered after
# TensorStore so that TensorStore's bundled package redirect is used.
find_package(nlohmann_json 3.11 CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Zstd REQUIRED)
if(NOT TARGET Blosc::blosc)
    message(FATAL_ERROR "TensorStore did not provide the Blosc::blosc target")
endif()

add_library(carta_zarr SHARED
    src/carta_zarr.cc
    src/dataset_size.cc
    src/read/pieces.cc
    src/read/pieces_read.cc
    src/reduce/occupancy.cc
    src/reduce/region_runs.cc
    src/reduce/pass.cc
    src/reduce/pass_read.cc
    src/reduce/plane_histogram.cc
    src/reduce/spectral_reduce.cc
    src/schema/profile.cc
    src/schema/xradio/probe_report.cc
    src/schema/xradio/beam_table.cc
    src/schema/xradio/direction.cc
    src/schema/xradio/flag.cc
    src/schema/xradio/linear_axis.cc
    src/schema/xradio/observation.cc
    src/schema/xradio/qualification.cc
    src/schema/xradio/image.cc
    src/zarr/array_metadata.cc
    src/zarr/array_view.cc
    src/zarr/pixel_reader.cc
    src/zarr/pixel_selection.cc
    src/zarr/store_context.cc
    src/zarr/string_array.cc
    src/zarr/transport.cc
    src/zarr/value_reader.cc
    src/store.cc
    src/work_pool.cc)
add_library(CARTA::zarr ALIAS carta_zarr)

target_compile_features(carta_zarr PUBLIC cxx_std_17)
target_compile_definitions(carta_zarr PRIVATE CARTA_ZARR_BUILDING_LIBRARY)
include(cmake/TuningOverrides.cmake)
carta_zarr_tuning_definitions("${CARTA_ZARR_TUNING_OVERRIDES}" CARTA_ZARR_TUNING_DEFINITIONS CARTA_ZARR_TUNING)
if(CARTA_ZARR_TUNING_DEFINITIONS)
    message(WARNING "carta-zarr is built with tuning overrides (${CARTA_ZARR_TUNING}), which are for "
                    "measuring them: do not deploy this build")
    target_compile_definitions(carta_zarr PRIVATE ${CARTA_ZARR_TUNING_DEFINITIONS})
endif()
target_include_directories(carta_zarr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
# The library starts its own worker threads now (WorkPool), so it asks for the threading
# support it uses rather than inheriting it from TensorStore's link interface.
find_package(Threads REQUIRED)
target_link_libraries(carta_zarr PRIVATE
    Threads::Threads
    nlohmann_json::nlohmann_json
    ZLIB::ZLIB
    Zstd::Zstd
    Blosc::blosc)

# Keep all TensorStore targets private. Public consumers link only CARTA::zarr and do not
# need TensorStore headers or CMake package metadata.
target_link_libraries(carta_zarr PRIVATE
    tensorstore::tensorstore
    tensorstore::driver_zarr3
    tensorstore::kvstore_file
    tensorstore::index_space_dim_expression
    tensorstore::driver_cast)

set_target_properties(carta_zarr PROPERTIES
    OUTPUT_NAME carta-zarr
    EXPORT_NAME zarr
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})

install(TARGETS carta_zarr
    EXPORT CartaZarrTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CartaZarrConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CartaZarrConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/CartaZarrConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CartaZarr")

install(EXPORT CartaZarrTargets
    FILE CartaZarrTargets.cmake
    NAMESPACE CARTA::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CartaZarr")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CartaZarrConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CartaZarrConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CartaZarr")

if(CARTA_ZARR_BUILD_BENCH)
    add_subdirectory(bench)
endif()

include(CTest)
if(CARTA_ZARR_BUILD_TESTS AND BUILD_TESTING)
    find_package(Threads REQUIRED)

    set(CARTA_ZARR_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/images/zarr")

    # The two fixture sets more than one test needs. Named here so that a regenerated fixture moves
    # in one place rather than in whichever targets happened to reference it.
    set(CARTA_ZARR_PIXEL_FIXTURES
        CARTA_ZARR_PIXEL_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels"
        CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels_l_fastest")
    set(CARTA_ZARR_WIDE_FIXTURE
        CARTA_ZARR_PIXEL_FIXTURE_WIDE="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels_wide")

    # Implementation groups a test compiles directly. Each is the set of sources that one internal
    # thing needs to stand up on its own, so a test says what it is testing rather than how the
    # dependency graph happens to be shaped this week.
    set(CARTA_ZARR_METADATA_SOURCES src/zarr/array_metadata.cc)
    set(CARTA_ZARR_PASS_SOURCES
        src/reduce/pass.cc
        src/reduce/pass_read.cc
        src/zarr/pixel_selection.cc)

    #[[
      Register one test executable.

      There are two kinds here and the difference is load-bearing, so it is a keyword rather than
      something inferred. A PUBLIC test links the shared library and can reach only what a consumer
      can reach. An INTERNAL test compiles implementation sources directly, because what it pins is
      not reachable from outside: hidden visibility, or a build deliberately assembled without
      TensorStore in it. An internal test gets tests/ on its include path too, for the support
      headers those builds share.

      TEST_NAME is optional. Leaving it out builds the target without registering it with ctest,
      which is what a timing harness with no pass or fail wants.
    ]]
    function(carta_zarr_add_test name)
        cmake_parse_arguments(ARG "PUBLIC;INTERNAL" "TEST_NAME" "SOURCES;DEFINITIONS;LIBRARIES" ${ARGN})
        if(ARG_PUBLIC AND ARG_INTERNAL)
            message(FATAL_ERROR "${name}: a test either links the library or compiles it, not both")
        endif()
        if(NOT ARG_PUBLIC AND NOT ARG_INTERNAL)
            message(FATAL_ERROR "${name}: say PUBLIC or INTERNAL")
        endif()

        add_executable(${name} ${ARG_SOURCES})
        target_compile_features(${name} PRIVATE cxx_std_17)
        # Every test reaches tests/support, whichever way it reaches the library: that is where the
        # assertion they are all written with lives. A PUBLIC test gets nothing else of the tree --
        # the point of it is that it sees only what a consumer sees.
        target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        if(ARG_PUBLIC)
            target_link_libraries(${name} PRIVATE CARTA::zarr)
        else()
            target_include_directories(${name} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/src)
        endif()
        if(ARG_LIBRARIES)
            target_link_libraries(${name} PRIVATE ${ARG_LIBRARIES})
        endif()
        if(ARG_DEFINITIONS)
            target_compile_definitions(${name} PRIVATE ${ARG_DEFINITIONS})
        endif()
        if(ARG_TEST_NAME)
            add_test(NAME ${ARG_TEST_NAME} COMMAND ${name})
        endif()
    endfunction()

    carta_zarr_add_test(carta_zarr_tests PUBLIC
        TEST_NAME carta-zarr-schema-probe
        SOURCES tests/schema_probe_test.cc
        DEFINITIONS
            CARTA_ZARR_REFERENCE_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/minimal"
            CARTA_ZARR_LEGACY_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/legacy")

    carta_zarr_add_test(carta_zarr_result_tests PUBLIC
        TEST_NAME carta-zarr-result
        SOURCES tests/result_test.cc)

    # The string decoder is an internal component with hidden visibility, so this test compiles it
    # directly instead of linking the shared library.
    carta_zarr_add_test(carta_zarr_string_tests INTERNAL
        TEST_NAME carta-zarr-string-array
        SOURCES tests/string_array_test.cc src/zarr/string_array.cc ${CARTA_ZARR_METADATA_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json ZLIB::ZLIB Zstd::Zstd Blosc::blosc
        DEFINITIONS CARTA_ZARR_STRING_FIXTURE_DIR="${CARTA_ZARR_FIXTURE_DIR}/string")

    # The schema profile's structural decisions read metadata only, so they run against an in-memory
    # transport instead of a directory tree. This target compiles the internals directly and links no
    # TensorStore: the value reader is stubbed, which is also the honest behaviour of a transport
    # holding no array data.
    carta_zarr_add_test(carta_zarr_profile_tests INTERNAL
        TEST_NAME carta-zarr-schema-profile
        SOURCES
            tests/schema_profile_test.cc
            tests/support/value_reader_stub.cc
            tests/support/pixel_reader_stub.cc
            src/dataset_size.cc
            src/store.cc
            src/schema/profile.cc
            src/schema/xradio/probe_report.cc
            src/schema/xradio/beam_table.cc
            src/schema/xradio/direction.cc
            src/schema/xradio/flag.cc
            src/schema/xradio/linear_axis.cc
            src/schema/xradio/observation.cc
            src/schema/xradio/qualification.cc
            src/schema/xradio/image.cc
            src/zarr/array_view.cc
            src/zarr/pixel_selection.cc
            src/zarr/string_array.cc
            src/zarr/transport.cc
            ${CARTA_ZARR_METADATA_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json ZLIB::ZLIB Zstd::Zstd Blosc::blosc)

    # ArrayView addresses a flat Zarr buffer by dimension name. It needs only its own metadata type,
    # so it is tested directly rather than through a store.
    # What an array's document says about how it is stored. Links only nlohmann: ParseStorageLayout
    # was a Store method reading the raw node metadata, so its sharding rules could be reached only
    # by writing a store to a directory and opening an image out of it, and two of its refusals
    # could not be reached at all.
    carta_zarr_add_test(carta_zarr_array_metadata_tests INTERNAL
        TEST_NAME carta-zarr-array-metadata
        SOURCES tests/array_metadata_test.cc ${CARTA_ZARR_METADATA_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json)

    carta_zarr_add_test(carta_zarr_array_view_tests INTERNAL
        TEST_NAME carta-zarr-array-view
        SOURCES tests/array_view_test.cc src/zarr/array_view.cc ${CARTA_ZARR_METADATA_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json)

    # Memo is a header-only utility, so this target compiles only the test. It pins the two
    # decisions Store leans on: a value is computed exactly once, and a failure is remembered.
    carta_zarr_add_test(carta_zarr_memo_tests INTERNAL
        TEST_NAME carta-zarr-memo
        SOURCES tests/memo_test.cc
        LIBRARIES Threads::Threads)

    # What the filesystem transport says is in a store. Compiled on its own because it needs nothing
    # above the seam: the listing is asserted as the transport hands it over, before Store sorts and
    # de-duplicates it and the shape of the walk stops being observable.
    carta_zarr_add_test(carta_zarr_transport_tests INTERNAL
        TEST_NAME carta-zarr-transport
        SOURCES tests/transport_test.cc src/zarr/transport.cc
        LIBRARIES nlohmann_json::nlohmann_json)

    # What a pool of its own is made of: its size, whose threads it runs on, and whether anything keeps
    # it. Compiled on its own because a consumer reading through one cannot see any of the three --
    # the pixels are the same whatever the pool.
    carta_zarr_add_test(carta_zarr_store_context_tests INTERNAL
        TEST_NAME carta-zarr-store-context
        SOURCES tests/store_context_test.cc src/zarr/store_context.cc
        LIBRARIES
            nlohmann_json::nlohmann_json
            tensorstore::tensorstore
            tensorstore::driver_zarr3
            tensorstore::kvstore_file)

    # The worker pool and the rule that decides whether to wake it. Compiled on its own because the
    # reductions' own fixtures are far smaller than the split's threshold, so the parallel path is
    # unreachable from there -- see the comment at the top of the test.
    carta_zarr_add_test(carta_zarr_work_pool_tests INTERNAL
        TEST_NAME carta-zarr-work-pool
        SOURCES tests/work_pool_test.cc src/work_pool.cc
        LIBRARIES Threads::Threads)

    # Where an image points, from its root attributes and its l and m samples. No store, no
    # transport: the whole thing takes JSON and two vectors. ADR 0002 rests on these fields reaching
    # casacore intact, and the two that were asserted nowhere -- the increment and the
    # transformation matrix -- are checked here.
    carta_zarr_add_test(carta_zarr_direction_tests INTERNAL
        TEST_NAME carta-zarr-direction
        SOURCES
            tests/direction_test.cc
            src/schema/xradio/direction.cc
            src/schema/xradio/linear_axis.cc
            ${CARTA_ZARR_METADATA_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json)

    # The linear axis fit is pure: values in, a linear description out. It is tested on its own,
    # with no store and no transport, so the cases that are awkward to build as a directory tree --
    # unevenly spaced samples, an extrapolated reference pixel, degenerate axes -- are cheap here.
    carta_zarr_add_test(carta_zarr_linear_axis_tests INTERNAL
        TEST_NAME carta-zarr-linear-axis
        SOURCES tests/linear_axis_test.cc src/schema/xradio/linear_axis.cc)

    # What an image's own attributes say about the observation. A pure JSON-to-struct function, so
    # this links nothing: the geodetic conversion and the obsdate split used to be reachable only by
    # opening a store, and the conversion was asserted nowhere at all.
    carta_zarr_add_test(carta_zarr_observation_tests INTERNAL
        TEST_NAME carta-zarr-observation
        SOURCES tests/observation_test.cc src/schema/xradio/observation.cc
        LIBRARIES nlohmann_json::nlohmann_json)

    # What a variable has to be before it can mask an image. RequireUsableFlag is inline, so this
    # target compiles one file and links nothing -- choosing a flag when none is declared needs a
    # store and is covered in the profile tests instead.
    carta_zarr_add_test(carta_zarr_flag_tests INTERNAL
        TEST_NAME carta-zarr-flag
        SOURCES tests/flag_test.cc
        LIBRARIES nlohmann_json::nlohmann_json)

    # The beam table's indexing, against the arrays already read rather than a store. It needs
    # ArrayView, which is what addresses the flat buffer by dimension name.
    carta_zarr_add_test(carta_zarr_beam_table_tests INTERNAL
        TEST_NAME carta-zarr-beam-table
        SOURCES
            tests/beam_table_test.cc
            src/schema/xradio/beam_table.cc
            src/zarr/array_view.cc
            ${CARTA_ZARR_METADATA_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json)

    # The stored layout permuted into the order the library reads. Links nothing: it takes a
    # descriptor and a layout, so the sharded and permuted cases stop depending on which committed
    # fixture happens to have them.
    carta_zarr_add_test(carta_zarr_chunk_geometry_tests INTERNAL
        TEST_NAME carta-zarr-chunk-geometry
        SOURCES tests/chunk_geometry_test.cc)

    # What an ordinary read decides before it reads anything. Compiled against pieces.cc alone:
    # PlanPieces touches no store and no transport, so where a read is cut and how big a piece is --
    # which is where the progress callback and the memory ceiling meet -- is checkable without a
    # directory tree.
    carta_zarr_add_test(carta_zarr_piece_plan_tests INTERNAL
        TEST_NAME carta-zarr-piece-plan
        SOURCES tests/piece_plan_test.cc src/read/pieces.cc)

    # What an ordinary read does once it starts, against pixels that were never written down. The
    # read takes a PixelSource like the pass does, so this links no Store, no TensorStore and no
    # WorkPool, since a read is not a reduction and has no arithmetic to run on one -- and
    # the flag-before-pixels contract, which a fixture on disk cannot show because it cannot be told
    # to fail one read and not another, is checkable here.
    carta_zarr_add_test(carta_zarr_read_synthetic_tests INTERNAL
        TEST_NAME carta-zarr-read-synthetic
        SOURCES
            tests/read_synthetic_test.cc
            src/read/pieces.cc
            src/read/pieces_read.cc
            src/zarr/pixel_selection.cc
        LIBRARIES nlohmann_json::nlohmann_json)

    # What a reduction may ask for, asked of nothing but a descriptor. This target compiles no
    # implementation source at all: a plane selection is checked from the image's axes and the
    # request, so every way one can be refused is reachable with no store, no pool and no pixel.
    carta_zarr_add_test(carta_zarr_plane_selection_tests INTERNAL
        TEST_NAME carta-zarr-plane-selection
        SOURCES tests/plane_selection_test.cc)

    # The one-pass histogram's arithmetic, against chosen values and exact counts. Links nothing:
    # the class takes floats and gives back counts, and its three past fixes were each a defect in
    # one of these four behaviours that could only be seen as a tolerance over a fixture.
    carta_zarr_add_test(carta_zarr_growing_histogram_tests INTERNAL
        TEST_NAME carta-zarr-growing-histogram
        SOURCES tests/growing_histogram_test.cc)

    # What a pass decides before it reads anything. Compiled against pass.cc alone: PlanPass touches
    # no store and no transport, so the read strategy -- which has moved more than once, and moved
    # in three places each time -- is checkable without a directory tree.
    carta_zarr_add_test(carta_zarr_pass_tests INTERNAL
        TEST_NAME carta-zarr-pass
        SOURCES tests/pass_test.cc ${CARTA_ZARR_PASS_SOURCES}
        LIBRARIES nlohmann_json::nlohmann_json)

    # What a reduction decides about its blocks, before it reads anything. Compiled against pass.cc
    # alone, and the walk it drives reads nothing: where a block is cut, what a part-filled one
    # reports, and what a sink saying no means are all answerable with no pixels.
    carta_zarr_add_test(carta_zarr_block_emit_tests INTERNAL
        TEST_NAME carta-zarr-block-emit
        SOURCES tests/block_emit_test.cc src/reduce/pass.cc
        LIBRARIES nlohmann_json::nlohmann_json)

    # The layout a spectral reduction accumulates in, and the block it hands over. Header-only and
    # compiled against nothing: which slot a statistic lands in and what an untouched extremum is
    # reported as are questions about doubles in a buffer, not about pixels.
    carta_zarr_add_test(carta_zarr_statistic_slots_tests INTERNAL
        TEST_NAME carta-zarr-statistic-slots
        SOURCES tests/statistic_slots_test.cc)

    # A region's raster turned into the runs a reduction walks, checked against the obvious encoder
    # the library avoids. Compiled against the encoder alone.
    carta_zarr_add_test(carta_zarr_region_runs_tests INTERNAL
        TEST_NAME carta-zarr-region-runs
        SOURCES tests/region_runs_test.cc src/reduce/region_runs.cc)

    # A whole reduction against pixels that were never written down. This target links no Store and
    # no TensorStore, which is the seam the pass's PixelSource exists to create: the oracle is the
    # formula the pixels come from, so the image can be whatever size the question needs rather than
    # whatever size is worth committing.
    carta_zarr_add_test(carta_zarr_reduce_synthetic_tests INTERNAL
        TEST_NAME carta-zarr-reduce-synthetic
        SOURCES
            tests/reduce_synthetic_test.cc
            src/reduce/occupancy.cc
            src/reduce/region_runs.cc
            src/reduce/plane_histogram.cc
            src/reduce/spectral_reduce.cc
            src/work_pool.cc
            ${CARTA_ZARR_PASS_SOURCES}
        LIBRARIES Threads::Threads)

    # The same reductions with the overridable constants of tuning.h at each extreme: a task for
    # every pixel and room for every worker's accumulator, so that every split there is is taken, and
    # no room for a second accumulator, so that none is. ADR 0014 lets these be replaced only because
    # they decide how the work is divided and never what it answers, and these two check that claim:
    # every answer reduce_synthetic_test.cc expects must come out the same either way.
    foreach(extreme split_all split_none)
        if(extreme STREQUAL "split_all")
            set(least 1ull)
            set(budget 1099511627776ull)
        else()
            set(least 4611686018427387904ull)
            set(budget 1ull)
        endif()
        string(REPLACE "_" "-" suffix "${extreme}")
        carta_zarr_add_test(carta_zarr_reduce_synthetic_${extreme}_tests INTERNAL
            TEST_NAME carta-zarr-reduce-synthetic-${suffix}
            SOURCES
                tests/reduce_synthetic_test.cc
                src/reduce/occupancy.cc
                src/reduce/region_runs.cc
                src/reduce/plane_histogram.cc
                src/reduce/spectral_reduce.cc
                src/work_pool.cc
                ${CARTA_ZARR_PASS_SOURCES}
            DEFINITIONS
                CARTA_ZARR_TUNING_LEAST_PIXELS_PER_TASK=${least}
                CARTA_ZARR_TUNING_SPECTRAL_PARTIAL_BUDGET_BYTES=${budget}
                CARTA_ZARR_TUNING_HISTOGRAM_PARTIAL_BUDGET_BYTES=${budget}
                CARTA_ZARR_TUNING_CUBE_ACCUMULATOR_CACHE_BYTES=${budget}
            LIBRARIES Threads::Threads)
    endforeach()

    # What CARTA_ZARR_TUNING_OVERRIDES accepts, and how a bench row will describe it.
    add_test(NAME carta-zarr-tuning-overrides
        COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/tuning_overrides_test.cmake)

    carta_zarr_add_test(carta_zarr_chunk_blocks_tests INTERNAL
        TEST_NAME carta-zarr-chunk-blocks
        SOURCES tests/chunk_blocks_test.cc)

    # Which chunks a set of regions occupies. LIBRARIES is empty and meant to stay empty: the
    # question is regions and a chunk shape in, an index out, so a target that starts needing a
    # plan, a store or a thread pool has been handed something that is not its question. The 180
    # lines behind Occupancy::Of were reachable only through ReduceSpectral over a real image
    # before this, and two of their refusals were reachable from nothing at all.
    carta_zarr_add_test(carta_zarr_occupancy_tests INTERNAL
        TEST_NAME carta-zarr-occupancy
        SOURCES tests/occupancy_test.cc src/reduce/occupancy.cc src/reduce/region_runs.cc)

    # What a pixel selection is: a descriptor and a request in, a selection out. It was reached only
    # through reads that checked their answers, so a selection that was right about which pixels and
    # wrong about where they land could only show up as a wrong sum.
    carta_zarr_add_test(carta_zarr_selection_tests INTERNAL
        TEST_NAME carta-zarr-selection
        SOURCES tests/pixel_selection_test.cc src/zarr/pixel_selection.cc)

    # Conformance runs against a store XRADIO itself produced, so that a bump of the pinned XRADIO
    # version reports which of this library's assumptions it broke.
    carta_zarr_add_test(carta_zarr_conformance_tests PUBLIC
        TEST_NAME carta-zarr-xradio-conformance
        SOURCES tests/conformance_test.cc
        DEFINITIONS CARTA_ZARR_CONFORMANCE_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/conformance")

    # Pixel reads run against a fixture whose values encode their own logical coordinates, so the
    # axis permutation, the stride, the fill value of a deleted chunk and the pixel mask each have a
    # wrong answer that the test can tell apart from the right one.
    carta_zarr_add_test(carta_zarr_pixel_read_tests PUBLIC
        TEST_NAME carta-zarr-pixel-read
        SOURCES tests/pixel_read_test.cc
        LIBRARIES Threads::Threads
        DEFINITIONS ${CARTA_ZARR_PIXEL_FIXTURES})

    # The spectral reduction is checked against an oracle recomputed from the fixture's encoding,
    # not against a recorded answer, so that a walk which reads the right pixels but double-counts a
    # region straddling two chunks disagrees with it.
    carta_zarr_add_test(carta_zarr_spectral_reduce_tests PUBLIC
        TEST_NAME carta-zarr-spectral-reduce
        SOURCES tests/spectral_reduce_test.cc
        DEFINITIONS ${CARTA_ZARR_PIXEL_FIXTURES} ${CARTA_ZARR_WIDE_FIXTURE})

    carta_zarr_add_test(carta_zarr_histogram_tests PUBLIC
        TEST_NAME carta-zarr-histogram
        SOURCES tests/histogram_test.cc
        DEFINITIONS ${CARTA_ZARR_PIXEL_FIXTURES} ${CARTA_ZARR_WIDE_FIXTURE})

    # Not a ctest: it reports durations and has no opinion about them, because every fixture here is
    # too small for a threshold to mean anything. It is the guard described in ADR 0005 -- run it
    # against build-release/ before and after a change to the pass, and compare. See its own header
    # for why a Debug number is worthless. No TEST_NAME, so it is built and never run by ctest.
    carta_zarr_add_test(carta_zarr_pass_timing PUBLIC
        SOURCES tests/pass_timing.cc
        DEFINITIONS ${CARTA_ZARR_WIDE_FIXTURE})

    #[[
      The layout generator under tools/zarr-bench is Python run by uv, which brings its own pinned
      interpreter and packages. The setup test writes small datasets with it into the build tree, and
      the test after it reads them back through the library: a layout the generator writes and the
      library cannot read is the one failure that would make every later measurement meaningless.
      Without uv there is nothing to run them with, which is a reason to skip them rather than fail.
    ]]
    if(CARTA_ZARR_BUILD_BENCH)
        # carta-zarr-bench below its command line -- where it reads, what it writes, and what it reads
        # there -- and then the executable itself, run as the sweep runs it. Neither needs uv.
        carta_zarr_add_test(carta_zarr_bench_tests PUBLIC
            TEST_NAME carta-zarr-bench
            SOURCES tests/bench_test.cc
            LIBRARIES carta_zarr_bench_core
            DEFINITIONS ${CARTA_ZARR_WIDE_FIXTURE})
        add_test(
            NAME carta-zarr-bench-run
            COMMAND ${CMAKE_COMMAND}
                -DBENCH=$<TARGET_FILE:carta_zarr_bench>
                -DFIXTURE=${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels_wide
                -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/bench-run
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_bench_smoke.cmake)

        find_program(CARTA_ZARR_UV uv)
        if(CARTA_ZARR_UV)
            set(CARTA_ZARR_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/bench-generated")
            add_test(
                NAME carta-zarr-bench-generate
                COMMAND ${CMAKE_COMMAND}
                    -DUV=${CARTA_ZARR_UV}
                    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                    -DOUTPUT_DIR=${CARTA_ZARR_GENERATED_DIR}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_bench_generator.cmake)
            set_tests_properties(carta-zarr-bench-generate PROPERTIES FIXTURES_SETUP bench-generated)

            carta_zarr_add_test(carta_zarr_bench_generator_tests PUBLIC
                TEST_NAME carta-zarr-bench-generator
                SOURCES tests/bench_generator_test.cc
                LIBRARIES nlohmann_json::nlohmann_json
                DEFINITIONS
                    CARTA_ZARR_GENERATED_DIR="${CARTA_ZARR_GENERATED_DIR}"
                    CARTA_ZARR_PIXEL_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels"
                    ${CARTA_ZARR_WIDE_FIXTURE})
            set_tests_properties(carta-zarr-bench-generator PROPERTIES FIXTURES_REQUIRED bench-generated)

            # The generator's layouts of one cube, fingerprinted alike by the bench at every position.
            add_test(NAME carta-zarr-bench-layouts
                COMMAND carta_zarr_bench_tests layouts ${CARTA_ZARR_GENERATED_DIR})
            set_tests_properties(carta-zarr-bench-layouts PROPERTIES FIXTURES_REQUIRED bench-generated)

            # The whole sweep -- generate, bench, rank, report, resume -- on a committed fixture.
            add_test(
                NAME carta-zarr-bench-sweep
                COMMAND ${CMAKE_COMMAND}
                    -DUV=${CARTA_ZARR_UV}
                    -DBENCH=$<TARGET_FILE:carta_zarr_bench>
                    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/bench-sweep
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_bench_sweep.cmake)
        else()
            message(STATUS "uv was not found; the zarr-bench generator tests are skipped")
        endif()
    endif()

    if(CARTA_ZARR_BUILD_PACKAGE_TEST)
        add_test(
            NAME carta-zarr-package-consumer
            COMMAND ${CMAKE_COMMAND}
                -DCARTA_ZARR_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
                -DCARTA_ZARR_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                -DCARTA_ZARR_PACKAGE_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-consumer
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_package_consumer.cmake)
        set_tests_properties(carta-zarr-package-consumer PROPERTIES DEPENDS carta-zarr-schema-probe)
    endif()
endif()
