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
    src/reduce/plane_histogram.cc
    src/reduce/spectral_reduce.cc
    src/schema/profile.cc
    src/schema/xradio/probe_report.cc
    src/schema/xradio/linear_axis.cc
    src/schema/xradio/image.cc
    src/zarr/array_metadata.cc
    src/zarr/array_view.cc
    src/zarr/pixel_reader.cc
    src/zarr/store_context.cc
    src/zarr/string_array.cc
    src/zarr/transport.cc
    src/zarr/value_reader.cc
    src/store.cc)
add_library(CARTA::zarr ALIAS carta_zarr)

target_compile_features(carta_zarr PUBLIC cxx_std_17)
target_compile_definitions(carta_zarr PRIVATE CARTA_ZARR_BUILDING_LIBRARY)
target_include_directories(carta_zarr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(carta_zarr PRIVATE
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

include(CTest)
if(CARTA_ZARR_BUILD_TESTS AND BUILD_TESTING)
    add_executable(carta_zarr_tests tests/schema_probe_test.cc)
    target_link_libraries(carta_zarr_tests PRIVATE CARTA::zarr)
    target_compile_features(carta_zarr_tests PRIVATE cxx_std_17)
    set(CARTA_ZARR_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/images/zarr")
    target_compile_definitions(carta_zarr_tests PRIVATE
        CARTA_ZARR_REFERENCE_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/minimal"
        CARTA_ZARR_LEGACY_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/legacy")
    add_test(NAME carta-zarr-schema-probe COMMAND carta_zarr_tests)

    # The string decoder is an internal component with hidden visibility, so this test compiles it
    # directly instead of linking the shared library.
    add_executable(carta_zarr_string_tests
        tests/string_array_test.cc
        src/zarr/string_array.cc
        src/zarr/array_metadata.cc)
    target_compile_features(carta_zarr_string_tests PRIVATE cxx_std_17)
    target_include_directories(carta_zarr_string_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(carta_zarr_string_tests PRIVATE
        nlohmann_json::nlohmann_json
        ZLIB::ZLIB
        Zstd::Zstd
        Blosc::blosc)
    target_compile_definitions(carta_zarr_string_tests PRIVATE
        CARTA_ZARR_STRING_FIXTURE_DIR="${CARTA_ZARR_FIXTURE_DIR}/string")
    add_test(NAME carta-zarr-string-array COMMAND carta_zarr_string_tests)

    # The schema profile's structural decisions read metadata only, so they run against an in-memory
    # transport instead of a directory tree. This target compiles the internals directly and links no
    # TensorStore: the value reader is stubbed, which is also the honest behaviour of a transport
    # holding no array data.
    add_executable(carta_zarr_profile_tests
        tests/schema_profile_test.cc
        tests/support/value_reader_stub.cc
        tests/support/pixel_reader_stub.cc
        src/store.cc
        src/schema/profile.cc
        src/schema/xradio/probe_report.cc
        src/schema/xradio/linear_axis.cc
        src/schema/xradio/image.cc
        src/zarr/array_metadata.cc
        src/zarr/array_view.cc
        src/zarr/string_array.cc
        src/zarr/transport.cc)
    target_compile_features(carta_zarr_profile_tests PRIVATE cxx_std_17)
    target_include_directories(carta_zarr_profile_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(carta_zarr_profile_tests PRIVATE
        nlohmann_json::nlohmann_json
        ZLIB::ZLIB
        Zstd::Zstd
        Blosc::blosc)
    add_test(NAME carta-zarr-schema-profile COMMAND carta_zarr_profile_tests)

    # ArrayView addresses a flat Zarr buffer by dimension name. It needs only its own metadata type,
    # so it is tested directly rather than through a store.
    add_executable(carta_zarr_array_view_tests
        tests/array_view_test.cc
        src/zarr/array_view.cc
        src/zarr/array_metadata.cc)
    target_compile_features(carta_zarr_array_view_tests PRIVATE cxx_std_17)
    target_include_directories(carta_zarr_array_view_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(carta_zarr_array_view_tests PRIVATE nlohmann_json::nlohmann_json)
    add_test(NAME carta-zarr-array-view COMMAND carta_zarr_array_view_tests)

    # Memo is a header-only utility, so this target compiles only the test. It pins the two
    # decisions Store leans on: a value is computed exactly once, and a failure is remembered.
    add_executable(carta_zarr_memo_tests tests/memo_test.cc)
    target_compile_features(carta_zarr_memo_tests PRIVATE cxx_std_17)
    target_include_directories(carta_zarr_memo_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    find_package(Threads REQUIRED)
    target_link_libraries(carta_zarr_memo_tests PRIVATE Threads::Threads)
    add_test(NAME carta-zarr-memo COMMAND carta_zarr_memo_tests)

    # The linear axis fit is pure: values in, a linear description out. It is tested on its own,
    # with no store and no transport, so the cases that are awkward to build as a directory tree --
    # unevenly spaced samples, an extrapolated reference pixel, degenerate axes -- are cheap here.
    add_executable(carta_zarr_linear_axis_tests
        tests/linear_axis_test.cc
        src/schema/xradio/linear_axis.cc)
    target_compile_features(carta_zarr_linear_axis_tests PRIVATE cxx_std_17)
    target_include_directories(carta_zarr_linear_axis_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME carta-zarr-linear-axis COMMAND carta_zarr_linear_axis_tests)

    add_executable(carta_zarr_chunk_blocks_tests tests/chunk_blocks_test.cc)
    target_compile_features(carta_zarr_chunk_blocks_tests PRIVATE cxx_std_17)
    target_include_directories(carta_zarr_chunk_blocks_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME carta-zarr-chunk-blocks COMMAND carta_zarr_chunk_blocks_tests)

    # Conformance runs against a store XRADIO itself produced, so that a bump of the pinned XRADIO
    # version reports which of this library's assumptions it broke.
    add_executable(carta_zarr_conformance_tests tests/conformance_test.cc)
    target_link_libraries(carta_zarr_conformance_tests PRIVATE CARTA::zarr)
    target_compile_features(carta_zarr_conformance_tests PRIVATE cxx_std_17)
    target_compile_definitions(carta_zarr_conformance_tests PRIVATE
        CARTA_ZARR_CONFORMANCE_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/conformance")
    add_test(NAME carta-zarr-xradio-conformance COMMAND carta_zarr_conformance_tests)

    # Pixel reads run against a fixture whose values encode their own logical coordinates, so the
    # axis permutation, the stride, the fill value of a deleted chunk and the pixel mask each have a
    # wrong answer that the test can tell apart from the right one.
    add_executable(carta_zarr_pixel_read_tests tests/pixel_read_test.cc)
    target_link_libraries(carta_zarr_pixel_read_tests PRIVATE CARTA::zarr)
    target_compile_features(carta_zarr_pixel_read_tests PRIVATE cxx_std_17)
    target_compile_definitions(carta_zarr_pixel_read_tests PRIVATE
        CARTA_ZARR_PIXEL_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels"
        CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels_l_fastest")
    target_link_libraries(carta_zarr_pixel_read_tests PRIVATE Threads::Threads)
    add_test(NAME carta-zarr-pixel-read COMMAND carta_zarr_pixel_read_tests)

    # The spectral reduction is checked against an oracle recomputed from the fixture's encoding,
    # not against a recorded answer, so that a walk which reads the right pixels but double-counts a
    # region straddling two chunks disagrees with it.
    add_executable(carta_zarr_spectral_reduce_tests tests/spectral_reduce_test.cc)
    target_link_libraries(carta_zarr_spectral_reduce_tests PRIVATE CARTA::zarr)
    target_compile_features(carta_zarr_spectral_reduce_tests PRIVATE cxx_std_17)
    target_compile_definitions(carta_zarr_spectral_reduce_tests PRIVATE
        CARTA_ZARR_PIXEL_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels"
        CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels_l_fastest")
    add_test(NAME carta-zarr-spectral-reduce COMMAND carta_zarr_spectral_reduce_tests)

    add_executable(carta_zarr_histogram_tests tests/histogram_test.cc)
    target_link_libraries(carta_zarr_histogram_tests PRIVATE CARTA::zarr)
    target_compile_features(carta_zarr_histogram_tests PRIVATE cxx_std_17)
    target_compile_definitions(carta_zarr_histogram_tests PRIVATE
        CARTA_ZARR_PIXEL_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels"
        CARTA_ZARR_PIXEL_FIXTURE_L_FASTEST="${CARTA_ZARR_FIXTURE_DIR}/xradio/pixels_l_fastest")
    add_test(NAME carta-zarr-histogram COMMAND carta_zarr_histogram_tests)

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
