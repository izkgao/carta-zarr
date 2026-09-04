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
    src/schema/registry.cc
    src/schema/xradio/common.cc
    src/schema/xradio/sky.cc
    src/zarr/array_metadata.cc
    src/zarr/store_context.cc
    src/zarr/string_array.cc
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
    tensorstore::kvstore_file)

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

    # Conformance runs against a store XRADIO itself produced, so that a bump of the pinned XRADIO
    # version reports which of this library's assumptions it broke.
    add_executable(carta_zarr_conformance_tests tests/conformance_test.cc)
    target_link_libraries(carta_zarr_conformance_tests PRIVATE CARTA::zarr)
    target_compile_features(carta_zarr_conformance_tests PRIVATE cxx_std_17)
    target_compile_definitions(carta_zarr_conformance_tests PRIVATE
        CARTA_ZARR_CONFORMANCE_FIXTURE="${CARTA_ZARR_FIXTURE_DIR}/xradio/conformance")
    add_test(NAME carta-zarr-xradio-conformance COMMAND carta_zarr_conformance_tests)

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
