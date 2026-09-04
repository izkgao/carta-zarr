if(NOT DEFINED CARTA_ZARR_BINARY_DIR OR NOT DEFINED CARTA_ZARR_SOURCE_DIR OR NOT DEFINED CARTA_ZARR_PACKAGE_TEST_DIR)
    message(FATAL_ERROR "Package consumer test variables are not set")
endif()

set(prefix "${CARTA_ZARR_PACKAGE_TEST_DIR}/prefix")
set(build_dir "${CARTA_ZARR_PACKAGE_TEST_DIR}/build")
file(REMOVE_RECURSE "${CARTA_ZARR_PACKAGE_TEST_DIR}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${CARTA_ZARR_BINARY_DIR}" --prefix "${prefix}"
    RESULT_VARIABLE result)
if(result)
    message(FATAL_ERROR "carta-zarr install failed: ${result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${CARTA_ZARR_SOURCE_DIR}/tests/consumer"
        -B "${build_dir}"
        -DCMAKE_PREFIX_PATH=${prefix}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_BUILD_RPATH=${prefix}/lib
    RESULT_VARIABLE result)
if(result)
    message(FATAL_ERROR "External consumer configure failed: ${result}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${build_dir}"
    RESULT_VARIABLE result)
if(result)
    message(FATAL_ERROR "External consumer build failed: ${result}")
endif()

execute_process(
    COMMAND "${build_dir}/carta_zarr_consumer"
    RESULT_VARIABLE result)
if(result)
    message(FATAL_ERROR "External consumer execution failed: ${result}")
endif()
