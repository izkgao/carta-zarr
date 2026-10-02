# Every symbol libcarta-zarr exports is its own.
#
# The library links TensorStore and the libraries it brings in statically, and the linker is told to
# export carta::zarr and nothing else (cmake/carta_zarr.map and carta_zarr.exp). Without that it
# exported over eleven thousand symbols of Abseil, riegeli, zstd, zlib and blosc beside its own 42,
# and a program linking its own zstd or zlib had two of each. This reads the built library's dynamic
# symbol table and fails on any defined symbol outside carta::zarr, and on a library that exports
# none of the API -- which an export list with a typo would produce.
#
# Takes LIBRARY, the built library, and NM.
cmake_minimum_required(VERSION 3.24)

if(APPLE)
    execute_process(COMMAND "${NM}" -gU "${LIBRARY}" OUTPUT_VARIABLE table RESULT_VARIABLE failed)
    set(prefix "__Z")
else()
    execute_process(COMMAND "${NM}" -D --defined-only "${LIBRARY}" OUTPUT_VARIABLE table RESULT_VARIABLE failed)
    set(prefix "_Z")
endif()
if(failed)
    message(FATAL_ERROR "${NM} could not read ${LIBRARY}")
endif()

# What the ELF toolchain defines in every shared object, and the version node the map declares.
set(always _init _fini __bss_start _edata _end CARTA_ZARR_0)

string(REPLACE "\n" ";" lines "${table}")
set(own 0)
set(foreign "")
foreach(line IN LISTS lines)
    if(line STREQUAL "")
        continue()
    endif()
    string(REGEX REPLACE "^.* " "" name "${line}")
    if(name MATCHES "^${prefix}(N|NK|TIN|TSN|TVN)5carta4zarr")
        math(EXPR own "${own} + 1")
    elseif(NOT name IN_LIST always)
        list(APPEND foreign "${name}")
    endif()
endforeach()

list(LENGTH foreign count)
if(count GREATER 0)
    list(SUBLIST foreign 0 20 shown)
    list(JOIN shown "\n  " shown)
    message(FATAL_ERROR "${LIBRARY} exports ${count} symbols that are not carta-zarr's, among them:\n  ${shown}")
endif()
if(own LESS 10)
    message(FATAL_ERROR "${LIBRARY} exports ${own} carta-zarr symbols; the export list has lost the API")
endif()
message(STATUS "${LIBRARY} exports ${own} symbols, all carta-zarr's")
