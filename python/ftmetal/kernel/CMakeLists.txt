cmake_minimum_required(VERSION 3.27)

project(ftmetal_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

# Same nanobind ABI as the mlx.core wheel (non-limited API, "mlx" domain) so the
# two modules share the mlx::array type casters.
nanobind_add_module(
  _C
  NB_STATIC
  FREE_THREADED
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/gdn/gdn_decode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/mha/mha_decode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/attention/mha/mha_prefill.cpp
  ${CMAKE_CURRENT_LIST_DIR}/embedding/ple_lookup.cpp
  ${CMAKE_CURRENT_LIST_DIR}/kvcache/pool_rows.cpp
  ${CMAKE_CURRENT_LIST_DIR}/moe/affine/affine_glu_pending.cpp
  ${CMAKE_CURRENT_LIST_DIR}/moe/gather_sorted.cpp
  ${CMAKE_CURRENT_LIST_DIR}/moe/moe_lookup.cpp
  ${CMAKE_CURRENT_LIST_DIR}/sample/sample_tokens.cpp
  ${CMAKE_CURRENT_LIST_DIR}/service/expert/expert_lru.cpp
  ${CMAKE_CURRENT_LIST_DIR}/service/expert/expert_mailbox.cpp
  ${CMAKE_CURRENT_LIST_DIR}/service/expert/expert_service.cpp
  ${CMAKE_CURRENT_LIST_DIR}/service/expert/victim_cache.cpp
  ${CMAKE_CURRENT_LIST_DIR}/service/ple/ple_row_store.cpp
  ${CMAKE_CURRENT_LIST_DIR}/service/ple/ple_service.cpp)

# Includes name their header from kernel/: "moe/moe_lookup.h".
target_include_directories(_C PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${MLX_INCLUDE_DIRS})
target_compile_options(_C PRIVATE ${MLX_CXX_FLAGS})
# libmlx symbols resolve from the mlx.core module already loaded in the
# process, so the build env's mlx wheel location never leaks into an rpath.
target_link_options(_C PRIVATE -undefined dynamic_lookup)
