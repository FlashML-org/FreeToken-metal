cmake_minimum_required(VERSION 3.27)

project(ftmetal_command_graph LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.13
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
  ${CMAKE_CURRENT_LIST_DIR}/command_graph.mm)

target_include_directories(_C PRIVATE ${MLX_INCLUDE_DIRS})
target_compile_options(_C PRIVATE ${MLX_CXX_FLAGS})
# The recorder holds Metal objects by manual retain/release, as metal-cpp does.
set_source_files_properties(
  ${CMAKE_CURRENT_LIST_DIR}/command_graph.mm PROPERTIES COMPILE_OPTIONS -fno-objc-arc)
target_link_libraries(_C PRIVATE "-framework Metal" "-framework Foundation")
# libmlx symbols resolve from the mlx.core module already loaded in the
# process, so the build env's mlx wheel location never leaks into an rpath.
target_link_options(_C PRIVATE -undefined dynamic_lookup)
