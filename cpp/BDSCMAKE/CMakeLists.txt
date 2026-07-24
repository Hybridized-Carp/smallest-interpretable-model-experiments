# CMakeList.txt : CMake project for BDSCMAKE, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("BDSCMAKE")

include(FetchContent)
FetchContent_Declare(gf2 URL https://github.com/nessan/gf2/releases/download/current/gf2.zip)
FetchContent_MakeAvailable(gf2)

# Add source to this project's executable.
add_executable (BDSCMAKE "BDSCMAKE.cpp" "BDSCMAKE.h")
target_include_directories(BDSCMAKE PRIVATE "${CMAKE_BINARY_DIR}/deps")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET BDSCMAKE PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
