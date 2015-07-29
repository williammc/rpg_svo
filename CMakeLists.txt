cmake_minimum_required(VERSION 2.8)
project(svo)

# user build settings ==========================================================

option(SVO_TRACE "Trace" ON)
option(SVO_HAVE_G2O "Have g2o" OFF) 
option(SVO_DEBUG_OUTPUT OFF) # Only relevant if build without ROS

# Set definitions
if(SVO_TRACE)
  add_definitions(-DSVO_TRACE)
endif()
if(SVO_HAVE_G2O)
  add_definitions(-DUSE_BUNDLE_ADJUSTMENT)
endif()
if(SVO_DEBUG_OUTPUT)
  add_definitions(-DSVO_DEBUG_OUTPUT)
endif()
add_definitions(-D__SSE2__=1)

get_filename_component(SVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
set(SVO_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DSVO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# Look for our CMake modules paths
set(CMAKER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/cmaker)
list(APPEND CMAKE_MODULE_PATH ${CMAKER_ROOT}/cmake
                              ${CMAKER_ROOT}/cmake/modules
                              ${CMAKE_CURRENT_SOURCE_DIR}/cmake
                              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
                              ${CMAKE_CURRENT_BINARY_DIR})

if(NOT THE_PROJECT_ROOT OR THE_PROJECT_ROOT STREQUAL "")
  set(THE_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
endif()

include(CMakerCommonUtil)  # include useful macros
include(CMakerCommonSettings)

set(SVO_CXX_FLAGS)
add_definitions(-DSVO_ROOT="${SVO_ROOT}")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(SVO_CXX_FLAGS " -stdlib=libc++ ")  # prevent default libc++ on new mac
endif()

if(NOT WIN32)
  set(SVO_CXX_FLAGS " ${SVO_CXX_FLAGS} -std=c++11 ") 
endif()
cmaker_print_status("SVO_CXX_FLAGS:${SVO_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS ${SVO_CXX_FLAGS})
cmaker_common_build_setting()
## Dependencies ================================================================

# Eigen
list(APPEND SVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen)

# Sophus
list(APPEND SVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/sophus)

## Boost
if(WIN32)
  set(Boost_USE_STATIC_LIBS  ON)
endif(WIN32)

find_package(Boost REQUIRED COMPONENTS serialization)

list(APPEND SVO_INC_DIRS ${Boost_INCLUDE_DIRS})
list(APPEND SVO_LIB_DIRS ${Boost_LIBRARY_DIR})
cmaker_print_status("Boost libs: ${Boost_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)
set(OCV_LIBS)
foreach(NAME ${OpenCV_LIBRARIES})
  set(NAME ${NAME}${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH})
  list(APPEND OCV_LIBS debug;${NAME}d;optimized;${NAME})
endforeach()
list(APPEND SVO_INC_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND SVO_LIB_DIRS ${OpenCV_LIB_DIR})
list(APPEND SVO_EXTERNAL_LIBS ${OCV_LIBS})
cmaker_print_status("OpenCV include dirs:${OpenCV_INCLUDE_DIRS}")
cmaker_print_status("OpenCV libs:${OpenCV_LIBRARIES}")
get_filename_component(OpenCV_BINARY_DIR "${OpenCV_LIB_DIR}/../bin" ABSOLUTE)
list(APPEND THE_DEPEDENCIES_BINARY_PATHS ${OpenCV_BINARY_DIR})

# Fast
list(APPEND SVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/fast/include)
add_subdirectory(external/fast)
list(APPEND SVO_EXTERNAL_LIBS ${OCV_LIBS} fast)

# VIKIT
add_subdirectory(external/vikit/vikit_common)
list(APPEND SVO_INC_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/vikit/vikit_common/include)
list(APPEND SVO_EXTERNAL_LIBS ${OCV_LIBS} vikit_common)
##==============================================================================
# G2O
find_package(G2O REQUIRED)
set(G2O_BINARY_DIR ${G2O_ROOT}/bin)

set(G2O_LIBRARIES optimized ${G2O_CORE_LIBRARY} debug ${G2O_CORE_LIBRARY_DEBUG} 
                  optimized ${G2O_STUFF_LIBRARY} debug ${G2O_STUFF_LIBRARY_DEBUG}
                  optimized ${G2O_SOLVER_DENSE} debug ${G2O_SOLVER_DENSE_DEBUG}
                  optimized ${G2O_TYPES_SLAM3D} debug ${G2O_TYPES_SLAM3D_DEBUG}
                  optimized ${G2O_SIMULATOR} debug ${G2O_SIMULATOR_DEBUG}
                  )
if(G2O_SOLVER_CSPARSE AND NOT APPLE)
  if(WIN32)
    list(APPEND G2O_INCLUDE_DIR "${G2O_ROOT}/include/EXTERNAL/csparse")
    list(APPEND G2O_INCLUDE_DIR "$ENV{G2O_ROOT}/include/EXTERNAL/csparse")
  else()
    list(APPEND G2O_INCLUDE_DIR "/usr/include/suitesparse")
  endif()
  list(APPEND G2O_LIBRARIES ${G2O_SOLVER_CSPARSE}
                            ${G2O_SOLVER_CSPARSE_EXTENSION}
                            ${G2O_EXT_CSPARSE})
  add_definitions(-DHAVE_SOLVER_CSPARSE)
endif()
if(G2O_SOLVER_CHOLMOD)
  list(APPEND G2O_LIBRARIES ${G2O_SOLVER_CHOLMOD})
  add_definitions(-DHAVE_SOLVER_CHOLMOD)
endif()
if(G2O_SOLVER_EIGEN)
  list(APPEND G2O_LIBRARIES ${G2O_SOLVER_EIGEN})
  add_definitions(-DHAVE_SOLVER_EIGEN)
endif()

if(APPLE)
  find_package(CSparse REQUIRED)
  list(APPEND G2O_LIBRARIES ${CSPARSE_LIBRARY})
  list(APPEND SVO_INC_DIRS ${CSPARSE_INCLUDE_DIR})
else()
  link_directories(${G2O_ROOT}/lib)
endif()
include_directories(${G2O_INCLUDE_DIR})
cmaker_cull_library_paths(G2O_LIBRARIES)
list(APPEND SVO_EXTERNAL_LIBS ${G2O_LIBRARIES})

##==============================================================================
include_directories(${SVO_INC_DIRS})
link_directories(${SVO_LIB_DIRS})
set(SVO_INC_DIRS_PARENT_SCOPE ${SVO_INC_DIRS} PARENT_SCOPE)
set(SVO_LIB_DIRS_PARENT_SCOPE ${SVO_LIB_DIRS} PARENT_SCOPE)
set(sub_dirs svo)

set(REG_EXT "[^/]*([.]cpp|[.]c|[.]cc|[.]h|[.]hpp)$")
foreach(sub_dir ${sub_dirs})
  file(GLOB svo_${sub_dir}_sources "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/*.cpp")
  file(GLOB svo_${sub_dir}_headers "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/*.h")
  list(APPEND svo_sources ${svo_${sub_dir}_sources}
                            ${svo_${sub_dir}_headers})
  source_group("${sub_dir}" REGULAR_EXPRESSION "${CMAKE_CURRENT_SOURCE_DIR}/${sub_dir}/${REG_EXT}")
endforeach()

## The library =================================================================
set(ALL_LIBRARIES ${SVO_EXTERNAL_LIBS})
cmaker_add_library(svo ${SM_LIB_TYPE} ${svo_sources})
set_property(TARGET svo PROPERTY FOLDER "external/svo")

cmaker_print_status("svo_sources:${svo_sources}")

set(ALL_LIBRARIES svo ${SVO_EXTERNAL_LIBS})
set(SVO_ALL_LIBRARIES_PARENT_SCOPE ${ALL_LIBRARIES} PARENT_SCOPE)

## Tests
file(GLOB svo_tests_sources "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")
foreach(test_src ${svo_tests_sources})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} svo  ${SVO_EXTERNAL_LIBS})
  set_property(TARGET ${test_name} PROPERTY FOLDER "external/svo/tests")
endforeach()

# Useful Windows paths settings ================================================
# Useful paths configuration for windows 
if(WIN32)
  if(MSVC)
    if(CMAKE_CL_64)
      set(WIN32_ARCH x64)
    else()
      set(WIN32_ARCH x86)
    endif()
    if(MSVC_VERSION EQUAL 1400)
      set(StructuralModeling_RUNTIME vc8)
    elseif(MSVC_VERSION EQUAL 1500)
      set(StructuralModeling_RUNTIME vc9)
    elseif(MSVC_VERSION EQUAL 1600)
      set(StructuralModeling_RUNTIME vc10)
      get_filename_component(VC_IDE_PATH $ENV{VS100COMNTOOLS}/../IDE ABSOLUTE)
    elseif(MSVC_VERSION EQUAL 1700)
      set(StructuralModeling_RUNTIME vc11)
      get_filename_component(VC_IDE_PATH $ENV{VS110COMNTOOLS}/../IDE ABSOLUTE)
    elseif(MSVC_VERSION EQUAL 1800)
      set(StructuralModeling_RUNTIME vc12)
      get_filename_component(VC_IDE_PATH $ENV{VS120COMNTOOLS}/../IDE ABSOLUTE)
    endif()
    get_filename_component(VC_PATH "${VC_IDE_PATH}/../../VC" ABSOLUTE)
  endif()

  cmaker_print_status("THE_DEPEDENCIES_BINARY_PATHS:${THE_DEPEDENCIES_BINARY_PATHS}")
  list(APPEND THE_DEPEDENCIES_BINARY_PATHS "${VC_IDE_PATH}/../../VC" 
"${VC_IDE_PATH}/../../VC/bin")
  configure_file("${CMAKER_ROOT}/scripts/set_paths.bat.in" 
                 "${CMAKE_CURRENT_BINARY_DIR}/set_paths.bat")
  configure_file("${CMAKER_ROOT}/scripts/set_paths.bat.in" 
                 "${THE_LIB_RUNTIME_OUTPUT_DIRECTORY}/set_paths.bat")
  configure_file("${CMAKER_ROOT}/scripts/set_paths_and_run_vc.bat.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/set_paths_and_run_vc.bat")
  set(SVO_THE_DEPEDENCIES_BINARY_PATHS_PARENT_SCOPE ${THE_DEPEDENCIES_BINARY_PATHS} PARENT_SCOPE)
endif()

# useful configuration (e.x.: put scripts in right places) =====================
if(WIN32)
  configure_file("${CMAKER_ROOT}/scripts/run_ninja_build.bat.in" 
                 "${CMAKE_CURRENT_BINARY_DIR}/run_ninja_build.bat")
  if(NOT EXISTS "${THE_LIB_RUNTIME_OUTPUT_DIRECTORY}/run_windbg.bat")
    configure_file("${CMAKER_ROOT}/scripts/run_windbg.bat.in" 
                 "${THE_LIB_RUNTIME_OUTPUT_DIRECTORY}/run_windbg.bat")
  endif()
  set(THE_BUILD_CMD "run_ninja_build.bat")
  set(THE_FILE_REGEX "^(.*)\\\\(([0-9]*)\\\\)(\\\\s:\\\\s).*error(.*)")
else()
  # configure it to a temp location
  configure_file("${CMAKER_ROOT}/scripts/run_make_build.sh.in" 
                 "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/run_make_build.sh")
  # now copy the temporary into the final destination, setting the permissions
  file(COPY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/run_make_build.sh"
          DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
          FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
          GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  set(THE_BUILD_CMD "run_make_build.sh")
  set(THE_FILE_REGEX "^(.*):([0-9]+):([0-9]+).*error(.*)")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/SVO.sublime-project.in"
               "${CMAKE_CURRENT_BINARY_DIR}/SVO-${CMAKE_BUILD_TYPE}.sublime-project" @ONLY)

