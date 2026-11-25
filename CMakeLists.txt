cmake_minimum_required(VERSION 3.25)
project(FFT VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(FFT_BUILD_SHARED "Build shared library" ON)
option(FFT_BUILD_STATIC "Build static library" ON)
option(FFT_BUILD_TESTS "Build tests" ON)
option(FFT_BUILD_EXAMPLES "Build examples" ON)

if(NOT FFT_BUILD_SHARED AND NOT FFT_BUILD_STATIC)
    message(FATAL_ERROR "At least one of FFT_BUILD_SHARED or FFT_BUILD_STATIC must be ON")
endif()

set(FFT_PUBLIC_HEADERS
    include/fft/fft.hpp
    include/fft/fft_export.hpp
    include/fft/version.hpp
    include/fft/types.hpp
    include/fft/exceptions.hpp
    include/fft/shape.hpp
    include/fft/workspace.hpp
    include/fft/fft_plan.hpp
)

set(FFT_SOURCES
    src/fft_plan.cpp
)

function(fft_configure_target target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_target_properties(${target} PROPERTIES
        CXX_EXTENSIONS OFF
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        EXPORT_NAME ${target}
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

if(FFT_BUILD_SHARED)
    add_library(FFTShared SHARED ${FFT_SOURCES} ${FFT_PUBLIC_HEADERS})
    target_compile_definitions(FFTShared PRIVATE FFT_BUILDING_LIBRARY)
    target_compile_definitions(FFTShared INTERFACE FFT_USING_SHARED)
    fft_configure_target(FFTShared)
    add_library(FFT::Shared ALIAS FFTShared)
    install(TARGETS FFTShared EXPORT FFTTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

if(FFT_BUILD_STATIC)
    add_library(FFTStatic STATIC ${FFT_SOURCES} ${FFT_PUBLIC_HEADERS})
    target_compile_definitions(FFTStatic PUBLIC FFT_STATIC)
    fft_configure_target(FFTStatic)
    add_library(FFT::Static ALIAS FFTStatic)
    install(TARGETS FFTStatic EXPORT FFTTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(FFT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FFT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT FFTTargets
    FILE FFTTargets.cmake
    NAMESPACE FFT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FFT
)

configure_package_config_file(
    cmake/FFTConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FFTConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FFT
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FFTConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/FFTConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FFTConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FFT
)
