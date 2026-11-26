add_executable(fft_core_example core_example.cpp)
add_executable(fft_advanced_example advanced_example.cpp)
add_executable(fft_image_clean image_clean.cpp)

foreach(example fft_core_example fft_advanced_example fft_image_clean)
    if(FFT_BUILD_STATIC)
        target_link_libraries(${example} PRIVATE FFTStatic)
    else()
        target_link_libraries(${example} PRIVATE FFTShared)
    endif()
    fft_enable_warnings(${example})
endforeach()

# stb is third-party code: its headers are included as system headers so that
# its warnings do not affect the project's warning set.
target_include_directories(fft_image_clean SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../3dParty)
