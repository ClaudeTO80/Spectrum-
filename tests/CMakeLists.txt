add_executable(fft_core_tests test_core.cpp)
target_link_libraries(fft_core_tests PRIVATE
    $<$<BOOL:${FFT_BUILD_STATIC}>:FFTStatic>
    $<$<NOT:$<BOOL:${FFT_BUILD_STATIC}>>:FFTShared>
)
fft_enable_warnings(fft_core_tests)
add_test(NAME fft_core_tests COMMAND fft_core_tests)

if(FFT_BUILD_STRESS_TESTS)
    add_executable(fft_stress_tests test_stress.cpp)
    target_link_libraries(fft_stress_tests PRIVATE
        $<$<BOOL:${FFT_BUILD_STATIC}>:FFTStatic>
        $<$<NOT:$<BOOL:${FFT_BUILD_STATIC}>>:FFTShared>
    )
    fft_enable_warnings(fft_stress_tests)
    add_test(NAME fft_stress_tests COMMAND fft_stress_tests)
endif()
