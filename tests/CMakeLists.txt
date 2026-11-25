add_executable(fft_core_tests test_core.cpp)
target_link_libraries(fft_core_tests PRIVATE
    $<$<BOOL:${FFT_BUILD_STATIC}>:FFTStatic>
    $<$<NOT:$<BOOL:${FFT_BUILD_STATIC}>>:FFTShared>
)
add_test(NAME fft_core_tests COMMAND fft_core_tests)
