#pragma once

#if defined(FFT_STATIC)
#  define FFT_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(FFT_BUILDING_LIBRARY)
#    define FFT_API __declspec(dllexport)
#  else
#    define FFT_API __declspec(dllimport)
#  endif
#else
#  if defined(FFT_BUILDING_LIBRARY)
#    define FFT_API __attribute__((visibility("default")))
#  else
#    define FFT_API
#  endif
#endif
