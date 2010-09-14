#pragma once

#include <cmath>
#include <limits>
#include <base/types.h>

namespace pbgi {

#if WIN32
#include <float.h>

inline bool is_finite(const double x) { return _finite(x) != 0; }
inline bool is_nan(const double x) { return _isnan(x) != 0; }
inline bool is_finite(const float x) { return _finite(x) != 0; }
inline bool is_nan(const float x) { return _isnan(x) != 0; }

#endif

/// round a floating point number
inline HLBVH_HOST HLBVH_DEVICE float round(const float x)
{
	const int y = x > 0.0f ? int(x) : int(x)-1;
	return (x - float(y) > 0.5f) ? float(y)+1.0f : float(y);
}

/// minimum of two floats
inline HLBVH_HOST HLBVH_DEVICE float min(const float a, const float b) { return a < b ? a : b; }

/// maximum of two floats
inline HLBVH_HOST HLBVH_DEVICE float max(const float a, const float b) { return a > b ? a : b; }

/// minimum of two int32
inline HLBVH_HOST HLBVH_DEVICE int32 min(const int32 a, const int32 b) { return a < b ? a : b; }

/// maximum of two int32
inline HLBVH_HOST HLBVH_DEVICE int32 max(const int32 a, const int32 b) { return a > b ? a : b; }

/// minimum of two uint32
inline HLBVH_HOST HLBVH_DEVICE uint32 min(const uint32 a, const uint32 b) { return a < b ? a : b; }

/// maximum of two uint32
inline HLBVH_HOST HLBVH_DEVICE uint32 max(const uint32 a, const uint32 b) { return a > b ? a : b; }

/// quantize the float x in [0,1] to an integer [0,...,n[
inline HLBVH_HOST HLBVH_DEVICE uint32 quantize(const float x, const uint32 n)
{
	return (uint32)max( min( int32( x * float(n) ), int32(n-1) ), int32(0) );
}

template <typename T>
struct Field_traits
{
#ifdef __CUDACC__
	HLBVH_HOST HLBVH_DEVICE static T min() { return T(); }
    HLBVH_HOST HLBVH_DEVICE static T max() { return T(); }
#else
	static T min()
    {
        return std::numeric_limits<T>::is_integer ?
             std::numeric_limits<T>::min() :
            -std::numeric_limits<T>::max();
    }
	static T max() { return std::numeric_limits<T>::max(); }
#endif
};

#ifdef __CUDACC__
template <>
struct Field_traits<float>
{
	HLBVH_HOST HLBVH_DEVICE static float min() { return -float(1.0e+30f); }
    HLBVH_HOST HLBVH_DEVICE static float max() { return  float(1.0e+30f); }
};
template <>
struct Field_traits<double>
{
	HLBVH_HOST HLBVH_DEVICE static double min() { return -double(1.0e+30); }
    HLBVH_HOST HLBVH_DEVICE static double max() { return  double(1.0e+30); }
};
template <>
struct Field_traits<int32>
{
	HLBVH_HOST HLBVH_DEVICE static int32 min() { return -(1 << 30); }
    HLBVH_HOST HLBVH_DEVICE static int32 max() { return  (1 << 30); }
};
template <>
struct Field_traits<int64>
{
	HLBVH_HOST HLBVH_DEVICE static int64 min() { return -(int64(1) << 62); }
    HLBVH_HOST HLBVH_DEVICE static int64 max() { return  (int64(1) << 62); }
};
template <>
struct Field_traits<uint32>
{
	HLBVH_HOST HLBVH_DEVICE static uint32 min() { return 0; }
    HLBVH_HOST HLBVH_DEVICE static uint32 max() { return (1u << 31u); }
};
template <>
struct Field_traits<uint64>
{
	HLBVH_HOST HLBVH_DEVICE static uint64 min() { return 0; }
    HLBVH_HOST HLBVH_DEVICE static uint64 max() { return (uint64(1) << 63u); }
};
#endif

} // namespace pbgi
