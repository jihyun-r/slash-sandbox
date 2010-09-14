#pragma once

#include <assert.h>

#ifdef __CUDACC__
    #define HLBVH_HOST   __host__
    #define HLBVH_DEVICE __device__
#else
    #define HLBVH_HOST
    #define HLBVH_DEVICE
#endif

#define PBGI_API_CS
#define PBGI_API_SS

namespace pbgi {

typedef unsigned char		uint8;
typedef char				int8;
typedef unsigned short		uint16;
typedef short				int16;
typedef unsigned int		uint32;
typedef int					int32;
typedef unsigned long long	uint64;
typedef long long			int64;

#define FORCE_INLINE __forceinline

template <typename Out, typename In>
union BinaryCast
{
    In  in;
    Out out;
};

template <typename Out, typename In>
Out binary_cast(const In in)
{
    BinaryCast<Out,In> inout;
    inout.in = in;
    return inout.out;
}

} // namespace pbgi
