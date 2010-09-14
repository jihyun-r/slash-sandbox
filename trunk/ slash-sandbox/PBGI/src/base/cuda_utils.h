#pragma once

#include <base/types.h>
#include <cuda_runtime_api.h>

namespace pbgi {
namespace gpu {

template <typename T, uint32 K>
struct Vec {};

template <> struct Vec<float,1> { typedef float  type; __device__ static float  make(const float a) {return a;} };
template <> struct Vec<float,2> { typedef float2 type; __device__ static float2 make(const float a) {return make_float2(a,a);} };
template <> struct Vec<float,3> { typedef float3 type; __device__ static float3 make(const float a) {return make_float3(a,a,a);} };
template <> struct Vec<float,4> { typedef float4 type; __device__ static float4 make(const float a) {return make_float4(a,a,a,a);} };

template <> struct Vec<uint32,1> { typedef uint32 type; };
template <> struct Vec<uint32,2> { typedef uint2 type; };
template <> struct Vec<uint32,3> { typedef uint3 type; };
template <> struct Vec<uint32,4> { typedef uint4 type; };

__device__ inline float2 operator*(const float a, const float2 b) { return make_float2( a*b.x, a*b.y ); }
__device__ inline float3 operator*(const float a, const float3 b) { return make_float3( a*b.x, a*b.y, a*b.z ); }
__device__ inline float4 operator*(const float a, const float4 b) { return make_float4( a*b.x, a*b.y, a*b.z, a*b.w ); }

__device__ inline float2 operator*(const float2 a, const float2 b) { return make_float2( a.x*b.x, a.y*b.y ); }
__device__ inline float3 operator*(const float3 a, const float3 b) { return make_float3( a.x*b.x, a.y*b.y, a.z*b.z ); }
__device__ inline float4 operator*(const float4 a, const float4 b) { return make_float4( a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w ); }

__device__ inline float2 operator-(const float a, const float2 b) { return make_float2( a-b.x, a-b.y ); }
__device__ inline float3 operator-(const float a, const float3 b) { return make_float3( a-b.x, a-b.y, a-b.z ); }
__device__ inline float4 operator-(const float a, const float4 b) { return make_float4( a-b.x, a-b.y, a-b.z, a-b.w ); }

__device__ inline float2 operator-(const float2 a, const float2 b) { return make_float2( a.x-b.x, a.y-b.y ); }
__device__ inline float3 operator-(const float3 a, const float3 b) { return make_float3( a.x-b.x, a.y-b.y, a.z-b.z ); }
__device__ inline float4 operator-(const float4 a, const float4 b) { return make_float4( a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w ); }

__device__ inline float2 operator-(const float2 a, const float b) { return make_float2( a.x-b, a.y-b ); }
__device__ inline float3 operator-(const float3 a, const float b) { return make_float3( a.x-b, a.y-b, a.z-b ); }
__device__ inline float4 operator-(const float4 a, const float b) { return make_float4( a.x-b, a.y-b, a.z-b, a.w-b ); }

__device__ inline float2 operator+(const float2 a, const float2 b) { return make_float2( a.x+b.x, a.y+b.y ); }
__device__ inline float3 operator+(const float3 a, const float3 b) { return make_float3( a.x+b.x, a.y+b.y, a.z+b.z ); }
__device__ inline float4 operator+(const float4 a, const float4 b) { return make_float4( a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w ); }

__device__ inline float2& operator+=(float2& a, const float2 b) { a.x += b.x; a.y += b.y; return a; }
__device__ inline float3& operator+=(float3& a, const float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
__device__ inline float4& operator+=(float4& a, const float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }

__device__ inline float  abs(const float  a) { return fabsf(a); }
__device__ inline float2 abs(const float2 a) { return make_float2( fabsf(a.x), fabsf(a.y) ); }
__device__ inline float3 abs(const float3 a) { return make_float3( fabsf(a.x), fabsf(a.y), fabsf(a.z) ); }
__device__ inline float4 abs(const float4 a) { return make_float4( fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w) ); }

__device__ inline float  rcp(const float  a) { return 1.0f / a; }
__device__ inline float2 rcp(const float2 a) { return make_float2( rcp(a.x), rcp(a.y) ); }
__device__ inline float3 rcp(const float3 a) { return make_float3( rcp(a.x), rcp(a.y), rcp(a.z) ); }
__device__ inline float4 rcp(const float4 a) { return make_float4( rcp(a.x), rcp(a.y), rcp(a.z), rcp(a.w) ); }


template <bool CHECKED, uint32 K>
struct rw
{
    typedef typename Vec<float,K>::type vec_type;

    template <uint32 OFFSET>
    __device__ static void read(
        vec_type&                       item,
        const float* __restrict__       in_keys,
        const uint32                    limit)
    {
        item = reinterpret_cast<const vec_type*>(in_keys)[ threadIdx.x + OFFSET ];
    }
    template <uint32 OFFSET>
    __device__ static void write(
        const vec_type                  item,
        float* __restrict__             out_keys,
        const uint32                    limit)
    {
        reinterpret_cast<vec_type*>(out_keys)[ threadIdx.x + OFFSET ] = item;
    }

};

template <>
struct rw<true, 1>
{
    typedef float vec_type;

    template <uint32 OFFSET>
    __device__ static void read(
        vec_type&                       item,
        const float* __restrict__       in_keys,
        const uint32                    limit)
    {
        item = (threadIdx.x + OFFSET < limit) ? in_keys[ threadIdx.x + OFFSET ] : 0;
    }
    template <uint32 OFFSET>
    __device__ static void write(
        const vec_type                  item,
        float* __restrict__             out_keys,
        const uint32                    limit)
    {
        if (threadIdx.x + OFFSET < limit) out_keys[ threadIdx.x + OFFSET ] = item;
    }
};
template <>
struct rw<true, 2>
{
    typedef typename Vec<float,2>::type vec_type;

    template <uint32 OFFSET>
    __device__ static void read(
        vec_type&                       item,
        const float* __restrict__       in_keys,
        const uint32                    limit)
    {
        item.x = (threadIdx.x*2 + OFFSET*2   < limit) ? in_keys[ threadIdx.x*2 + OFFSET*2   ] : 0;
        item.y = (threadIdx.x*2 + OFFSET*2+1 < limit) ? in_keys[ threadIdx.x*2 + OFFSET*2+1 ] : 0;
    }
    template <uint32 OFFSET>
    __device__ static void write(
        const vec_type                  item,
        float* __restrict__             out_keys,
        const uint32                    limit)
    {
        if (threadIdx.x*2 + OFFSET*2   < limit) out_keys[ threadIdx.x*2 + OFFSET*2   ] = item.x;
        if (threadIdx.x*2 + OFFSET*2+1 < limit) out_keys[ threadIdx.x*2 + OFFSET*2+1 ] = item.y;
    }
};
template <>
struct rw<true, 3>
{
    typedef typename Vec<float,3>::type vec_type;

    template <uint32 OFFSET>
    __device__ static void read(
        vec_type&                       item,
        const float* __restrict__       in_keys,
        const uint32                    limit)
    {
        item.x = (threadIdx.x*3 + OFFSET*3   < limit) ? in_keys[ threadIdx.x*3 + OFFSET*3   ] : 0;
        item.y = (threadIdx.x*3 + OFFSET*3+1 < limit) ? in_keys[ threadIdx.x*3 + OFFSET*3+1 ] : 0;
        item.z = (threadIdx.x*3 + OFFSET*3+2 < limit) ? in_keys[ threadIdx.x*3 + OFFSET*3+2 ] : 0;
    }
    template <uint32 OFFSET>
    __device__ static void write(
        const vec_type                  item,
        float* __restrict__             out_keys,
        const uint32                    limit)
    {
        if (threadIdx.x*3 + OFFSET*3   < limit) out_keys[ threadIdx.x*3 + OFFSET*3   ] = item.x;
        if (threadIdx.x*3 + OFFSET*3+1 < limit) out_keys[ threadIdx.x*3 + OFFSET*3+1 ] = item.y;
        if (threadIdx.x*3 + OFFSET*3+2 < limit) out_keys[ threadIdx.x*3 + OFFSET*3+2 ] = item.z;
    }
};
template <>
struct rw<true, 4>
{
    typedef typename Vec<float,4>::type vec_type;

    template <uint32 OFFSET>
    __device__ static void read(
        vec_type&                       item,
        const float* __restrict__       in_keys,
        const uint32                    limit)
    {
        item.x = (threadIdx.x*4 + 4*OFFSET   < limit) ? in_keys[ threadIdx.x*4 + OFFSET*4   ] : 0;
        item.y = (threadIdx.x*4 + 4*OFFSET+1 < limit) ? in_keys[ threadIdx.x*4 + OFFSET*4+1 ] : 0;
        item.z = (threadIdx.x*4 + 4*OFFSET+2 < limit) ? in_keys[ threadIdx.x*4 + OFFSET*4+2 ] : 0;
        item.w = (threadIdx.x*4 + 4*OFFSET+3 < limit) ? in_keys[ threadIdx.x*4 + OFFSET*4+3 ] : 0;
    }
    template <uint32 OFFSET>
    __device__ static void write(
        const vec_type                  item,
        float* __restrict__             out_keys,
        const uint32                    limit)
    {
        if (threadIdx.x*4 + 4*OFFSET   < limit) out_keys[ threadIdx.x*4 + OFFSET*4   ] = item.x;
        if (threadIdx.x*4 + 4*OFFSET+1 < limit) out_keys[ threadIdx.x*4 + OFFSET*4+1 ] = item.y;
        if (threadIdx.x*4 + 4*OFFSET+2 < limit) out_keys[ threadIdx.x*4 + OFFSET*4+2 ] = item.z;
        if (threadIdx.x*4 + 4*OFFSET+3 < limit) out_keys[ threadIdx.x*4 + OFFSET*4+3 ] = item.w;
    }
};

} // namespace gpu
} // namespace pbgi