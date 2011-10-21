#include <nih/basic/types.h>
#include <nih/basic/numbers.h>
#include <nih/time/timer.h>
#include <cuda_runtime_api.h>

#include <PBGI.h>
#include <thrust/device_vector.h>
#include <thrust/detail/device/cuda/arch.h>

using namespace nih;

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


template <uint32 K>
struct rw {};

template <>
struct rw<1>
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
struct rw<2>
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
struct rw<3>
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
struct rw<4>
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

template <uint32 CTA_SIZE, uint32 K, bool CHECKED>
__device__ void pbgi_block(
    typename Vec<float,K>::type*       smem,
    const uint32    src_begin,
    const uint32    src_end,
    const uint32    block_offset,
    const uint32    block_end,
    PBGI_state      state,
    PBGI_values     in_values,
    PBGI_values     out_values)
{
    typedef typename Vec<float,K>::type vec_type;

    const uint32 thread_id = threadIdx.x;

    vec_type* x;     x     = &smem[0];
    vec_type* y;     y     = &smem[0] + CTA_SIZE*1;
    vec_type* z;     z     = &smem[0] + CTA_SIZE*2;
    vec_type* nx;    nx    = &smem[0] + CTA_SIZE*3;
    vec_type* ny;    ny    = &smem[0] + CTA_SIZE*4;
    vec_type* nz;    nz    = &smem[0] + CTA_SIZE*5;
    vec_type* rec_r; rec_r = &smem[0] + CTA_SIZE*6;
    vec_type* rec_g; rec_g = &smem[0] + CTA_SIZE*7;
    vec_type* rec_b; rec_b = &smem[0] + CTA_SIZE*8;

    // read block in shared memory (not caring about overflows)
    if (thread_id < CTA_SIZE) // help the poor compiler reducing register pressure
    {
        if (CHECKED == false || block_offset + thread_id*K + K-1 < block_end)
        {
            x[ thread_id ]     = reinterpret_cast<vec_type*>(state.x + block_offset)[ thread_id ];
            y[ thread_id ]     = reinterpret_cast<vec_type*>(state.y + block_offset)[ thread_id  ];
            z[ thread_id ]     = reinterpret_cast<vec_type*>(state.z + block_offset)[ thread_id  ];
            nx[ thread_id ]    = reinterpret_cast<vec_type*>(state.nx + block_offset)[ thread_id  ];
            ny[ thread_id ]    = reinterpret_cast<vec_type*>(state.ny + block_offset)[ thread_id  ];
            nz[ thread_id ]    = reinterpret_cast<vec_type*>(state.nz + block_offset)[ thread_id  ];
            rec_r[ thread_id ] = reinterpret_cast<vec_type*>(out_values.r + block_offset)[ thread_id  ];
            rec_g[ thread_id ] = reinterpret_cast<vec_type*>(out_values.g + block_offset)[ thread_id  ];
            rec_b[ thread_id ] = reinterpret_cast<vec_type*>(out_values.b + block_offset)[ thread_id  ];
        }
        else
        {
            rw<K>::read<0>( x[ thread_id ], &state.x[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( y[ thread_id ], &state.y[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( z[ thread_id ], &state.z[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( nx[ thread_id ], &state.nx[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( ny[ thread_id ], &state.ny[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( nz[ thread_id ], &state.nz[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( rec_r[ thread_id ], &in_values.r[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( rec_g[ thread_id ], &in_values.g[ block_offset ], block_end - block_offset );
            rw<K>::read<0>( rec_b[ thread_id ], &in_values.b[ block_offset ], block_end - block_offset );
        }
    }

    // process block
    {
        __syncthreads();

        float* x;     x     = reinterpret_cast<float*>(&smem[0]);
        float* y;     y     = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*1);
        float* z;     z     = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*2);
        float* nx;    nx    = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*3);
        float* ny;    ny    = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*4);
        float* nz;    nz    = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*5);
        float* rec_r; rec_r = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*6);
        float* rec_g; rec_g = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*7);
        float* rec_b; rec_b = reinterpret_cast<float*>(&smem[0] + CTA_SIZE*8);

        // iterating over batches of N_SENDERS senders at a time
        #if __CUDA_ARCH__ < 200
        const uint32 N_SENDERS = 4;
        #else
        const uint32 N_SENDERS = 64;
        #endif

        __shared__ float s_x[N_SENDERS], s_y[N_SENDERS], s_z[N_SENDERS], s_nx[N_SENDERS], s_ny[N_SENDERS], s_nz[N_SENDERS], s_r[N_SENDERS], s_g[N_SENDERS], s_b[N_SENDERS];

        for (uint32 i = src_begin; i < src_end; i += N_SENDERS)
        {
            // load N_SENDERS senders in shared memory (issuing float4 loads)
            if (thread_id < N_SENDERS/4)
            {
                *reinterpret_cast<float4*>(s_x + thread_id*4)  = *reinterpret_cast<float4*>(state.x + i + thread_id*4);
                *reinterpret_cast<float4*>(s_y + thread_id*4)  = *reinterpret_cast<float4*>(state.y + i + thread_id*4);
                *reinterpret_cast<float4*>(s_z + thread_id*4)  = *reinterpret_cast<float4*>(state.z + i + thread_id*4);
                *reinterpret_cast<float4*>(s_nx + thread_id*4) = *reinterpret_cast<float4*>(state.nx + i + thread_id*4);
                *reinterpret_cast<float4*>(s_ny + thread_id*4) = *reinterpret_cast<float4*>(state.ny + i + thread_id*4);
                *reinterpret_cast<float4*>(s_nz + thread_id*4) = *reinterpret_cast<float4*>(state.nz + i + thread_id*4);
                *reinterpret_cast<float4*>(s_r + thread_id*4)  = *reinterpret_cast<float4*>(in_values.r + i + thread_id*4);
                *reinterpret_cast<float4*>(s_g + thread_id*4)  = *reinterpret_cast<float4*>(in_values.g + i + thread_id*4);
                *reinterpret_cast<float4*>(s_b + thread_id*4)  = *reinterpret_cast<float4*>(in_values.b + i + thread_id*4);
            }
            __syncthreads();

            // and compute their contribution to all K receivers per thread
            for (uint32 k = 0; k < K; ++k)
            {
                #if __CUDA_ARCH__ < 200
                #pragma unroll
                #endif
                for (uint32 c = 0; c < N_SENDERS; ++c)
                {
                    const float dx = s_x[c] - x[ thread_id + CTA_SIZE*k ];
                    const float dy = s_y[c] - y[ thread_id + CTA_SIZE*k ];
                    const float dz = s_z[c] - z[ thread_id + CTA_SIZE*k ];

                    const float d2 = dx*dx + dy*dy + dz*dz;

                    const float g1 = nx[ thread_id + CTA_SIZE*k ]*dx + ny[ thread_id + CTA_SIZE*k ]*dy + nz[ thread_id + CTA_SIZE*k ]*dz;
                    const float g2 = s_nx[c]*dx + s_ny[c]*dy + s_nz[c]*dz;

                    const float G_tmp = abs( g1*g2 ) * rcp( fmaxf( d2*d2, 1.0e-8f ) );
                    const float G = (i+c == block_offset + thread_id + CTA_SIZE*k) ? 1.0f : G_tmp; // if the sender is the receiver the weight should be 1
                        //__syncthreads(); // helps lowering register usage for wide K

                    rec_r[ thread_id + CTA_SIZE*k ] += s_r[c] * G;
                    rec_g[ thread_id + CTA_SIZE*k ] += s_g[c] * G;
                    rec_b[ thread_id + CTA_SIZE*k ] += s_b[c] * G;
                }
            }
        }

        __syncthreads();
    }

    // write block to global memory
    if (thread_id < CTA_SIZE) // help the poor compiler reducing register pressure
    {
        if (CHECKED == false || block_offset + thread_id*K + K-1 < block_end)
        {
            reinterpret_cast<vec_type*>(out_values.r + block_offset)[ thread_id ] = rec_r[thread_id];
            reinterpret_cast<vec_type*>(out_values.g + block_offset)[ thread_id ] = rec_g[thread_id];
            reinterpret_cast<vec_type*>(out_values.b + block_offset)[ thread_id ] = rec_b[thread_id];
        }
        else
        {
            rw<K>::write<0>( rec_r[ thread_id ], &out_values.r[ block_offset ], block_end - block_offset );
            rw<K>::write<0>( rec_g[ thread_id ], &out_values.g[ block_offset ], block_end - block_offset );
            rw<K>::write<0>( rec_b[ thread_id ], &out_values.b[ block_offset ], block_end - block_offset );
        }
    }
}

template <uint32 CTA_SIZE, uint32 K>
__global__  void pbgi_kernel(
    const uint32    n_points,
    const uint32    src_begin,
    const uint32    src_end,
    const uint32    rec_begin,
    const uint32    rec_end,
    const uint32    n_blocks,
    const uint32    n_elements_per_block,
    PBGI_state      state,
    PBGI_values     in_values,
    PBGI_values     out_values)
{
    const uint32 group_size   = CTA_SIZE * K;           // compile-time constant
    const uint32 block_id     = blockIdx.x;             // constant across CTA

    const uint32 block_begin = rec_begin + block_id * n_elements_per_block;              // constant across CTA
    const uint32 block_end   = nih::min( block_begin + n_elements_per_block, rec_end );       // constant across CTA

    //if (block_begin >= rec_end)
    //    return;

    typedef typename Vec<float,K>::type vec_type;

    __shared__ vec_type smem[CTA_SIZE*9];

    uint32 block_offset = block_begin;

    // process all the batches which don´t need overflow checks
    while (block_offset + group_size <= block_end)
    {
        pbgi_block<CTA_SIZE,K,false>(
            smem,
            src_begin,
            src_end,
            block_offset,
            block_end,
            state,
            in_values,
            out_values );

		block_offset += group_size;
	}

    // process the last batch
    if (block_offset < block_end)
    {
        pbgi_block<CTA_SIZE,K,true>(
            smem,
            src_begin,
            src_end,
            block_offset,
            block_end,
            state,
            in_values,
            out_values );
    }
}

// check for cuda runtime errors
void check_cuda_errors(const uint32 code)
{
    cudaError_t error = cudaGetLastError();
    if (error)
    {
        fprintf(stderr, "*** error (%u) ***\n  %s\n", code, cudaGetErrorString(error));
        exit(1);
    }
}

template <uint32 CTA_SIZE, uint32 K>
void test_pbgi_t(const uint32 n_points)
{
    cudaSetDeviceFlags( cudaDeviceMapHost );

    float* arena;
    //cudaHostAlloc( &arena, sizeof(float)*12*n_points, cudaHostAllocMapped );
    arena = (float*)malloc( sizeof(float)*12*n_points );

    float* ptr = arena;
    float* x     = ptr; ptr += n_points;
    float* y     = ptr; ptr += n_points;
    float* z     = ptr; ptr += n_points;
    float* nx    = ptr; ptr += n_points;
    float* ny    = ptr; ptr += n_points;
    float* nz    = ptr; ptr += n_points;
    float* in_r  = ptr; ptr += n_points;
    float* in_g  = ptr; ptr += n_points;
    float* in_b  = ptr; ptr += n_points;
    float* out_r = ptr; ptr += n_points;
    float* out_g = ptr; ptr += n_points;
    float* out_b = ptr; ptr += n_points;

    for (uint32 i = 0; i < n_points; ++i)
    {
        x[i] = float(i) / float(n_points);
        y[i] = 1.0f - float(i) / float(n_points);
        z[i] = sinf( float(i) / float(n_points) * 2.0f * float(M_PI) );

        nx[i] = 1.0f;
        ny[i] = 0.0f;
        nz[i] = 0.0f;

        in_r[i] = fabsf( sinf( float(i) / float(n_points) * 2.0f * float(M_PI) ) );
        in_g[i] = fabsf( sinf( float(i) / float(n_points) * 4.0f * float(M_PI) ) );
        in_b[i] = fabsf( sinf( float(i) / float(n_points) * 8.0f * float(M_PI) ) );

        out_r[i] = 0.0f;
        out_g[i] = 0.0f;
        out_b[i] = 0.0f;
    }

    PBGI_state  state;
    PBGI_values in_values;
    PBGI_values out_values;

    // compute the number of blocks we can launch to fill the machine
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(pbgi_kernel<CTA_SIZE,K>, CTA_SIZE, 0);
    const uint32 group_size = CTA_SIZE * K;
    const uint32 n_groups   = (n_points + group_size-1) / group_size;
    const size_t n_blocks   = nih::min( max_blocks, n_groups );

    // assume we can process 1 billion pairs per kernel launch avoiding timeout
    const float  pairs_per_kernel = 1.0e9f;

    // compute the number of receivers and the number of senders we want to process per block:
    // our strategy is to process 4 times as many receivers as we can process concurrently, and
    // set the number of senders correspondingly to reach our quota
    uint32 n_receivers = nih::min( n_blocks * group_size * 4u, n_points );
    uint32 n_senders   = nih::max( uint32( pairs_per_kernel / float(n_receivers) ), 1u );
    if (n_senders % 32)
        n_senders += 32 - (n_senders % 32);

    uint32 n_elements_per_block = (n_receivers + n_blocks-1) / n_blocks;
    if (n_elements_per_block % 4 != 0)
        n_elements_per_block += 4 - (n_elements_per_block % 4);

    float* cuda_arena;
    //cudaHostGetDevicePointer( &cuda_arena, arena, 0 );
    cudaMalloc( &cuda_arena, sizeof(float)*12*n_points );
    cudaMemcpy( cuda_arena, arena, sizeof(float)*12*n_points, cudaMemcpyHostToDevice );
    check_cuda_errors( 0 );

    ptr = cuda_arena;
    state.x      = ptr; ptr += n_points;
    state.y      = ptr; ptr += n_points;
    state.z      = ptr; ptr += n_points;
    state.nx     = ptr; ptr += n_points;
    state.ny     = ptr; ptr += n_points;
    state.nz     = ptr; ptr += n_points;
    in_values.r  = ptr; ptr += n_points;
    in_values.g  = ptr; ptr += n_points;
    in_values.b  = ptr; ptr += n_points;
    out_values.r = ptr; ptr += n_points;
    out_values.g = ptr; ptr += n_points;
    out_values.b = ptr; ptr += n_points;


    fprintf(stderr, "test pbgi gpu\n");
    fprintf(stderr,"  points       : %u\n", n_points);
    fprintf(stderr,"  block size   : %u\n", CTA_SIZE);
    fprintf(stderr,"  K            : %u\n", K);
    fprintf(stderr,"  n_blocks     : %u\n", n_blocks);
    fprintf(stderr,"  pts / block  : %u\n", n_elements_per_block);
    fprintf(stderr,"  src / kernel : %u\n", n_senders);
    fprintf(stderr,"  rec / kernel : %u\n", n_receivers);

    uint32 min_pairs = uint32(-1);

    nih::Timer timer;
    timer.start();
    {
        cudaThreadSynchronize();

        uint32 rec_begin = 0;
        while (rec_begin < n_points)
        {
            uint32 rec_end = nih::min( rec_begin + n_receivers, n_points );
            if (rec_end + n_receivers/2 > n_points) // if only a few points are missing,
                rec_end = n_points;                 // merge them in...

            n_elements_per_block = (rec_end - rec_begin + n_blocks-1) / n_blocks;
            if (n_elements_per_block % 4 != 0)
                n_elements_per_block += 4 - (n_elements_per_block % 4);

            uint32 sender_begin = 0;
            while (sender_begin < n_points)
            {
                uint32 sender_end = nih::min( sender_begin + n_senders, n_points );
                if (sender_end + n_senders/2 > n_points) // if only a few points are missing,
                    sender_end = n_points;               // merge them in...

                min_pairs = nih::min( min_pairs, (sender_end - sender_begin)*(rec_end - rec_begin) );

                pbgi_kernel<CTA_SIZE,K><<<n_blocks,CTA_SIZE>>>(
                    n_points,
                    sender_begin,
                    sender_end,
                    rec_begin,
                    rec_end,
                    n_blocks,
                    n_elements_per_block,
                    state,
                    in_values,
                    out_values );

                cudaThreadSynchronize();
                check_cuda_errors( 1 );

                sender_begin = sender_end;
            }

            rec_begin = rec_end;
        }

        cudaThreadSynchronize();
    }

    check_cuda_errors( 1 );

    timer.stop();

    cudaMemcpy( arena, cuda_arena, sizeof(float)*12*n_points, cudaMemcpyDeviceToHost );
    float sum = 0.0f;
    for (uint32 i = 0; i < n_points; ++i)
        sum += (out_r[i] + out_g[i] + out_b[i]) / 3.0f;

    fprintf(stderr,"  min pairs   : %u\n", min_pairs);
    fprintf(stderr,"  avg energy  : %.3f\n", sum / float(n_points));
    fprintf(stderr,"  time        : %.3f s\n", float(timer.seconds()));
    fprintf(stderr,"  pairs/s     : %.3f G\n", (float(n_points)/1000.0f)*(float(n_points)/1000.0f) / float(timer.seconds()*1000.0f));

    //cudaFreeHost( arena );
    cudaFree( cuda_arena );
    free(arena);
}

void test_pbgi(const uint32 n_points)
{
/*    test_pbgi_t<32,1>( n_points );
    test_pbgi_t<32,2>( n_points );
    test_pbgi_t<32,4>( n_points );
    test_pbgi_t<64,1>( n_points );
    test_pbgi_t<64,2>( n_points );
    test_pbgi_t<64,4>( n_points );
    test_pbgi_t<96,1>( n_points );
    test_pbgi_t<96,2>( n_points );
    test_pbgi_t<96,4>( n_points );
    test_pbgi_t<128,1>( n_points );
    test_pbgi_t<128,2>( n_points );
    test_pbgi_t<128,4>( n_points );
    test_pbgi_t<256,1>( n_points );
    test_pbgi_t<256,2>( n_points );*/
    test_pbgi_t<512,1>( n_points );
//    test_pbgi_t<512,2>( n_points );
}

} // namespace gpu
} // namespace pbgi