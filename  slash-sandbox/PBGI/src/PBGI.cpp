// PBGI.cpp : Defines the entry point for the console application.
//

#include <PBGI.h>

#include <nih/linalg/vector.h>
#include <nih/time/timer.h>
#include <vector>

#include <xmmintrin.h>	// Need this for SSE compiler intrinsics

using namespace nih;

namespace pbgi {

inline Vector4f abs(const Vector4f x)
{
    return Vector4f(
        fabsf(x[0]),
        fabsf(x[1]),
        fabsf(x[2]),
        fabsf(x[3]) );
}
inline Vector4f rcp(const Vector4f x)
{
    return Vector4f(
        1.0f/x[0],
        1.0f/x[1],
        1.0f/x[2],
        1.0f/x[3] );
}

namespace cpu {

struct PBGI_state_scope
{
    PBGI_state_scope(PBGI_state* data, const uint32 n_points) : m_data( data )
    {
        m_data->x = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->y = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->z = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->nx = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->ny = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->nz = (float*)_aligned_malloc(n_points * sizeof(float), 16);
    }
    ~PBGI_state_scope()
    {
        _aligned_free( m_data->x );
        _aligned_free( m_data->y );
        _aligned_free( m_data->z );
        _aligned_free( m_data->nx );
        _aligned_free( m_data->ny );
        _aligned_free( m_data->nz );
    }

    PBGI_state* m_data;
};
struct PBGI_values_scope
{
    PBGI_values_scope(PBGI_values* data, const uint32 n_points) : m_data( data )
    {
        m_data->r = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->g = (float*)_aligned_malloc(n_points * sizeof(float), 16);
        m_data->b = (float*)_aligned_malloc(n_points * sizeof(float), 16);
    }
    ~PBGI_values_scope()
    {
        _aligned_free( m_data->r );
        _aligned_free( m_data->g );
        _aligned_free( m_data->b );
    }

    PBGI_values* m_data;
};

void process_pbgi_block(
    const PBGI_state&   pbgi,
    const PBGI_values&  in_values,
    PBGI_values&        out_values,
    const uint32        block1_begin,
    const uint32        block1_end,
    const uint32        block2_begin,
    const uint32        block2_end)
{
    for (uint32 i = block1_begin; i < block1_end; ++i)
    {
        // process a single sender at a time, smeared across 4 lanes
        const Vector4f x1( pbgi.x[ i ] );
        const Vector4f y1( pbgi.y[ i ] );
        const Vector4f z1( pbgi.z[ i ] );

        const Vector4f nx1( pbgi.nx[ i ] );
        const Vector4f ny1( pbgi.ny[ i ] );
        const Vector4f nz1( pbgi.nz[ i ] );

        for (uint32 j = block2_begin; j < block2_end; j += 4)
        {
            // process 4 receivers in one go
            const Vector4f x2( &pbgi.x[ j ] );
            const Vector4f y2( &pbgi.y[ j ] );
            const Vector4f z2( &pbgi.z[ j ] );

            const Vector4f dx = x2 - x1;
            const Vector4f dy = y2 - y1;
            const Vector4f dz = z2 - z1;

            const Vector4f nx2( &pbgi.nx[ j ] );
            const Vector4f ny2( &pbgi.ny[ j ] );
            const Vector4f nz2( &pbgi.nz[ j ] );

            const Vector4f d2 = dx*dx + dy*dy + dz*dz;
            const Vector4f g1 = nx1*dx + ny1*dy + nz1*dz;
            const Vector4f g2 = nx2*dx + ny2*dy + nz2*dz;

            Vector4f inv_d2 = rcp( max( d2*d2, Vector4f(1.0e-8f) ) );
            if (i >= j && i < j+4)  // TODO: use masking here
            {
                const Vector4f masks[4] = {
                    Vector4f(0.0f,      1.0e20f,    1.0e20f,    1.0e20f),
                    Vector4f(1.0e20f,   0.0f,       1.0e20f,    1.0e20f),
                    Vector4f(1.0e20f,   1.0e20f,    0.0f,       1.0e20f),
                    Vector4f(1.0e20f,   1.0e20f,    1.0e20f,    0.0f),
                };
                inv_d2 = min( inv_d2, masks[i-j] );
            }

            const Vector4f G = abs( g1 * g2 ) * inv_d2;

            const Vector4f  src_r( in_values.r[ i ] );
            Vector4f*       rec_r = reinterpret_cast<Vector4f*>( &out_values.r[ j ] );
            *rec_r += src_r * G;

            const Vector4f  src_g( in_values.g[ i ] );
            Vector4f*       rec_g = reinterpret_cast<Vector4f*>( &out_values.g[ j ] );
            *rec_g += src_g * G;

            const Vector4f  src_b( in_values.b[ i ] );
            Vector4f*       rec_b = reinterpret_cast<Vector4f*>( &out_values.b[ j ] );
            *rec_b += src_b * G;
        }
    }
}

void process_pbgi_block_sse(
    const PBGI_state&   pbgi,
    const PBGI_values&  in_values,
    PBGI_values&        out_values,
    const uint32        block1_begin,
    const uint32        block1_end,
    const uint32        block2_begin,
    const uint32        block2_end)
{
    typedef __m128 sse_t;

    const sse_t zero = _mm_set_ps1( 0.0f );

    for (uint32 i = block1_begin; i < block1_end; ++i)
    {
        // process a single sender at a time, smeared across 4 lanes
        const sse_t x1 = _mm_set_ps1( pbgi.x[ i ] );
        const sse_t y1 = _mm_set_ps1( pbgi.y[ i ] );
        const sse_t z1 = _mm_set_ps1( pbgi.z[ i ] );

        const sse_t nx1 = _mm_set_ps1( pbgi.nx[ i ] );
        const sse_t ny1 = _mm_set_ps1( pbgi.ny[ i ] );
        const sse_t nz1 = _mm_set_ps1( pbgi.nz[ i ] );

        const sse_t min_d2 = _mm_set_ps1( 1.0e-8f );

        for (uint32 j = block2_begin; j < block2_end; j += 4)
        {
            // process 4 receivers in one go
            const sse_t x2 = _mm_load_ps( &pbgi.x[ j ] );
            const sse_t y2 = _mm_load_ps( &pbgi.y[ j ] );
            const sse_t z2 = _mm_load_ps( &pbgi.z[ j ] );

            const sse_t dx = _mm_sub_ps( x2, x1 );
            const sse_t dy = _mm_sub_ps( y2, y1 );
            const sse_t dz = _mm_sub_ps( z2, z1 );

            const sse_t nx2 = _mm_load_ps( &pbgi.nx[ j ] );
            const sse_t ny2 = _mm_load_ps( &pbgi.ny[ j ] );
            const sse_t nz2 = _mm_load_ps( &pbgi.nz[ j ] );

            const sse_t d2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx,dx),  _mm_mul_ps(dy,dy)),  _mm_mul_ps(dz,dz));
            const sse_t g1 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(nx1,dx), _mm_mul_ps(ny1,dy)), _mm_mul_ps(nz1,dz));
            const sse_t g2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(nx2,dx), _mm_mul_ps(ny2,dy)), _mm_mul_ps(nz2,dz));

            sse_t inv_d2 = _mm_rcp_ps( _mm_max_ps( _mm_mul_ps(d2,d2), min_d2 ) );
            if (i >= j && i < j+4)  // TODO: use masking here
            {
                const sse_t masks[4] = {
                    _mm_set_ps(0x00000000,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF),
                    _mm_set_ps(0xFFFFFFFF,0x00000000,0xFFFFFFFF,0xFFFFFFFF),
                    _mm_set_ps(0xFFFFFFFF,0xFFFFFFFF,0x00000000,0xFFFFFFFF),
                    _mm_set_ps(0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0x00000000)
                };
                inv_d2 = _mm_and_ps( inv_d2, masks[i-j] );
            }

            const sse_t G_num = _mm_mul_ps(g1,g2);
            const sse_t G     = _mm_mul_ps( _mm_max_ps( G_num, _mm_sub_ps( zero, G_num ) ), inv_d2 );

            const sse_t     src_r = _mm_set_ps1( in_values.r[ i ] );
            const sse_t     rec_r = _mm_load_ps( &out_values.r[ j ] );
            _mm_store_ps( &out_values.r[ j ], _mm_add_ps( rec_r, _mm_mul_ps( src_r, G ) ) );

            const sse_t     src_g = _mm_set_ps1( in_values.g[ i ] );
            const sse_t     rec_g = _mm_load_ps( &out_values.g[ j ] );
            _mm_store_ps( &out_values.g[ j ], _mm_add_ps( rec_g, _mm_mul_ps( src_g, G ) ) );

            const sse_t     src_b = _mm_set_ps1( in_values.b[ i ] );
            const sse_t     rec_b = _mm_load_ps( &out_values.b[ j ] );
            _mm_store_ps( &out_values.b[ j ], _mm_add_ps( rec_b, _mm_mul_ps( src_b, G ) ) );
        }
    }
}

float test_pbgi(const uint32 n_points, const uint32 block_size)
{
    PBGI_state  pbgi;
    PBGI_values in_values;
    PBGI_values out_values;

    PBGI_state_scope  pbgi_scope( &pbgi, n_points );
    PBGI_values_scope in_values_scope( &in_values, n_points );
    PBGI_values_scope out_values_scope( &out_values, n_points );

    for (uint32 i = 0; i < n_points; ++i)
    {
        pbgi.x[i] = float(i) / float(n_points);
        pbgi.y[i] = 1.0f - float(i) / float(n_points);
        pbgi.z[i] = sinf( float(i) / float(n_points) * 2.0f * float(M_PI) );

        pbgi.nx[i] = 1.0f;
        pbgi.ny[i] = 0.0f;
        pbgi.nz[i] = 0.0f;

        in_values.r[i] = fabsf( sinf( float(i) / float(n_points) * 2.0f * float(M_PI) ) );
        in_values.g[i] = fabsf( sinf( float(i) / float(n_points) * 4.0f * float(M_PI) ) );
        in_values.b[i] = fabsf( sinf( float(i) / float(n_points) * 8.0f * float(M_PI) ) );

        out_values.r[i] = 0.0f;
        out_values.g[i] = 0.0f;
        out_values.b[i] = 0.0f;
    }

    Timer timer;
    timer.start();

    const uint32 n_blocks = (n_points + block_size-1) / block_size;

    for (uint32 block1_index = 0; block1_index < n_blocks; ++block1_index)
    {
        const uint32 block1_begin = block1_index * block_size;
        const uint32 block1_end   = std::min( block1_begin + block_size, n_points );

        for (uint32 block2_index = 0; block2_index < n_blocks; ++block2_index)
        {
            const uint32 block2_begin = block2_index * block_size;
            const uint32 block2_end   = std::min( block2_begin + block_size, n_points );

            process_pbgi_block_sse(
                pbgi,
                in_values,
                out_values,
                block1_begin,
                block1_end,
                block2_begin,
                block2_end );
        }
    }

    timer.stop();

    float sum = 0.0f;
    for (uint32 i = 0; i < n_points; ++i)
        sum += (out_values.r[i] + out_values.g[i] + out_values.b[i]) / 3.0f;

    fprintf(stderr,"test pbgi cpu\n");
    fprintf(stderr,"  points     : %u\n", n_points);
    fprintf(stderr,"  block size : %u\n", block_size);
    fprintf(stderr,"  avg energy  : %.3f\n", sum / float(n_points));
    fprintf(stderr,"  time       : %.3f s\n", float(timer.seconds()));
    fprintf(stderr,"  pairs/s    : %.3f M\n", (float(n_points)/1000.0f)*(float(n_points)/1000.0f) / float(timer.seconds()));
    return float(n_points) / timer.seconds();
}

} // namespace cpu
} // namespace pbgi

int main(int agc, char* argv[])
{
    using namespace pbgi;

    for (uint32 n_points = 32*1024; n_points <= 1024*1024; n_points *= 2)
        gpu::test_pbgi( n_points );

    for (uint32 n_points = 16*1024; n_points <= 1024*1024; n_points *= 2)
    {
        for (uint32 block_size = 32; block_size <= std::min( 2048u, n_points ); block_size *= 2)
            cpu::test_pbgi( n_points, block_size );
    }
	return 0;
}

