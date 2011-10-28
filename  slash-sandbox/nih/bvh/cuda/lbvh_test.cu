/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <nih/bvh/cuda/lbvh_builder.h>
#include <nih/sampling/random.h>
#include <nih/time/timer.h>
#include <nih/basic/cuda_domains.h>
#include <nih/bvh/bvh_tree.h>
#include <nih/tree/cuda/reduce.h>

namespace nih {

struct bbox_functor
{
    NIH_HOST_DEVICE Bbox4f operator() (
        const Vector4f op1,
        const Vector4f op2) const
    {
        Bbox4f result;
        result.insert( op1 );
        result.insert( op2 );
        return result;
    }
    NIH_HOST_DEVICE Bbox4f operator() (
        const Bbox4f op1,
        const Bbox4f op2) const
    {
        Bbox4f result;
        result.insert( op1 );
        result.insert( op2 );
        return result;
    }
};

void lbvh_test()
{
    fprintf(stderr, "lbvh test... started\n");

    const uint32 n_points = 4*1024*1024;
    const uint32 n_tests = 100;

    thrust::host_vector<Vector4f> h_points( n_points );

    Random random;
    for (uint32 i = 0; i < n_points; ++i)
        h_points[i] = Vector4f( random.next(), random.next(), random.next(), 1.0f );

    thrust::device_vector<Vector4f> d_points( h_points );
    thrust::device_vector<Vector4f> d_unsorted_points( h_points );

    thrust::device_vector<Bvh_node> bvh_nodes;
    thrust::device_vector<uint2>    bvh_leaves;
    thrust::device_vector<uint32>   bvh_index;

    cuda::LBVH_builder<uint64> builder( bvh_nodes, bvh_leaves, bvh_index );

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    float time = 0.0f;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        d_points = d_unsorted_points;
        cudaThreadSynchronize();

        float dtime;
        cudaEventRecord( start, 0 );

        builder.build(
            Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),
            d_points.begin(),
            d_points.end(),
            16u );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    fprintf(stderr, "lbvh test... done\n");
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
    fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );

    fprintf(stderr, "  nodes  : %u\n", builder.m_node_count );
    fprintf(stderr, "  leaves : %u\n", builder.m_leaf_count );
    for (uint32 level = 0; level < 60; ++level)
        fprintf(stderr, "  level %u : %u nodes\n", level, builder.m_levels[level+1] - builder.m_levels[level] );

    fprintf(stderr, "lbvh bbox reduction test... started\n");

    Bvh_tree<breadth_first_tree,device_domain> bvh(
        thrust::raw_pointer_cast( &bvh_nodes.front() ),
        builder.m_leaf_count,
        thrust::raw_pointer_cast( &bvh_leaves.front() ),
        60u,
        builder.m_levels );

    thrust::device_vector<Bbox4f> d_leaf_bboxes( builder.m_leaf_count );
    thrust::device_vector<Bbox4f> d_node_bboxes( builder.m_node_count );

    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    time = 0.0f;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        float dtime;
        cudaEventRecord( start, 0 );

        cuda::tree_reduce(
            bvh,
            thrust::raw_pointer_cast( &d_points.front() ),
            thrust::raw_pointer_cast( &d_leaf_bboxes.front() ),
            thrust::raw_pointer_cast( &d_node_bboxes.front() ),
            bbox_functor(),
            Bbox4f() );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    fprintf(stderr, "lbvh bbox reduction test... done\n");
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
    fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );
}

} // namespace nih

