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
 *     documentation and/or far materials provided with the distribution.
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

#include <thrust/detail/backend/dereference.h>

namespace nih {
namespace cuda {

namespace kd_knn {

template <typename VectorType, typename PointIterator>
__device__ void lookup(
    const VectorType        query,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const PointIterator     kd_points,
    Kd_knn<3>::Result*      results)
{
    //
    // 1-st pass: find the leaf containing the query point and compute an upper bound
    // on the search distance
    //

    uint32 idx;
    float  dist2 = 1.0e16f;

    // start from the root node
    uint32 node_index = 0;

    // keep track of which leaf we visited
    uint32 first_leaf = 0;

    while (1)
    {
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf())
        {
            // find the closest neighbor in this leaf
            const uint2 leaf = kd_leaves[ node.get_leaf_index() ];
            for (uint32 i = leaf.x; i < leaf.y; ++i)
            {
                const VectorType delta = kd_points[i] - query;
                const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                if (dist2 > d2)
                {
                    dist2 = d2;
                    idx   = i;
                }
            }

            // keep track of which leaf we found
            first_leaf = node_index;
            break;
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            node_index =
                split_dim == 0 ? node.get_child_offset() + (query[0] < split_plane ? 0u : 1u) :
                split_dim == 1 ? node.get_child_offset() + (query[1] < split_plane ? 0u : 1u) :
                                 node.get_child_offset() + (query[2] < split_plane ? 0u : 1u);
        }
    }

    //
    // 2-nd pass: visit the tree with a stack and careful pruning
    //

    int32  stackp = 1;
    float4 stack[32];

    // place a sentinel node in the stack
    stack[0] = make_float4( 0.0f, 0.0f, 0.0f, binary_cast<float>(uint32(-1)) );

    // start from the root node
    node_index = 0;

    float3 cdist = make_float3( 0.0f, 0.0f, 0.0f );

    while (node_index != uint32(-1))
    {
        const Kd_node node = kd_nodes[ node_index ];

        if (node.is_leaf())
        {
            if (first_leaf != node_index)
            {
                // find the closest neighbor in this leaf
                const uint2 leaf = kd_leaves[ node.get_leaf_index() ];
                for (uint32 i = leaf.x; i < leaf.y; ++i)
                {
                    const VectorType delta = kd_points[i] - query;
                    const float d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                    if (dist2 > d2)
                    {
                        dist2 = d2;
                        idx   = i;
                    }
                }
            }

            // pop the next node from the stack
            while (stackp > 0)
            {
                const float4 stack_node = stack[ --stackp ];
                node_index = binary_cast<uint32>( stack_node.w );
                cdist      = make_float3( stack_node.x, stack_node.y, stack_node.z );

                if (cdist.x*cdist.x + cdist.y*cdist.y + cdist.z*cdist.z > dist2)
                    continue;
            }
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            const float split_dist = 
                split_dim == 0 ? (query[0] - split_plane) :
                split_dim == 1 ? (query[1] - split_plane) :
                                 (query[2] - split_plane);

            const uint32 select = split_dist <= 0.0f ? 0u : 1u;

            node_index = node.get_child_offset() + select;

            // compute the vector distance to the far node
            float3 cdist_far = cdist;
            if (split_dim == 0)      cdist_far.x = split_dist;
            else if (split_dim == 1) cdist_far.y = split_dist;
            else                     cdist_far.z = split_dist;

            // check whether we should push the far node on the stack
            const float dist_far2 = cdist_far.x*cdist_far.x +
                                    cdist_far.y*cdist_far.y +
                                    cdist_far.z*cdist_far.z;

            if (dist_far2 <= dist2)
            {
                stack[ stackp++ ] = make_float4(
                    cdist_far.x,
                    cdist_far.y,
                    cdist_far.z,
                    binary_cast<float>( node.get_child_offset() + 1u - select ) );
            }
        }
    }

    // write the result
    results->index = idx;
    results->dist2 = dist2;
}

template <typename QueryIterator, typename PointIterator>
__global__ void lookup_kernel(
    const uint32                    n_points,
    const QueryIterator             points_begin,
    const Kd_node*                  kd_nodes,
    const uint2*                    kd_ranges,
    const uint2*                    kd_leaves,
    const PointIterator             kd_points,
    Kd_knn<3>::Result*              results)
{
    const uint32 grid_size = gridDim.x * blockDim.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * blockDim.x;
                base_idx < n_points;
                base_idx += grid_size)
    {
        const uint32 index = threadIdx.x + base_idx;
        if (index >= n_points)
            return;

        lookup(
            thrust::detail::backend::dereference( points_begin + index ),
            kd_nodes,
            kd_ranges,
            kd_leaves,
            kd_points,
            results + index );
    }
}

} // namespace knn

// perform a k-nn lookup for a set of query points
//
// \param points_begin     beginning of the query point sequence
// \param points_end       end of the query point sequence
// \param kd_nodes         k-d tree nodes
// \param kd_ranges        k-d tree node ranges
// \param kd_leaves        k-d tree leaves
// \param kd_points        k-d tree points
template <typename QueryIterator, typename PointIterator>
void Kd_knn<3>::run(
    const QueryIterator             points_begin,
    const QueryIterator             points_end,
    const Kd_node*                  kd_nodes,
    const uint2*                    kd_ranges,
    const uint2*                    kd_leaves,
    const PointIterator             kd_points,
    Result*                         results)
{
    const uint32 n_points = uint32( points_end - points_begin );

    const uint32 BLOCK_SIZE = 128;
    const size_t max_blocks = thrust::detail::backend::cuda::arch::max_active_blocks(
        kd_knn::lookup_kernel<QueryIterator,PointIterator>, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (n_points + BLOCK_SIZE-1) / BLOCK_SIZE );

    kd_knn::lookup_kernel<<<n_blocks,BLOCK_SIZE>>> (
        n_points,
        points_begin,
        kd_nodes,
        kd_ranges,
        kd_leaves,
        kd_points,
        results );

    cudaThreadSynchronize();
}

} // namespace cuda
} // namespace nih

