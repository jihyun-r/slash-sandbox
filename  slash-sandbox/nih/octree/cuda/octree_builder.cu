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

#include <nih/octree/cuda/octree_builder.h>
#include <nih/bintree/cuda/bintree_context.h>
#include <nih/bintree/cuda/bintree_gen.h>
#include <nih/basic/cuda/scan.h>
#include <nih/basic/utils.h>
#include <nih/bits/morton.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

namespace nih {

namespace octree_builder { // anonymous namespace

typedef cuda::Bintree_gen_context::Split_task Split_task;
typedef Bintree_node Kd_node;

// Helper class to collect the 8 children of a given node from
// a binary kd-tree structure.
// The class implements template based compile-time recursion.
template <uint32 LEVEL, uint32 STRIDE>
struct Octree_collector
{
    static NIH_HOST_DEVICE void find_children(
        const uint32    node_index,
        const Kd_node*  nodes,
        uint32&         counter,
        uint32&         leaf_counter,
        uint32*         children,
        uint32&         bitmask,
        const uint32    octant = 0)
    {
        const Kd_node node = nodes[ node_index ];

        if (LEVEL == 0)
            bitmask = 0;

        if (node.is_leaf())
        {
            // here we have some trouble... this guy is likely not
            // fitting within an octant of his parent.
            // Which octant do we assign this guy to?
            // Let's say the minimum...
            children[ STRIDE * counter++ ] = node_index;
            bitmask &= 1u << (octant << (3 - LEVEL));
            leaf_counter++;
        }
        else
        {
            // traverse the children in Morton order: preserving
            // the order is key here, as we want the output counter
            // to match the bitmask pop-counts.
            const bool active0 = node.is_active(0);
            const bool active1 = node.is_active(1);

            if (active0)
            {
                Octree_collector<LEVEL+1,STRIDE>::find_children(
                    node.get_child(0),
                    nodes,
                    counter,
                    leaf_counter,
                    children,
                    bitmask,
                    octant * 2 );
            }
            if (active1)
            {
                Octree_collector<LEVEL+1,STRIDE>::find_children(
                    node.get_child(1),
                    nodes,
                    counter,
                    leaf_counter,
                    children,
                    bitmask,
                    octant * 2 + 1 );
            }
        }
    }
};
// Terminal node of Octree_collector's compile-time recursion.
template <uint32 STRIDE>
struct Octree_collector<3,STRIDE>
{
    static NIH_HOST_DEVICE void find_children(
        const uint32    node_index,
        const Kd_node*  nodes,
        uint32&         counter,
        uint32&         leaf_counter,
        uint32*         children,
        uint32&         bitmask,
        const uint32    octant)
    {
        // we got to one of the octants
        children[ STRIDE * counter++ ] = node_index;
        bitmask &= 1u << octant;

        const Kd_node node = nodes[ node_index ];
        if (node.is_leaf())
            leaf_counter++;
    }
};

// collect octants from a kd-tree
template <uint32 BLOCK_SIZE>
__global__ void collect_octants_kernel(
    const uint32        grid_size,
    const Kd_node*      kd_nodes,
    const uint32        in_tasks_count,
    const Split_task*   in_tasks,
    uint32*             out_tasks_count,
    Split_task*         out_tasks,
    uint32*             out_nodes_count,
    Octree_node_base*   out_nodes)
{
    const uint32 LOG_WARP_SIZE = 5;
    const uint32 WARP_SIZE = 1u << LOG_WARP_SIZE;

    volatile __shared__ uint32 warp_offset[ BLOCK_SIZE >> LOG_WARP_SIZE ];

    const uint32 warp_tid = threadIdx.x & (WARP_SIZE-1);
    const uint32 warp_id  = threadIdx.x >> LOG_WARP_SIZE;

    __shared__ uint32 sm_children[ BLOCK_SIZE * 8 ];

    volatile __shared__ uint32 sm_red[ BLOCK_SIZE * 2 ];
    volatile uint32* warp_red = sm_red + WARP_SIZE * 2 * warp_id;

    uint32* children = sm_children + threadIdx.x;

    // loop through all logical blocks associated to this physical one
    for (uint32 base_idx = blockIdx.x * BLOCK_SIZE;
        base_idx < in_tasks_count;
        base_idx += grid_size)
    {
        const uint32 task_id = threadIdx.x + base_idx;

        uint32 node;

        uint32 child_counter = 0;
        uint32 leaf_counter  = 0;
        uint32 bitmask       = 0;

        // check if the task id is in range, and if so try to collect its treelet
        if (task_id < in_tasks_count)
        {
            const Split_task in_task = in_tasks[ task_id ];

            node = in_task.m_node;

            Octree_collector<0,BLOCK_SIZE>::find_children(
                in_task.m_input,
                kd_nodes,
                child_counter,
                leaf_counter,
                children, 
                bitmask );
        }

        // allocate output nodes, output tasks, and write all leaves
        {
            uint32 task_counter = child_counter - leaf_counter;

            uint32 node_offset = cuda::alloc( child_counter, out_nodes_count, warp_tid, warp_red, warp_offset + warp_id );
            uint32 task_offset = cuda::alloc( task_counter,  out_tasks_count, warp_tid, warp_red, warp_offset + warp_id );

            // write the parent node
            if (task_id < in_tasks_count)
                out_nodes[ node ] = Octree_node_base( bitmask, node_offset );

            // write out all outputs
            for (uint32 i = 0; i < child_counter; ++i)
            {
                const uint32  kd_node_index = children[ i * BLOCK_SIZE ];
                const Kd_node kd_node       = kd_nodes[ kd_node_index ];

                if (kd_node.is_leaf() == false)
                    out_tasks[ task_offset++ ] = Split_task( node_offset, 0, 0, kd_node_index );
                else
                    out_nodes[ node_offset ] = Octree_node_base( kd_node.get_child_offset() );

                node_offset++;
            }
        }
    }
}

// collect octants from a kd-tree
void collect_octants(
    const Kd_node*      kd_nodes,
    const uint32        in_tasks_count,
    const Split_task*   in_tasks,
    uint32*             out_tasks_count,
    Split_task*         out_tasks,
    uint32*             out_nodes_count,
    Octree_node_base*   out_nodes)
{
    const uint32 BLOCK_SIZE = 128;
    const size_t max_blocks = thrust::detail::device::cuda::arch::max_active_blocks(collect_octants_kernel<BLOCK_SIZE>, BLOCK_SIZE, 0);
    const size_t n_blocks   = nih::min( max_blocks, (in_tasks_count + BLOCK_SIZE-1) / BLOCK_SIZE );
    const size_t grid_size  = n_blocks * BLOCK_SIZE;

    collect_octants_kernel<BLOCK_SIZE> <<<n_blocks,BLOCK_SIZE>>> (
        grid_size,
        kd_nodes,
        in_tasks_count,
        in_tasks,
        out_tasks_count,
        out_tasks,
        out_nodes_count,
        out_nodes );

    cudaThreadSynchronize();
}

} // namespace octree_builder

// build an octree given a set of points
void Octree_builder::build(
    const Bbox3f                           bbox,
    const thrust::device_vector<Vector4f>& points,
    const uint32                           max_leaf_size)
{
    typedef cuda::Bintree_gen_context::Split_task Split_task;

    const uint32 n_points = uint32( points.size() );

    m_bbox = bbox;
    need_space( m_codes, n_points );
    need_space( *m_index, n_points );
    need_space( *m_octree, (n_points / max_leaf_size) * 8 );
    need_space( *m_leaves, n_points );

    // compute the Morton code for each point
    thrust::transform(
        points.begin(),
        points.begin() + n_points,
        m_codes.begin(),
        morton_functor( bbox ) );

    // setup the point indices, from 0 to n_points-1
    thrust::copy(
        thrust::counting_iterator<uint32>(0),
        thrust::counting_iterator<uint32>(0) + n_points,
        m_index->begin() );

    // sort the indices by Morton code
    // TODO: use Duane's library directly here... this is doing shameful allocations!
    thrust::sort_by_key(
        m_codes.begin(),
        m_codes.begin() + n_points,
        m_index->begin() );

    // generate a kd-tree
    cuda::Bintree_context tree( m_kd_nodes, *m_leaves );

    cuda::generate(
        m_kd_context,
        n_points,
        thrust::raw_pointer_cast( &m_codes.front() ),
        30u,
        max_leaf_size,
        true,
        tree );

    m_leaf_count = m_kd_context.m_leaves;

    // start building the octree
    thrust::device_vector<Split_task>* m_task_queues = m_kd_context.m_task_queues;
    thrust::device_vector<uint32>&     m_counters    = m_kd_context.m_counters;

    need_space( m_task_queues[0], n_points );
    need_space( m_task_queues[1], n_points );

    Split_task* task_queues[2] = {
        thrust::raw_pointer_cast( &(m_task_queues[0]).front() ),
        thrust::raw_pointer_cast( &(m_task_queues[1]).front() )
    };

    uint32 in_queue  = 0;
    uint32 out_queue = 1;

    // convert the kd-tree into an octree.
    need_space( m_counters, 3 );
    m_counters[ in_queue ]  = 1;
    m_counters[ out_queue ] = 0;
    m_counters[ 2 ]         = 0; // output node counter

    m_task_queues[ in_queue ][0] = Split_task( 0, 0, 0, 0 );

    // loop until there's tasks left in the input queue
    while (m_counters[ in_queue ])
    {
        need_space( *m_octree, m_counters[2] + m_counters[ in_queue ]*8 );

        // clear the output queue
        m_counters[ out_queue ] = 0;

        octree_builder::collect_octants(
            thrust::raw_pointer_cast( &m_kd_nodes.front() ),
            m_counters[ in_queue ],
            task_queues[ in_queue ],
            thrust::raw_pointer_cast( &m_counters.front() ) + out_queue,
            task_queues[ out_queue ],
            thrust::raw_pointer_cast( &m_counters.front() ) + 2,
            thrust::raw_pointer_cast( &m_octree->front() ) );

        const uint32 out_queue_size = m_counters[ out_queue ];

        // swap the input and output queues
        std::swap( in_queue, out_queue );
    }
    m_node_count = m_counters[2];
}

} // namespace nih
