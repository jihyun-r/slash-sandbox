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

#pragma once

#include <nih/octree/octree.h>
#include <nih/linalg/vector.h>
#include <nih/linalg/bbox.h>
#include <nih/bintree/bintree_node.h>
#include <nih/bintree/cuda/bintree_gen_context.h>
#include <thrust/device_vector.h>

namespace nih {

/// GPU-based octree builder
struct Octree_builder
{
    /// constructor
    Octree_builder(
        thrust::device_vector<Octree_node_base>& octree,
        thrust::device_vector<uint2>&            leaves,
        thrust::device_vector<uint32>&           index) :
        m_octree( &octree ), m_leaves( &leaves ), m_index( &index ) {}

    /// build an octree given a set of points
    void build(
        const Bbox3f                           bbox,
        const thrust::device_vector<Vector4f>& points,
        const uint32                           max_leaf_size);

    thrust::device_vector<Octree_node_base>* m_octree;
    thrust::device_vector<uint2>*            m_leaves;
    thrust::device_vector<uint32>*           m_index;
    thrust::device_vector<uint32>            m_codes;
    uint32                                   m_levels[32];
    Bbox3f                                   m_bbox;
    uint32                                   m_node_count;
    uint32                                   m_leaf_count;

    thrust::device_vector<Bintree_node>      m_kd_nodes;
    cuda::Bintree_gen_context                m_kd_context;
};

} // namespace nih
