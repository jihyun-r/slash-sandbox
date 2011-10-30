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

#include <nih/basic/types.h>
#include <nih/tree/model.h>
#include <iterator>

namespace nih {
namespace cuda {

///
/// Reduce a bunch of values attached to the elemens in the leaves of a tree.
/// The Tree template type has to provide the following breadth-first tree
/// interface:
///
/// struct Tree
/// {
///     // return the number of levels
///     uint32 get_level_count() const;
///
///     // return the index of the first node in the i-th level
///     uint32 get_level(const uint32 i) const;
///
///     // retrieve a node
///     node_type get_node(const uint32 index) const;
///
///     // return the number of leaves
///     uint32 get_leaf_count() const;
///
///     // retrieve a leaf
///     uint2 get_leaf(const uint32 index) const;
/// };
///
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void tree_reduce(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         leaf_values,
    Output_iterator         node_values,
    const Operator          op,
    const typename std::iterator_traits<Output_iterator>::value_type def_value);

///
/// Reduce a bunch of values attached to the leaves of a tree.
///
template <typename Tree, typename Input_iterator, typename Output_iterator, typename Operator>
void tree_reduce(
    const Tree              tree,
    const Input_iterator    in_values,
    Output_iterator         node_values,
    const Operator          op);

} // namespace cuda
} // namespace nih

#include <nih/tree/cuda/reduce_inline.h>

