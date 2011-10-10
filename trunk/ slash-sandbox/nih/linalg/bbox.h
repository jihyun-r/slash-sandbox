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

#ifdef min
#undef min
#endif
#ifdef max
#unded max
#endif

#include <nih/basic/numbers.h>
#include <nih/linalg/vector.h>
#include <limits>
#include <algorithm>

namespace nih {

///
/// Bbox class, templated over an arbitrary vector type
///
template <typename Vector_t>
struct Bbox
{
	typedef typename Vector_t::Field_type	Field_type;
	typedef typename Vector_t				Vector_type;

	NIH_HOST NIH_DEVICE Bbox();
	NIH_HOST NIH_DEVICE Bbox(
		const Vector_t& v);
	NIH_HOST NIH_DEVICE Bbox(
		const Vector_t& v1,
		const Vector_t& v2);
	NIH_HOST NIH_DEVICE Bbox(
		const Bbox<Vector_t>& bb1,
		const Bbox<Vector_t>& bb2);
	NIH_HOST NIH_DEVICE Bbox(
		const Bbox<Vector_t>& bb);

	NIH_HOST NIH_DEVICE void insert(
		const Vector_t& v);
	NIH_HOST NIH_DEVICE void insert(
		const Bbox& v);

	NIH_HOST NIH_DEVICE void clear();

	NIH_HOST NIH_DEVICE const Vector_t& operator[](const size_t i) const	{ return (&m_min)[i]; }
	NIH_HOST NIH_DEVICE Vector_t& operator[](const size_t i)				{ return (&m_min)[i]; }

	NIH_HOST NIH_DEVICE Bbox<Vector_t>& operator=(const Bbox<Vector_t>& bb)
    {
        m_min = bb.m_min;
        m_max = bb.m_max;
        return *this;
    }

    Vector_t m_min;
	Vector_t m_max;
};

typedef Bbox<Vector2f> Bbox2f;
typedef Bbox<Vector3f> Bbox3f;
typedef Bbox<Vector4f> Bbox4f;
typedef Bbox<Vector2d> Bbox2d;
typedef Bbox<Vector3d> Bbox3d;
typedef Bbox<Vector4d> Bbox4d;

inline NIH_HOST NIH_DEVICE float area(const Bbox3f& bbox)
{
    const Vector3f edge = bbox[1] - bbox[0];
    return edge[0] * edge[1] + edge[2] * (edge[0] + edge[1]);
}

template <typename Vector_t>
inline NIH_HOST NIH_DEVICE bool contains(const Bbox<Vector_t>& bbox, const Vector_t& p)
{
    for (uint32 i = 0; i < p.dimension(); ++i)
    {
        if (p[i] < bbox[0][i] ||
            p[i] > bbox[1][i])
            return false;
    }
    return true;
}

} // namespace nih

#include <nih/linalg/bbox_inline.h>
