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

#include <nih/basic/numbers.h>
#include <nih/linalg/vector.h>
#include <nih/analysis/project.h>
#include <nih/basic/functors.h>

namespace nih {

template <typename Vector3>
NIH_HOST_DEVICE float sh(const int32 l, const int32 m, const Vector3& v);

template <int32 l, typename Vector3>
NIH_HOST_DEVICE float sh(const int32 m, const Vector3& v);

template <int32 l, int32 m, typename Vector3>
NIH_HOST_DEVICE float sh(const Vector3& v);

template <typename ZHVector, typename SHVector, typename Vector3>
NIH_HOST_DEVICE void rotate_ZH(const int32 L, const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff);

template <int32 L, typename ZHVector, typename SHVector, typename Vector3>
NIH_HOST_DEVICE void rotate_ZH(const ZHVector& zh_coeff, const Vector3& d, SHVector& sh_coeff);

template <int32 l, int32 m, typename Vector3>
NIH_HOST_DEVICE float rotate_ZH(const float zh_l, const Vector3& d);


template <int32 L>
struct SH_basis
{
    static const int32 ORDER  = L;
    static const int32 COEFFS = L*L;

    template <typename Vector3>
    static NIH_HOST_DEVICE float eval(const int32 i, const Vector3& d)
    {
        if (i == 0)
            return sh<0>( 0, d );
        else if (i < 4)
            return sh<1>( i - 2, d );
        else if (i < 9)
            return sh<2>( i - 6, d );
        else
            return sh<3>( i - 12, d );
    }

    static NIH_HOST_DEVICE void clamped_cosine(const Vector3f& normal, const float w, float* coeffs)
    {
        const float zh[4] = {
            0.891209f,
            1.031964f,
            0.506579f,
            0.013224f };

        float sh[COEFFS];
        rotate_ZH<L>( zh, normal, sh );

        for (uint32 i = 0; i < COEFFS; ++i)
            coeffs[i] += sh[i] * w;
    }

    static NIH_HOST_DEVICE void constant(float k, float* coeffs)
    {
        coeffs[0] = k * 2.0f*sqrtf(M_PIf);
        for (int32 i = 1; i < COEFFS; ++i)
            coeffs[i] = 0.0f;
    }

    static NIH_HOST_DEVICE float integral(const float* coeffs) { return coeffs[0]; }

    template <typename Vector_type>
    static NIH_HOST_DEVICE float integral(const Vector_type& coeffs) { return coeffs[0]; }

    static NIH_HOST_DEVICE void solve(float* coeffs) {}
};

} // namespace nih

#include <nih/spherical/sh_inline.h>