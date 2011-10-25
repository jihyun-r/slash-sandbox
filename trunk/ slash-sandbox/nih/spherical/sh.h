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

NIH_HOST NIH_DEVICE float sh(const int32 l, const int32 m, const Vector3f& v);

template <int32 L>
struct SH_basis
{
    static const int32 ORDER  = L;
    static const int32 COEFFS = L*L;

    static NIH_HOST NIH_DEVICE float eval(const int32 i, const Vector3f& d)
    {
        if (i == 0)
            return sh( 0, 0, d );
        else if (i < 4)
            return sh( 1, i - 2, d );
        else if (i < 9)
            return sh( 2, i - 6, d );
        else
            return sh( 3, i - 12, d );
    }
    static NIH_HOST NIH_DEVICE float eval(const int32 l, const int32 m, const Vector3f& d) { return sh( l, m, d ); }

    static void clamped_cosine(const Vector3f& normal, const float w, float* coeffs)
    {
        const int32 n_samples = 256;

        project(
            SH_basis<L>(),
            clamped_cosine_fun<Vector3f>( normal ),
            w,
            0.0f,
            n_samples,
            coeffs );
    }

    static void constant(float k, float* coeffs)
    {
        coeffs[0] = k * 2.0f*sqrtf(M_PIf);
        for (int32 i = 1; i < COEFFS; ++i)
            coeffs[i] = 0.0f;
    }

    static float integral(const float* coeffs) { return coeffs[0]; }

    template <typename Vector_type>
    static float integral(const Vector_type& coeffs) { return coeffs[0]; }

    static void solve(float* coeffs) {}
};

template <typename ZHVector, typename SHVector>
void rotate_ZH(const int32 L, const ZHVector& zh_coeff, const Vector3f& d, SHVector& sh_coeff);

template <typename ZHVector, typename SHVector>
inline void rotate_ZH(const int32 L, const ZHVector& zh_coeff, const Vector3f& d, SHVector& sh_coeff)
{
    for (int32 l = 0; l < L; ++l)
        for (int32 m = -l; m <= l; ++m)
            sh_coeff[ l*l + m+l ] = sqrtf( 4.0f*M_PIf / float(2*l + 1) ) * zh_coeff[l] * sh( l, m, d );
}

NIH_HOST NIH_DEVICE inline float sh(const int32 l, const int32 m, const Vector3f& v)
{
#if 0
    if (l == 0)
        return 0.282095f;
    else if (l == 1)
        return 0.488603f * (m == -1 ? v[0] : (m == 0 ? v[2] : v[1]));
    else if (l == 2)
    {
        if (m == 0)
            return 0.315392f * (3*v[2]*v[2] - 1.0f);
        else if (m == 2)
            return 0.546274f * (v[0]*v[0] - v[1]*v[1]);
        else if (m == -2)
            return 1.092548f * v[0]*v[2];
        else if (m == -1)
            return 1.092548f * v[1]*v[2];
        else
            return 1.092548f * v[0]*v[1];
    }
#else
    const float X = v[0];
    const float Y = v[1];
    const float Z = v[2];

    const float m_15_over_4sqrtPI           = 0.54627419f; //sqrtf(15.0f)/(4.0f*sqrtf(M_PIf));
    const float m_15_over_2sqrtPI           = 1.09254837f; //sqrtf(15.0f)/(2.0f*sqrtf(M_PIf));
    const float m_5_over_4sqrtPI            = 0.31539154f; //sqrtf(5.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt35_over_8sqrtPI  = 0.59004354f; //sqrtf(2.0f*35.0f)/(8.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt35_over_4sqrtPI  = 1.18008709f; //sqrtf(2.0f*35.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt7_over_4sqrtPI        = 0.37317631f; //sqrtf(7.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt21_over_8sqrtPI  = 0.45704576f; //sqrtf(2.0f*21.0f)/(8.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_4sqrtPI      = 1.44530571f; //sqrtf(105.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_2sqrtPI      = 2.89061141f; //sqrtf(105.0f)/(2.0f*sqrtf(M_PIf));

    if (l == 0)
        return 0.282095f;
    else if (l == 1)
        return 0.488603f * (m == -1 ? -Y : (m == 0 ? Z : -X));
    else if (l == 2)
    {
        if (m == 0)
            return m_5_over_4sqrtPI * (3*Z*Z - 1.0f);
        else if (m == 1)
            return -m_15_over_2sqrtPI * X*Z;
        else if (m == 2)
            return m_15_over_4sqrtPI * (X*X - Y*Y);
        else if (m == -1)
            return -m_15_over_2sqrtPI * Y*Z;
        else if (m == -2)
            return m_15_over_2sqrtPI * X*Y;
    }
    else if (l == 3)
    {
        if (m == 0)
            return m_sqrt7_over_4sqrtPI * Z * (5*Z*Z - 3);
        else if (m == 1)
            return -m_sqrt2sqrt21_over_8sqrtPI * X * (5*Z*Z - 1);
        else if (m == 2)
            return m_sqrt105_over_4sqrtPI * (X*X - Y*Y) * Z;
        else if (m == 3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (X*X - 3*Y*Y)*X;
        else if (m == -1)
            return -m_sqrt2sqrt21_over_8sqrtPI * Y * (5*Z*Z - 1);
        else if (m == -2)
            return m_sqrt105_over_2sqrtPI * X*Y*Z;
        else if (m == -3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (3*X*X - Y*Y)*Y;
    }
#endif
    return 0.0f;
}
template <int32 l, int32 m>
NIH_HOST NIH_DEVICE inline float sh(const Vector3f& v)
{
    const float X = v[0];
    const float Y = v[1];
    const float Z = v[2];

    const float m_15_over_4sqrtPI           = 0.54627419f; //sqrtf(15.0f)/(4.0f*sqrtf(M_PIf));
    const float m_15_over_2sqrtPI           = 1.09254837f; //sqrtf(15.0f)/(2.0f*sqrtf(M_PIf));
    const float m_5_over_4sqrtPI            = 0.31539154f; //sqrtf(5.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt35_over_8sqrtPI  = 0.59004354f; //sqrtf(2.0f*35.0f)/(8.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt35_over_4sqrtPI  = 1.18008709f; //sqrtf(2.0f*35.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt7_over_4sqrtPI        = 0.37317631f; //sqrtf(7.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt2sqrt21_over_8sqrtPI  = 0.45704576f; //sqrtf(2.0f*21.0f)/(8.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_4sqrtPI      = 1.44530571f; //sqrtf(105.0f)/(4.0f*sqrtf(M_PIf));
    const float m_sqrt105_over_2sqrtPI      = 2.89061141f; //sqrtf(105.0f)/(2.0f*sqrtf(M_PIf));

    if (l == 0)
        return 0.282095f;
    else if (l == 1)
        return 0.488603f * (m == -1 ? -Y : (m == 0 ? Z : -X));
    else if (l == 2)
    {
        if (m == 0)
            return m_5_over_4sqrtPI * (3*Z*Z - 1.0f);
        else if (m == 1)
            return -m_15_over_2sqrtPI * X*Z;
        else if (m == 2)
            return m_15_over_4sqrtPI * (X*X - Y*Y);
        else if (m == -1)
            return -m_15_over_2sqrtPI * Y*Z;
        else if (m == -2)
            return m_15_over_2sqrtPI * X*Y;
    }
    else if (l == 3)
    {
        if (m == 0)
            return m_sqrt7_over_4sqrtPI * Z * (5*Z*Z - 3);
        else if (m == 1)
            return -m_sqrt2sqrt21_over_8sqrtPI * X * (5*Z*Z - 1);
        else if (m == 2)
            return m_sqrt105_over_4sqrtPI * (X*X - Y*Y) * Z;
        else if (m == 3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (X*X - 3*Y*Y)*X;
        else if (m == -1)
            return -m_sqrt2sqrt21_over_8sqrtPI * Y * (5*Z*Z - 1);
        else if (m == -2)
            return m_sqrt105_over_2sqrtPI * X*Y*Z;
        else if (m == -3)
            return -m_sqrt2sqrt35_over_8sqrtPI * (3*X*X - Y*Y)*Y;
    }
}

} // namespace nih
