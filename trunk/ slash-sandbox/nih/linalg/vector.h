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
#include <nih/basic/numbers.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace nih {

///
/// Abstract linear algebra vector class, templated over type and dimension
///
template <typename T, size_t DIM>
struct Vector
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = DIM;

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,DIM-1>& v, const T w)
	{
		for (size_t i = 0; i < DIM-1; i++)
			x[i] = v[i];

		x[DIM-1] = w;
	}

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

    // compatibility with std::vector and sparse/dynamic vectors
    void resize(const size_t n) {}

	T x[DIM];
};


///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 2
///
template <typename T>
struct Vector<T,2>
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = 2u;

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const T v0, const T v1)
	{
		x[0] = v0;
		x[1] = v1;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,1>& v, const T v1)
	{
		x[0] = v[0];
		x[1] = v1;
	}

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

	T x[2];
};
///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 3
///
template <typename T>
struct Vector<T,3>
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = 3u;

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const T v0, const T v1, const T v2)
	{
		x[0] = v0;
		x[1] = v1;
		x[2] = v2;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,2>& v, const T v2)
	{
		x[0] = v[0];
		x[1] = v[1];
		x[2] = v2;
	}

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

	T x[3];
};

///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 4
///
template <typename T>
struct Vector<T,4>
{
	typedef T           value_type;
	typedef T           Field_type;
	static const size_t kDimension = 4u;

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector() {}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE explicit Vector(const T* v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v[i];
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const T v0, const T v1, const T v2, const T v3)
	{
		x[0] = v0;
		x[1] = v1;
		x[2] = v2;
		x[3] = v3;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE Vector(const Vector<T,3>& v, const T v3)
	{
		x[0] = v[0];
		x[1] = v[1];
		x[2] = v[2];
		x[3] = v3;
	}

	NIH_HOST NIH_DEVICE FORCE_INLINE Vector& operator=(const Vector& v)
	{
		for (size_t i = 0; i < kDimension; i++)
			x[i] = v.x[i];
		return *this;
	}
	NIH_HOST NIH_DEVICE FORCE_INLINE const T& operator[](const size_t i) const	{ return x[i]; }
	NIH_HOST NIH_DEVICE FORCE_INLINE T& operator[](const size_t i)		{ return x[i]; }

	NIH_HOST NIH_DEVICE size_t dimension() const { return kDimension; }

	T x[4];
};

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE bool operator==(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	for (size_t i = 0; i < DIM; i++)
    {
		if (op1[i] != op2[i])
            return false;
    }
	return true;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE bool operator!=(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	for (size_t i = 0; i < DIM; i++)
    {
		if (op1[i] != op2[i])
            return true;
    }
	return false;
}


template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator*(const Vector<T,DIM>& op1, const T op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1[i] * op2;
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator/(const Vector<T,DIM>& op1, const T op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1[i] / op2;
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator*(const T op1, const Vector<T,DIM>& op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1 * op2[i];
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator*=(Vector<T,DIM>& op1, const T op2)
{
	for (size_t i = 0; i < DIM; i++)
		op1[i] *= op2;
	return op1;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator+(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1[i] + op2[i];
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator-(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1[i] - op2[i];
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator*(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1[i] * op2[i];
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator/(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = op1[i] / op2[i];
	return r;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator+=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	for (size_t i = 0; i < DIM; i++)
		op1[i] += op2[i];
	return op1;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator-=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	for (size_t i = 0; i < DIM; i++)
		op1[i] -= op2[i];
	return op1;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator*=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	for (size_t i = 0; i < DIM; i++)
		op1[i] *= op2[i];
	return op1;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator/=(Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	for (size_t i = 0; i < DIM; i++)
		op1[i] /= op2[i];
	return op1;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM>& operator/=(Vector<T,DIM>& op1, const T op2)
{
	for (size_t i = 0; i < DIM; i++)
		op1[i] /= op2;
	return op1;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> operator-(const Vector<T,DIM>& op1)
{
	Vector<T,DIM> r;
	for (size_t i = 0; i < DIM; i++)
		r[i] = -op1[i];
	return r;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T intensity(const Vector<T,DIM>& v)
{
	T r(0.0);
	for (size_t i = 0; i < DIM; i++)
		r += v[i];

	return r / float(v.dimension());
}
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE T intensity(const T v) { return v; }

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T dot(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	T r(0.0);
	for (size_t i = 0; i < DIM; i++)
		r += op1[i] * op2[i];

	return r;
}
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,3u> cross(const Vector<T,3u>& op1, const Vector<T,3u>& op2)
{
	return Vector<T,3u>(
		op1[1]*op2[2] - op1[2]*op2[1],
		op1[2]*op2[0] - op1[0]*op2[2],
		op1[0]*op2[1] - op1[1]*op2[0]);
}
template <typename T>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,3> reflect(const Vector<T,3> I, const Vector<T,3> N)
{
	return I - T(2.0)*dot(I,N)*N;
}
template <typename T>
NIH_HOST NIH_DEVICE Vector<T,3>	orthogonal(const Vector<T,3> v)
{
    if (v[0]*v[0] < v[1]*v[1])
	{
		if (v[0]*v[0] < v[2]*v[2])
		{
			// r = -cross( v, (1,0,0) )
			return Vector<T,3>( 0.0f, -v[2], v[1] );
		}
		else
		{
			// r = -cross( v, (0,0,1) )
			return Vector<T,3>( -v[1], v[0], 0.0 );
		}
	}
	else
	{
		if (v[1]*v[1] < v[2]*v[2])
		{
			// r = -cross( v, (0,1,0) )
			return Vector<T,3>( v[2], 0.0, -v[0] );
		}
		else
		{
			// r = -cross( v, (0,0,1) )
			return Vector<T,3>( -v[1], v[0], 0.0 );
		}
	}
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T euclidean_distance(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	const Vector<T,DIM> d( op1 - op2 );
	return sqrtf( dot( d, d ) );
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T square_euclidean_distance(const Vector<T,DIM>& op1, const Vector<T,DIM>& op2)
{
	const Vector<T,DIM> d( op1 - op2 );
	return dot( d, d );
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T norm(const Vector<T,DIM>& op)
{
	return sqrtf( dot( op, op ) );
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T sq_norm(const Vector<T,DIM>& op)
{
	return dot( op, op );
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> normalize(const Vector<T,DIM> v)
{
#ifdef __CUDACC__
	const T invNorm = rsqrtf( dot( v, v ) );
#else
	const T invNorm = 1.0f / sqrtf( dot( v, v ) );
#endif
	return v * invNorm;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE T max_comp(const Vector<T,DIM>& v)
{
	float r = v[0];
	for (uint32 i = 1; i < DIM; i++)
		r = max( r, v[i] );
	return r;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE uint32 max_element(const Vector<T,DIM>& v)
{
	float  max_r = v[0];
    uint32 max_i = 0u;
	for (uint32 i = 1; i < DIM; i++)
    {
        if (max_r < v[i])
        {
            max_r = v[i];
            max_i = i;
        }
    }
	return max_i;
}

template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> min(const Vector<T,DIM>& v1, const Vector<T,DIM>& v2)
{
	Vector<T,DIM> r;
	for (uint32 i = 0; i < DIM; i++)
		r[i] = min( v1[i], v2[i] );
	return r;
}
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> max(const Vector<T,DIM>& v1, const Vector<T,DIM>& v2)
{
	Vector<T,DIM> r;
	for (uint32 i = 0; i < DIM; i++)
		r[i] = max( v1[i], v2[i] );
	return r;
}

/// compute the largest dimension of a given vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE uint32 largest_dim(const Vector<T,DIM>& v)
{
	return uint32( std::max_element( &v[0], &v[0] + DIM ) - &v[0] );
}

/// return a normal facing in the opposite direction wrt the view vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> face_forward(const Vector<T,DIM>& n, const Vector<T,DIM>& view)
{
    return dot( n, view ) > 0.0f ? -n : n;
}

/// compute the modulus of a vector
template <typename T, size_t DIM>
NIH_HOST NIH_DEVICE FORCE_INLINE Vector<T,DIM> mod(const Vector<T,DIM>& v, const float m)
{
    Vector<T,DIM> r;
	for (uint32 i = 0; i < DIM; i++)
        r[i] = nih::mod( v[i], m );
	return r;
}

typedef Vector<float,2> Vector2f;
typedef Vector<float,3> Vector3f;
typedef Vector<float,4> Vector4f;
typedef Vector<double,2> Vector2d;
typedef Vector<double,3> Vector3d;
typedef Vector<double,4> Vector4d;
typedef Vector<int32,2> Vector2i;
typedef Vector<int32,3> Vector3i;
typedef Vector<int32,4> Vector4i;

template <typename T>
struct Vector_traits
{
	typedef T Field_type;
	typedef T value_type;
};
template <typename T>
struct Vector_traits<T*>
{
	typedef T Field_type;
	typedef T value_type;
};

template <typename T, size_t DIM>
struct Vector_traits< Vector<T,DIM> >
{
	typedef T Field_type;
    typedef T value_type;
};

///
/// Abstract linear algebra vector class, templated over type and specialized to dimension 4
///
template <typename T>
struct Dynamic_vector
{
	typedef T           value_type;
	typedef T           Field_type;

	NIH_HOST FORCE_INLINE Dynamic_vector() {}
    NIH_HOST explicit Dynamic_vector(const size_t dim) : x( dim ) {}
    NIH_HOST Dynamic_vector(const size_t dim, const T v) : x( dim, v ) {}
    NIH_HOST Dynamic_vector(const size_t dim, const T* v) :
        x( dim )
	{
		for (size_t i = 0; i < dim; i++)
			x[i] = v[i];
	}
    NIH_HOST FORCE_INLINE Dynamic_vector(const Dynamic_vector& v) : x( v.x ) {}

	NIH_HOST FORCE_INLINE Dynamic_vector& operator=(const Dynamic_vector& v)
	{
        x = v.x;
		return *this;
	}
	NIH_HOST FORCE_INLINE const T& operator[](const size_t i) const   { return x[i]; }
	NIH_HOST FORCE_INLINE T&       operator[](const size_t i)         { return x[i]; }

	NIH_HOST FORCE_INLINE size_t dimension() const { return x.size(); }

    NIH_HOST void resize(const size_t n) { x.resize(n); }

    std::vector<T> x;
};

template <typename T>
struct Vector_traits< Dynamic_vector<T> >
{
	typedef T Field_type;
    typedef T value_type;
};


template <typename T>
NIH_HOST bool operator==(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
    {
		if (op1[i] != op2[i])
            return false;
    }
	return true;
}
template <typename T, size_t DIM>
NIH_HOST bool operator!=(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
    {
		if (op1[i] != op2[i])
            return true;
    }
	return false;
}


template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator*(const Dynamic_vector<T>& op1, const T op2)
{
	Dynamic_vector<T> r( op1.dimension() );
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = op1[i] * op2;
	return r;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator/(const Dynamic_vector<T>& op1, const T op2)
{
	Dynamic_vector<T> r( op1.dimension() );
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = op1[i] / op2;
	return r;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator*(const T op1, const Dynamic_vector<T>& op2)
{
	Dynamic_vector<T> r( op2.dimension() );
	for (size_t i = 0; i < op2.dimension(); i++)
		r[i] = op1 * op2[i];
	return r;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator*=(Dynamic_vector<T>& op1, const T op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
		op1[i] *= op2;
	return op1;
}

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator+(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	Dynamic_vector<T> r;
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = op1[i] + op2[i];
	return r;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator-(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	Dynamic_vector<T> r( op1.dimension() );
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = op1[i] - op2[i];
	return r;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator*(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	Dynamic_vector<T> r( op1.dimension() );
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = op1[i] * op2[i];
	return r;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator/(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	Dynamic_vector<T> r( op1.dimension() );
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = op1[i] / op2[i];
	return r;
}

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator+=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
		op1[i] += op2[i];
	return op1;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator-=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
		op1[i] -= op2[i];
	return op1;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator*=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
		op1[i] *= op2[i];
	return op1;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator/=(Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
		op1[i] /= op2[i];
	return op1;
}
template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T>& operator/=(Dynamic_vector<T>& op1, const T op2)
{
	for (size_t i = 0; i < op1.dimension(); i++)
		op1[i] /= op2;
	return op1;
}

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> operator-(const Dynamic_vector<T>& op1)
{
	Dynamic_vector<T> r( op1.dimension() );
	for (size_t i = 0; i < op1.dimension(); i++)
		r[i] = -op1[i];
	return r;
}

template <typename T>
NIH_HOST FORCE_INLINE T intensity(const Dynamic_vector<T>& v)
{
	T r(0.0);
	for (size_t i = 0; i < v.dimension(); i++)
		r += v[i];

	return r / float(v.dimension());
}

template <typename T>
NIH_HOST FORCE_INLINE T dot(const Dynamic_vector<T>& op1, const Dynamic_vector<T>& op2)
{
	T r(0.0);
	for (size_t i = 0; i < op1.dimension(); i++)
		r += op1[i] * op2[i];

	return r;
}

template <typename T>
NIH_HOST FORCE_INLINE T norm(const Dynamic_vector<T>& op)
{
	return sqrtf( dot( op, op ) );
}

template <typename T>
NIH_HOST FORCE_INLINE T sq_norm(const Dynamic_vector<T>& op)
{
	return dot( op, op );
}

template <typename T>
NIH_HOST FORCE_INLINE Dynamic_vector<T> normalize(const Dynamic_vector<T> v)
{
	const T invNorm = 1.0f / sqrtf( dot( v, v ) );
	return v * invNorm;
}

template <typename T>
NIH_HOST FORCE_INLINE T max_comp(const Dynamic_vector<T>& v)
{
	float r = v[0];
	for (uint32 i = 1; i < v.dimension(); i++)
		r = max( r, v[i] );
	return r;
}

} // namespace nih
