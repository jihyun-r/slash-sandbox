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

#include <cmath>
#include <limits>
#include <vector>
#include <nih/basic/types.h>

namespace nih {

///
/// A small helper class to keep statistics
///
template <typename T>
struct Stats
{
	Stats() :
		m_sum(T(0)),
		m_sum_sq(T(0)),
		m_min(std::numeric_limits<T>::max()),
		m_max(std::numeric_limits<T>::min()),
		m_count(0u) {}

	Stats(const T min, const T max) :
		m_sum(T(0)),
		m_sum_sq(T(0)),
		m_min(max),
		m_max(min),
		m_count(0u) {}

	void operator+= (const T v)
	{
		m_sum += v;
		m_sum_sq += v*v;
		m_count++;
		m_min = std::min( v, m_min );
		m_max = std::max( v, m_max );
	}
	void operator+= (const Stats<T>& v)
	{
		m_sum    += v.m_sum;
		m_sum_sq += v.m_sum_sq;
		m_count  += v.m_count;
		m_min = std::min( v.m_min, m_min );
		m_max = std::max( v.m_max, m_max );
	}

	T			avg() const { return m_count ? m_sum / T(m_count) : T(0); }
	T			sum() const { return m_sum; }
	T			var() const { return m_count ? (m_sum_sq - m_sum) / T(m_count) : T(0); }
	T			sigma() const { return T( sqrt( var() ) ); }
	T			min() const { return m_min; }
	T			max() const { return m_max; }
	long long	count() const { return m_count; }

	T			m_sum;
	T			m_sum_sq;
	T			m_min;
	T			m_max;
	long long	m_count;
};

///
/// A small helper class to keep statistics
///
template <typename T>
struct Histogram
{
	typedef Stats<T> Stats_type;

	Histogram() {}
	Histogram(
		const Stats_type& stats,
		const uint32 bins) :
		m_stats( stats ),
		m_bins( bins ),
		m_var(T(0))
	{
	}

	uint32 bin(const T v) const
	{
		const double x = double(m_bins.size()) * std::max( double(v - m_stats.min()), 0.0 ) / std::max( double(m_stats.max() - m_stats.min()), 1.0e-8 );
		const uint32 b = std::min( uint32(x), (uint32)m_bins.size()-1 );
		return b;
	}
	T bin_min(const uint32 b) const
	{
		return (m_stats.max() - m_stats.min()) * float(b) / float(m_bins.size()) + m_stats.min();
	}
	T bin_max(const uint32 b) const
	{
		return (m_stats.max() - m_stats.min()) * float(b+1) / float(m_bins.size()) + m_stats.min();
	}

	void operator+= (const T v)
	{
		uint32 b = bin( v );
		m_bins[b] += v;
		m_var += (v - m_stats.avg()) * (v - m_stats.avg());
	}

	T			avg(const uint32 b) const { return m_bins[b].avg(); }
	T			sum(const uint32 b) const { return m_bins[b].sum(); }
	T			min(const uint32 b) const { return m_bins[b].min(); }
	T			max(const uint32 b) const { return m_bins[b].max(); }
	T			var() const { return m_var / std::max( (float)m_stats.count(), 1.0f ); }
	T			sigma() const { return T( sqrt( var() ) ); }
	long long	count(const uint32 b) const { return m_bins[b].count(); }
	float		percentage(const uint32 b) const { return float( m_bins[b].count() ) / float( m_stats.count() ); }

	Stats_type				m_stats;
	std::vector<Stats_type>	m_bins;
	T						m_var;
};

/// A small class to keep a running estimate of avg/variance
class RunningStat
{
public:
    NIH_HOST_DEVICE RunningStat() : m_n(0) {}

    NIH_HOST_DEVICE void clear()
    {
        m_n = 0;
    }

    NIH_HOST_DEVICE void push(float x)
    {
        m_n++;

        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (m_n == 1)
        {
            m_oldM = m_newM = x;
            m_oldS = 0.0;
        }
        else
        {
            m_newM = m_oldM + (x - m_oldM)/m_n;
            m_newS = m_oldS + (x - m_oldM)*(x - m_newM);

            // set up for next iteration
            m_oldM = m_newM; 
            m_oldS = m_newS;
        }
    }

    NIH_HOST_DEVICE int count() const
    {
        return m_n;
    }

    NIH_HOST_DEVICE float mean() const
    {
        return (m_n > 0) ? m_newM : 0.0f;
    }

    NIH_HOST_DEVICE float variance() const
    {
        return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0f );
    }

private:
    int m_n;
    float m_oldM, m_newM, m_oldS, m_newS;
};

} // namespace nih
