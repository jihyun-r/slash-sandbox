#pragma once

#include <base/vector.h>

namespace pbgi {

#ifdef WIN32

///
/// A simple timer class
///
struct Timer
{
	/// constructor
	Timer();

	/// start timing
	void start();

	/// stop timing
	void stop();

	float seconds() const;

	uint64			m_freq;
	uint64			m_start;
	uint64			m_stop;
};

///
/// A helper timer which measures the time from its instantiation
/// to the moment it goes out of scope
///
template <typename T>
struct Scoped_timer
{
	 Scoped_timer(T* time) : m_time( time ), m_timer() { m_timer.start(); }
	~Scoped_timer() { m_timer.stop(); *m_time += m_timer.seconds(); }

	T*		m_time;
	Timer	m_timer;
};

#endif

} // namespace pbgi
