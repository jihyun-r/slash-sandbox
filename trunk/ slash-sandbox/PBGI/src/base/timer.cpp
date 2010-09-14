#include <base/timer.h>

#ifdef WIN32

#include <windows.h>
#include <winbase.h>

namespace pbgi {

Timer::Timer()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	m_freq = freq.QuadPart;
}

void Timer::start()
{
	LARGE_INTEGER tick;
	QueryPerformanceCounter(&tick);

	m_start = tick.QuadPart;
}
void Timer::stop()
{
	LARGE_INTEGER tick;
	QueryPerformanceCounter(&tick);

	m_stop = tick.QuadPart;
}

float Timer::seconds() const
{
	return float(m_stop - m_start) / float(m_freq);
}

} // namespace pbgi

#endif