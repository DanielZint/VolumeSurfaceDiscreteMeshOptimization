#include "Timer.h"
#include <iostream>

Timer::Timer()
{
	start();
}

Timer::Timer(std::string name) : m_name(name)
{
	start();
}

void Timer::start() {
	m_isPaused = false;
	m_millisecondsPassed = 0LL;
	m_startTime = std::chrono::system_clock::now();
}

void Timer::pause() {
	if (m_isPaused) return;

	auto currentTime = std::chrono::system_clock::now();
	m_millisecondsPassed += std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_startTime).count();

	m_isPaused = true;
}

void Timer::unpause() {
	if (!m_isPaused) return;

	m_startTime = std::chrono::system_clock::now();

	m_isPaused = false;
}

void Timer::togglePause() {
	if (m_isPaused) unpause();
	else pause();
}

long long Timer::timeInMillis() {
	auto currentTime = std::chrono::system_clock::now();
	long long result = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - m_startTime).count();

	if (m_isPaused) {
		result = 0LL;
	}

	return result + m_millisecondsPassed;
}

double Timer::timeInSeconds() {
	long long millis = timeInMillis();
	return static_cast<double>(millis) / 1000.0;
}

void Timer::printTimeInSeconds() {
	long long millis = timeInMillis();
	std::cout << m_name << (static_cast<double>(millis) / 1000.0) << "s" << std::endl;
}
