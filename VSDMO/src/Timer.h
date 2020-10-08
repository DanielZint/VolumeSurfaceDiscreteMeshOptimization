#pragma once

#include <chrono>
#include <string>

class Timer {
public:
	Timer();
	Timer(std::string name);

	// starts or restarts the timer.
	void start();

	// toggles pause
	void togglePause();

	// pauses the timer. If the timer was not unpaused before this call
	// does nothing.
	void pause();

	// unpauses the timer. If the timer was not paused before this call
	// does nothing.
	void unpause();

	// returns the current time in milli seconds
	long long timeInMillis();

	// returns the current time in seconds
	double timeInSeconds();

	void printTimeInSeconds();
protected:
	bool m_isPaused;

	std::chrono::time_point<std::chrono::system_clock> m_startTime;

	long long m_millisecondsPassed;

	std::string m_name;

};

class ScopeTimer : public Timer {
	ScopeTimer(std::string name) : Timer(name) {}
	~ScopeTimer() {
		printTimeInSeconds();
	}
};
