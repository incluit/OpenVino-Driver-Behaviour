#pragma once
#include <ao/ao.h>
#include <signal.h>
#include <sndfile.h>
#include <string>

#ifdef SIMULATOR
#include "rclcpp/rclcpp.hpp"
#include "ets_msgs/msg/truck.hpp"
#endif

#define BUFFER_SIZE 8192

struct Acc_t {
	double x;
	double y;
	double z;
};

struct Pos_t {
	double x;
	double y;
	double z;
	double heading;
	double pitch;
	double roll;
};

class Truck
{
private:
	double		speed;
	Acc_t		acc;
	int		rpm;
	int		gear;
	bool		engine_running;
	bool		trailer_connected;
	Pos_t		position;
public:
	/* Member Initializer & Constructor*/
	Truck() : speed(.0), rpm(0), gear(0), engine_running(false), trailer_connected(false) {
		this->setAcc(.0, .0, .0);
		this->setPosition(.0, .0, .0, .0, .0, .0);
	}
	/* Get Function */
	double		getSpeed() { return this->speed; }
	Acc_t		getAcc() { return this->acc; }
	int		getRpm() { return this->rpm; }
	int		getGear() { return this->gear; }
	bool		getEngine() { return this->engine_running; }
	bool		getTrailer() { return this->trailer_connected; }
	Pos_t		getPosition() { return this->position; }

	/* Set Function */
	void setSpeed(double _speed) { this->speed = _speed; }
	void setAcc(Acc_t _acc) { this->acc = _acc; }
	void setAcc(double _acc_x, double _acc_y, double _acc_z) { this->acc.x = _acc_x; this->acc.y = _acc_y; this->acc.z = _acc_z; }
	void setRpm(int _rpm) { this->rpm = _rpm; }
	void setGear(int _gear) { this->gear = _gear; }
	void setEngine(bool _engine) { this->engine_running = _engine; }
	void setTrailer(bool _trailer) { this->trailer_connected = _trailer; }
	void setPosition(Pos_t _position) { this->position = _position; }
	void setPosition(double _x, double _y, double _z, double _heading, double _pitch, double _roll) { this->position.x = _x; this->position.y = _y; this->position.z = _z; this->position.heading = _heading; this->position.pitch = _pitch; this->position.roll = _roll; }

#ifdef SIMULATOR
        void ros_callback(const ets_msgs::msg::Truck::SharedPtr msg);
#endif
};

class Player
{
private:
	ao_device *device;
	ao_sample_format format;
	SF_INFO sfinfo;
	int default_driver;
	short *buffer;
	SNDFILE * file;
	SNDFILE * file_bkp;
	std::string file_path;
        float gain;
	bool cancel_playback;
	bool playing;
	bool inited;
public:
	/* Member Initializer & Constructor*/
	Player(std::string _file_path) : cancel_playback(false), inited(false), playing(false), file_path(_file_path), gain(1) {
		this->init();
	}
	/* Get Function */
	std::string	getFilePath() { return this->file_path; }
	float		getGain() { return this->gain; }
	bool		getCancelPlayback() { return this->cancel_playback; }
	bool		isPlaying() { return this->playing; }
	bool		isInited() { return this->inited; }

	/* Set Function */
	void setFilePath(std::string _file_path) { this->file_path = _file_path; }
	void setGain(float _gain) { this->gain = _gain; }
	void toggleCancelPlayback() { this->cancel_playback = !cancel_playback; }

	void clean();
	int init();
	void play();
};
