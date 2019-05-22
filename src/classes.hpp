#pragma once

#ifdef SIMULATOR
#include "rclcpp/rclcpp.hpp"
#include "ets_msgs/msg/truck.hpp"
#endif

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
