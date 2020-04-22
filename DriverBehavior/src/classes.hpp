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
	bool		parking_brake;
	double		air_pressure;
	double		wear_engine;
	double		wear_transmission;
	double		battery_voltage;
	double		wear_wheels;
	double		cruise_control;
	double		fuel;
	double		fuel_average_consumption;
	double		wear_chassis;
	double		cargo_mass;
	double 		latitude;
	double 		longitude;
public:
	/* Member Initializer & Constructor*/
	Truck() : speed(.0), rpm(0), gear(0), engine_running(false), trailer_connected(false), parking_brake(false), air_pressure(.0), wear_engine(.0), wear_transmission(.0), battery_voltage(.0), wear_wheels(.0), cruise_control(.0), fuel(.0), fuel_average_consumption(.0), wear_chassis(.0), latitude(.0),longitude(.0) {
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
	bool		getParkingBrake() { return this->parking_brake; }
	double		getAirPressure() { return this->air_pressure; }
	double		getWearEngine() { return this->wear_engine; }
	double		getWearTransmission() { return this->wear_transmission; }
	double		getBattery() { return this->battery_voltage; }
	double		getWearWheels() { return this->wear_wheels; }
	double		getCruiseControl() { return this->cruise_control; }
	double		getFuel() { return this->fuel; }
	double		getFuelAverage() { return this->fuel_average_consumption; }
	double		getWearChassis() { return this->wear_chassis; }
	double		getCargoMass() { return this->cargo_mass; }
	double		getLatitude() { return this->latitude; }
	double		getLongitude() { return this->longitude; }

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
	void setParkingBrake(bool _parking_brake) { this->parking_brake = _parking_brake; }
	void setAirPressure(double _air_pressure) { this->air_pressure = _air_pressure; }
	void setWearEngine(double _wear_engine) { this->wear_engine = _wear_engine; }
	void setWearTransmission(double _wear_transmission) { this->wear_transmission = _wear_transmission; }
	void setBattery(double _battery_voltage) { this->battery_voltage = _battery_voltage; }
	void setWearWheels(double _wear_wheels) { this->wear_wheels = _wear_wheels; }
	void setCruiseControl(double _cruise_control) { this->cruise_control = _cruise_control; }
	void setFuel(double _fuel) { this->fuel = _fuel; }
	void setFuelAverage(double _fuel_average_consumption) { this->fuel_average_consumption = _fuel_average_consumption; }
	void setWearChassis(double _wear_chassis) { this->wear_chassis = _wear_chassis; }
	void setCargoMass(double _cargo_mass) { this->cargo_mass = _cargo_mass; }
	void setLatitude(double _latitude) {this->latitude = _latitude; }
	void setLongitude(double _longitude) {this->longitude = _longitude; }

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
