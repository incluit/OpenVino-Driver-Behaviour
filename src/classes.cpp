#include "classes.hpp"

#ifdef SIMULATOR
void Truck::ros_callback(const ets_msgs::msg::Truck::SharedPtr msg)
{
    this->setSpeed(msg->speed);
    this->setAcc(msg->acc_x, msg->acc_y, msg->acc_z);
    this->setRpm(msg->rpm);
    this->setGear(msg->gear);
    this->setEngine(msg->engine_running);
    this->setTrailer(msg->trailer_connected);
    this->setPosition(msg->x, msg->y, msg->z, msg->heading, msg->pitch, msg->roll);
}
#endif
