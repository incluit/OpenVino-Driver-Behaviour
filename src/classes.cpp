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
    this->setParkingBrake(msg->parking_brake);
}
#endif

void Player::clean() {

    if (this->device != NULL)
        ao_close(this->device);
    if (this->file != NULL)
        sf_close(this->file);
    if (this->buffer != NULL)
        free(this->buffer);

    ao_shutdown();
}

int Player::init() {

    this->buffer = (short*)calloc(BUFFER_SIZE, sizeof(short));
    if (this->buffer == NULL)
	return 1;

    ao_initialize();
    this->default_driver = ao_default_driver_id();

    this->file = sf_open(this->file_path.c_str(), SFM_READ, &sfinfo);

    switch (sfinfo.format & SF_FORMAT_SUBMASK) {
        case SF_FORMAT_PCM_16:
            format.bits = 16;
            break;
        case SF_FORMAT_PCM_24:
            format.bits = 24;
            break;
        case SF_FORMAT_PCM_32:
            format.bits = 32;
            break;
        case SF_FORMAT_PCM_S8:
            format.bits = 8;
            break;
        case SF_FORMAT_PCM_U8:
            format.bits = 8;
            break;
        default:
            format.bits = 16;
            break;
    }

    this->format.channels = this->sfinfo.channels;
    this->format.rate = this->sfinfo.samplerate;
    this->format.byte_format = AO_FMT_NATIVE;
    this->format.matrix = 0;

    this->device = ao_open_live(default_driver, &format, NULL);

    if (this->device == NULL) {
        fprintf(stderr, "Error opening device.\n");
        return 1;
    }

    return 0;
}

void Player::play() {

	int read = sf_read_short(this->file, this->buffer, BUFFER_SIZE);

	if (!cancel_playback) {
		if (read == 0)
			sf_seek(this->file, 0, SEEK_SET); // Reset offset, play again!

		for (int i = 0; i < BUFFER_SIZE; i++)
			this->buffer[i] *= this->gain;

		if (ao_play(this->device, (char *) this->buffer, (uint_32) (read * sizeof(short))) == 0) {
			printf("ao_play: failed.\n");
			sf_seek(this->file, 0, SEEK_SET); // Reset offset, play again!
		}
	}
}
