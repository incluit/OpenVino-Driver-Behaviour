#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <thread>
#include <mutex>
#include <math.h>
#include <sys/types.h>
#include <signal.h>

#include <inference_engine.hpp>

#include <samples/slog.hpp>

#include "customflags.hpp"
#include "detectors.hpp"
#include "cnn.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "classes.hpp"
#include "picojson.hpp"


#include <ie_iextension.h>

#include <opencv2/opencv.hpp>

#include <aws/crt/Api.h>
#include <aws/crt/StlAllocator.h>
#include <aws/iot/MqttClient.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <boost/circular_buffer.hpp>


#include "rclcpp/rclcpp.hpp"
#include "ets_msgs/msg/truck.hpp"

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>


#include <sys/ioctl.h>  // Library to use ioctl function


void ros_client(Truck *truck)
{
	auto node = rclcpp::Node::make_shared("ets_client");

	auto sub = node->create_subscription<ets_msgs::msg::Truck>(
			"truck", std::bind(&Truck::ros_callback, truck, std::placeholders::_1), rmw_qos_profile_default);

	rclcpp::spin(node);
}


using namespace InferenceEngine;

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

float distanceAtoB(cv::Point2f A, cv::Point2f B)
{
	float distance_l = sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
	return distance_l;
}

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
	// ---------------------------Parsing and validation of input args--------------------------------------
	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
	if (FLAGS_h)
	{
		showUsage();
		return false;
	}
	slog::info << "Parsing input parameters" << slog::endl;

	if (FLAGS_i.empty())
	{
		throw std::logic_error("Parameter -i is not set");
	}

	if (FLAGS_m.empty())
	{
		throw std::logic_error("Parameter -m is not set");
	}

	if (FLAGS_n_ag < 1)
	{
		throw std::logic_error("Parameter -n_ag cannot be 0");
	}

	if (FLAGS_n_hp < 1)
	{
		throw std::logic_error("Parameter -n_hp cannot be 0");
	}

	// no need to wait for a key press from a user if an output image/video file is not shown.
	FLAGS_no_wait |= FLAGS_no_show;

	return true;
}

enum distractionLevel
{
	NOT_DISTRACTED = 0,
	DISTRACTED,
	PHONE,
};

int isDistracted(float y, float p, float r)
{
	int result = 0;
	if (abs(y) > 30 || abs(p) > 30)
	{
		if (abs(y) > 20 && p > 10 && r < 0)
			result = PHONE;
		else
			result = DISTRACTED;
	}
	return result;
}

bool identify_driver(cv::Mat frame, std::vector<FaceDetection::Result> *results, VectorCNN *landmarks_detector,
		VectorCNN *face_reid, EmbeddingsGallery *face_gallery, std::string *driver_name)
{
	bool ret = false;
	std::vector<cv::Mat> face_rois, landmarks, embeddings;

	if (results->empty())
		return ret;

	for (const auto &face : *results)
	{
		cv::Rect rect = face.location;
		float scale_factor_x = 0.15;
		float scale_factor_y = 0.20;
		double aux_x = (rect.x > 3 ? rect.x : 3);
		double aux_y = (rect.y > 3 ? rect.y : 3);
		double aux_width = (rect.width + aux_x < frame.cols ? rect.width : frame.cols - aux_x);
		double aux_height = (rect.height + aux_y < frame.rows ? rect.height : frame.rows - aux_y);
		aux_x += scale_factor_x * aux_width;
		aux_y += scale_factor_y * aux_height;
		aux_width = aux_width * (1 - 2 * scale_factor_x);
		aux_height = aux_height * (1 - scale_factor_y);
		cv::Rect aux_rect = cv::Rect(aux_x, aux_y, aux_width, aux_height);
		face_rois.push_back(frame(aux_rect));
	}

	if (!face_rois.empty())
	{
		landmarks_detector->Compute(face_rois, &landmarks, cv::Size(2, 5));
		AlignFaces(&face_rois, &landmarks);
		face_reid->Compute(face_rois, &embeddings);
		auto ids = face_gallery->GetIDsByEmbeddings(embeddings);

		if (!ids.empty() && ids[0] != EmbeddingsGallery::unknown_id)
		{
			ret = true;
			*driver_name = face_gallery->GetLabelByID(ids[0]);
		}
		else
			*driver_name = "Unknown";
	}

	return ret;
}

// Global variables declaration
int timer_off = 0; // N frames for Welcome sign
bool face_identified = false;
bool first_stage_completed = (FLAGS_d_recognition ? false : true);
int biggest_head = 0;
bool falarmDistraction = false;
std::string driver_name = "";
Timer timer;
int firstTime = 0;
Truck truck;
bool fSim = false;


int maxNormal = 40;
int maxWarning = 70;
int maxCritical = 100; //Max Drowsiness value
int x_vum = 20;
int y_vum = 150;
double y_vum_unit = 1.5; // y_vum_unit = y_vum/maxCritical
double tDrowsiness = 0;

double tDistraction = 0;
int vDistraction = 0;
bool firstDistraction = true;
double timeDistraction = 0.0;
int startNoDistraction = 0;

// Yawn/Blink Variables
int vYawn = 0;
int vBlink = 0;
double timeBlink = 0.0;
int startNoDrowsiness = 0;

bool firstBlink = true;

bool startHeadbutt = true;

int vHeadbutt = 0;

std::string labelAlarm = "";

cv::Mat face_save;
bool firstPhoto = true;

void alarmDrowsiness(cv::Mat prev_frame, int yawn_total, int blinl_total, int width, int height, int x_alarm, int y_alarm, int x_truck_i, bool headbutt)
{
	// Headbutt Logic
	if (headbutt && startHeadbutt)
	{
		vHeadbutt = 1;
		startHeadbutt = false;
		timer.start("headbutt");
	}
	else
		vHeadbutt = 0;

	if (!startHeadbutt && timer["headbutt"].getSmoothedDuration() >= 2000)
	{
		startHeadbutt = true;
	}

	//VU Meter Logic
	if ((tDrowsiness <= maxNormal) && (vYawn != 0 || vBlink != 0) && !truck.getParkingBrake())
	{
		if (tDrowsiness <= 100){
			tDrowsiness += 10 * vYawn + 5 * vBlink * (timeBlink / 1000);
		}
		else
			tDrowsiness=100.0;
		
		startNoDrowsiness = 0;
	}

	else if ((tDrowsiness > maxNormal) && (vYawn != 0 || vBlink != 0 || vHeadbutt != 0) && !truck.getParkingBrake())
	{
		if (tDrowsiness <= 100){
			tDrowsiness += 5 * vYawn + 10 * vBlink * (timeBlink / 1000) + vHeadbutt * 30;
		}
		else
			tDrowsiness=100.0;
		
		startNoDrowsiness = 0;
	}
	else
	{
		if (startNoDrowsiness == 0)
		{
			startNoDrowsiness = 1;
			timer.start("NoDrowsiness");
		}
		if (startNoDrowsiness == 1 && timer["NoDrowsiness"].getSmoothedDuration() >= 1000)
		{
			if (tDrowsiness >= 1)
				tDrowsiness--;
			else
				tDrowsiness = 0.0; //This is because the variable is rounded when it is displayed, and could show 1.
			startNoDrowsiness = 0;
		}
	}
	vYawn = 0;

	int x_vum_drow = x_truck_i + 35;
	int y_vum_drow = y_alarm + 55;

	//Drawing VU Meters
	// VUmeter Drowsiness: Rectangle Background
	cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 50, 0), -1);
	cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxWarning), x_vum, y_vum_unit * (maxWarning - maxNormal)), cv::Scalar(0, 50, 50), -1);
	cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxCritical), x_vum, y_vum_unit * (maxCritical - maxWarning)), cv::Scalar(0, 0, 50), -1);

	// VU Meter Logic
	if (tDrowsiness <= maxNormal)
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - y_vum_unit * tDrowsiness, x_vum, y_vum_unit * tDrowsiness), cv::Scalar(0, 255, 0), -1);
		// tDrowsiness Label
		cv::putText(prev_frame, cv::format("%3.0f", tDrowsiness), cv::Point2f(x_vum_drow + 30, y_vum_drow + y_vum - y_vum_unit * tDrowsiness + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}
	else if (tDrowsiness > maxNormal && tDrowsiness <= maxWarning)
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 255, 0), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * tDrowsiness), x_vum, y_vum_unit * (tDrowsiness - maxNormal)), cv::Scalar(0, 255, 255), -1);
		// tDrowsiness Label
		cv::putText(prev_frame, cv::format("%3.0f", tDrowsiness), cv::Point2f(x_vum_drow + 30, y_vum_drow + y_vum - y_vum_unit * tDrowsiness + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}

	else if (tDrowsiness > maxWarning && tDrowsiness <= maxCritical)
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 255, 0), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxWarning), x_vum, y_vum_unit * (maxWarning - maxNormal)), cv::Scalar(0, 255, 255), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * tDrowsiness), x_vum, y_vum_unit * (tDrowsiness - maxWarning)), cv::Scalar(0, 0, 255), -1);
		// tDrowsiness Label
		cv::putText(prev_frame, cv::format("%3.0f", tDrowsiness), cv::Point2f(x_vum_drow + 30, y_vum_drow + y_vum - y_vum_unit * tDrowsiness + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}
	else
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 255, 0), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxWarning), x_vum, y_vum_unit * (maxWarning - maxNormal)), cv::Scalar(0, 255, 255), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow + y_vum - (y_vum_unit * maxCritical), x_vum, y_vum_unit * (maxCritical - maxWarning)), cv::Scalar(0, 0, 255), -1);
		// tDrowsiness Label
		cv::putText(prev_frame, cv::format("%3.0f", tDrowsiness), cv::Point2f(x_vum_drow + 30, y_vum_drow + y_vum - (y_vum_unit * maxCritical) + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}

	// VUmeter Drowsiness: Rectangle
	cv::rectangle(prev_frame, cv::Rect(x_vum_drow, y_vum_drow, x_vum, y_vum), cv::Scalar(255, 255, 255), 1);
}

void alarmDistraction(cv::Mat prev_frame, int is_dist, int y_alarm, int x_truck_i, int pid_da)
{
	// Alarm Logic Function
	if (is_dist)
	{
		switch (is_dist)
		{
			case DISTRACTED:
				if (truck.getSpeed() * 3.6 >= 5 || !fSim)
				{
					labelAlarm = "EYES OUT OF ROAD";
					falarmDistraction = true;
				}
				break;
			case PHONE:
				if (truck.getSpeed() * 3.6 >= 2 || truck.getSpeed() * 3.6 <= -2 || !fSim)
				{
					labelAlarm = "LOOKING AT THE PHONE";
					falarmDistraction = true;
				}
				break;
			default:
				falarmDistraction = false;
				break;
		}
	}
	else
	{
		labelAlarm = "";
		falarmDistraction = false;
	}

	if (falarmDistraction)
	{
		// Distraction Logic
		vDistraction = 1;
		if (firstDistraction)
		{
			timeDistraction = 0;
			firstDistraction = false;
		}
		else
		{
			timeDistraction = timer["timeDistraction"].getSmoothedDuration() - timeDistraction;
		}
		timer.start("timeDistraction");
		// End Distraction Logic
		if(tDistraction<100.0){
			tDistraction += 10 * vDistraction * (timeDistraction / 1000);
		}
		else
			tDistraction=100.0;
		
	}
	else
	{
		vDistraction = 0;
		timeDistraction = 0;
		firstDistraction = true;

		if (startNoDistraction == 0)
		{
			startNoDistraction = 1;
			timer.start("NoDistraction");
		}
		if (startNoDistraction == 1 && timer["NoDistraction"].getSmoothedDuration() >= 1000)
		{
			if (tDistraction >= 1)
				tDistraction--;
			else
				tDistraction = 0.0; //This is because the variable is rounded when it is displayed, and could show 1.
			startNoDistraction = 0;
		}
	}

	cv::putText(prev_frame, labelAlarm, cv::Point2f(x_truck_i, y_alarm + y_vum + 95), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 2);

	int x_vum_dist = x_truck_i + 135;
	int y_vum_dist = y_alarm + 55;

	//Drawing VU Meters
	// VUmeter Distraction: Rectangle Background
	cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 50, 0), -1);
	cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxWarning), x_vum, y_vum_unit * (maxWarning - maxNormal)), cv::Scalar(0, 50, 50), -1);
	cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxCritical), x_vum, y_vum_unit * (maxCritical - maxWarning)), cv::Scalar(0, 0, 50), -1);

	// VU Meter Logic
	if (tDistraction > maxNormal && pid_da != 0)
		kill(pid_da, SIGUSR1);
	if (tDistraction <= maxNormal)
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - y_vum_unit * tDistraction, x_vum, y_vum_unit * tDistraction), cv::Scalar(0, 255, 0), -1);
		// tDistraction Label
		cv::putText(prev_frame, cv::format("%3.0f", tDistraction), cv::Point2f(x_vum_dist + 30, y_vum_dist + y_vum - y_vum_unit * tDistraction + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}
	else if (tDistraction > maxNormal && tDistraction <= maxWarning)
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 255, 0), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * tDistraction), x_vum, y_vum_unit * (tDistraction - maxNormal)), cv::Scalar(0, 255, 255), -1);
		// tDistraction Label
		cv::putText(prev_frame, cv::format("%3.0f", tDistraction), cv::Point2f(x_vum_dist + 30, y_vum_dist + y_vum - y_vum_unit * tDistraction + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}

	else if (tDistraction > maxWarning && tDistraction <= maxCritical)
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 255, 0), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxWarning), x_vum, y_vum_unit * (maxWarning - maxNormal)), cv::Scalar(0, 255, 255), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * tDistraction), x_vum, y_vum_unit * (tDistraction - maxWarning)), cv::Scalar(0, 0, 255), -1);
		// tDistraction Label
		cv::putText(prev_frame, cv::format("%3.0f", tDistraction), cv::Point2f(x_vum_dist + 30, y_vum_dist + y_vum - y_vum_unit * tDistraction + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}
	else
	{
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxNormal), x_vum, y_vum_unit * maxNormal), cv::Scalar(0, 255, 0), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxWarning), x_vum, y_vum_unit * (maxWarning - maxNormal)), cv::Scalar(0, 255, 255), -1);
		cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist + y_vum - (y_vum_unit * maxCritical), x_vum, y_vum_unit * (maxCritical - maxWarning)), cv::Scalar(0, 0, 255), -1);
		// tDistraction Label
		cv::putText(prev_frame, cv::format("%3.0f", tDistraction), cv::Point2f(x_vum_dist + 30, y_vum_dist + y_vum - (y_vum_unit * maxCritical) + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
	}

	// VUmeter Drowsiness: Rectangle
	cv::rectangle(prev_frame, cv::Rect(x_vum_dist, y_vum_dist, x_vum, y_vum), cv::Scalar(255, 255, 255), 1);
}

// Thread 1: Driver Recognition
void driver_recognition(cv::Mat prev_frame, std::vector<FaceDetection::Result> prev_detection_results, VectorCNN landmarks_detector, VectorCNN face_reid, EmbeddingsGallery face_gallery, std::string *driver_name, int x_truck_i, int y_driver_i)
{
	if (timer["face_identified"].getSmoothedDuration() > 60000.0 && face_identified && firstTime == 1 ||
			timer["face_identified"].getSmoothedDuration() > 1000.0 && !face_identified && firstTime == 1 ||
			firstTime == 0)
	{
		cv::Mat aux_prev_frame = prev_frame.clone();
		
		face_identified = identify_driver(aux_prev_frame, &prev_detection_results, &landmarks_detector, &face_reid, &face_gallery, driver_name);
		
		if (!prev_detection_results.empty())
			cv::rectangle(prev_frame, prev_detection_results[0].location, cv::Scalar(255, 255, 255), 1);
		firstTime = 1;
		timer.start("face_identified");
		//Take Photo
		if (!face_identified && firstPhoto && !face_save.empty()) // Only save the first picture of the "Not Authorized Driver".
		{
			cv::imwrite("../../../drivers/unknown/Unknown-Driver.jpg", face_save);
			firstPhoto = false;
		}
	}

	// Driver Label
	cv::putText(prev_frame, "Driver Information", cv::Point2f(x_truck_i, y_driver_i + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

	if (face_identified)
	{
		cv::putText(prev_frame, "Driver: " + *driver_name, cv::Point2f(x_truck_i, y_driver_i + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
	}
	else if (!face_identified && *driver_name == "Unknown")
	{
		cv::putText(prev_frame, "Driver: " + *driver_name, cv::Point2f(x_truck_i, y_driver_i + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
	}
}

void beeping(Player *beep, bool *finished)
{
	while (!(*finished))
	{
		if (tDrowsiness <= 100)
			beep->setGain(exp((float)((tDrowsiness - 100) / 10)));
		if (tDrowsiness > 40)
		{
			if (!beep->isPlaying())
				beep->play();
		}
	}
}

int headbuttDetection(boost::circular_buffer<double> *angle_p)
{
	boost::circular_buffer<double> &pitch = *angle_p;
	bool ret = false;
	const int full = 5;
	double delta = 0;

	int lim = std::min(full, (int)pitch.size() - 1); // Wait for one frame to calculate speed
	if (lim > 1)
	{
		for (int i = 0; i < lim; i++)
			delta = delta + (pitch[i] - pitch[i + 1]);
		delta = delta / lim;
		if (delta > 6) // magic threshold (adjust)
			ret = true;
	}
	return ret;
}

int main(int argc, char *argv[])
{
    

rclcpp::init(argc, argv);
std::thread truck_data(ros_client, &truck);
fSim = true;


	try
	{
		timer.start("face_identified"); //Initializate timers
		timer.start("headbutt");
		timer.start("send2aws");

		dlib::shape_predictor sp;
		dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		std::vector<dlib::full_object_detection> shapes;

		std::chrono::high_resolution_clock::time_point slp1, slp2;

		float EYE_AR_THRESH = 0.195;
		float MOUTH_EAR_THRESH = 0.65;
		float EYE_AR_CONSEC_FRAMES = 3;
		float MOUTH_EAR_CONSEC_FRAMES = 5;

		bool eye_closed = false;

		int blink_counter = 0;
		int yawn_counter = 0;
		int last_blink_counter = 0;
		int blinl_total = 0;
		int yawn_total = 0;
		boost::circular_buffer<float> ear_5(5);
		boost::circular_buffer<float> ear_5_mouth(5);

		std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

		// ------------------------------ Parsing and validation of input args ---------------------------------
		if (!ParseAndCheckCommandLine(argc, argv))
		{
			return 0;
		}
		// Init drowsiness and distraction levels
		tDrowsiness = FLAGS_init_drow;
		tDistraction = FLAGS_init_dist;
		int pid_da = FLAGS_pid_da;
		slog::info << "Reading input" << slog::endl;
		cv::VideoCapture cap;

		if (FLAGS_i == "cam")
		{
			if (!cap.open(0))
				throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
		}
		else if (!cap.open(FLAGS_i))
		{
			throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
		}

		// Size force
		cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
		cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
		cap.set(cv::CAP_PROP_FPS, 10);

		const size_t width = (size_t)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		const size_t height = (size_t)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

		int x = 200;
		int y = 155;
		int x_truck_i = width - (x + 30);
		int y_truck_i = y + 200;
		int y_driver_i = 5;
		int y_driver = y - 60;

		// read input (video) frame
		cv::Mat frame;
		if (!cap.read(frame))
		{
			throw std::logic_error("Failed to get frame from cv::VideoCapture");
		}
		// -----------------------------------------------------------------------------------------------------
		// --------------------------- 1. Load Plugin for inference engine -------------------------------------
		std::map<std::string, Core> pluginsForDevices;
		std::vector<std::pair<std::string, std::string>> cmdOptions = {
			{FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp},
			{FLAGS_d_em, FLAGS_m_em}, {FLAGS_d_lm, FLAGS_m_lm}, {FLAGS_d_reid, FLAGS_m_reid}};

		for (auto &&option : cmdOptions)
		{
			auto deviceName = option.first;
			auto networkName = option.second;

			if (deviceName == "" || networkName == "")
				continue;

			if (pluginsForDevices.find(deviceName) != pluginsForDevices.end())
				continue;
			slog::info << "Loading plugin " << deviceName << slog::endl;
			Core core;
			/** Load extensions for the CPU plugin **/
			if ((deviceName.find("CPU") != std::string::npos)) {
				if (!FLAGS_l.empty())
				{
					// CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
					#if (OPENVINO_VER==2019)
                        auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    #else
                        IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    #endif
					core.AddExtension(extension_ptr, deviceName);
					slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
				}
				//core.SetConfig({{ CONFIG_KEY(CPU_THROUGHPUT_STREAMS), std::to_string(12) }}, deviceName);
				core.SetConfig({{ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(12) }},deviceName);
				core.SetConfig({{ CONFIG_KEY(CPU_BIND_THREAD), "YES" }}, deviceName);

			}
			else if (!FLAGS_c.empty())
				core.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, deviceName);

			core.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}},deviceName);
			pluginsForDevices[deviceName] = core;
		}

		FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r);
		HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async);
		//	FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async);

		auto fr_model_path = FLAGS_m_reid;
		std::cout << fr_model_path << std::endl;
		auto fr_weights_path = fileNameNoExt(FLAGS_m_reid) + ".bin";
		auto lm_model_path = FLAGS_m_lm;
		auto lm_weights_path = fileNameNoExt(FLAGS_m_lm) + ".bin";

		CnnConfig reid_config(fr_model_path, fr_weights_path);
		reid_config.max_batch_size = 16;
		reid_config.enabled = /*face_config.enabled*/ true && !fr_model_path.empty() && !lm_model_path.empty();
		reid_config.plugin = pluginsForDevices[FLAGS_d_reid];
		reid_config.deviceName = FLAGS_d_reid;
		VectorCNN face_reid(reid_config);

		// Load landmarks detector
		CnnConfig landmarks_config(lm_model_path, lm_weights_path);
		landmarks_config.max_batch_size = 16;
		landmarks_config.enabled = /*face_config.enabled*/ true && reid_config.enabled && !lm_model_path.empty();
		landmarks_config.plugin = pluginsForDevices[FLAGS_d_lm];
		landmarks_config.deviceName = FLAGS_d_lm;
		VectorCNN landmarks_detector(landmarks_config);

		double t_reid = 0.4; // Cosine distance threshold between two vectors for face reidentification.
		EmbeddingsGallery face_gallery(FLAGS_fg, t_reid, landmarks_detector, face_reid);
		// -----------------------------------------------------------------------------------------------------

		// --------------------------- 2. Read IR models and load them to plugins ------------------------------
		// Disable dynamic batching for face detector as long it processes one image at a time.
		Load(faceDetector).into(pluginsForDevices[FLAGS_d], FLAGS_d, false);
		Load(headPoseDetector).into(pluginsForDevices[FLAGS_d_hp], FLAGS_d_hp, FLAGS_dyn_hp);
		// -----------------------------------------------------------------------------------------------------

		// --------------------------- 3. Do inference ---------------------------------------------------------
		// Start inference & calc performance.
		slog::info << "Start inference " << slog::endl;
		if (!FLAGS_no_show)
		{
			std::cout << "Press any key to stop" << std::endl;
		}

		bool isFaceAnalyticsEnabled = headPoseDetector.enabled();

		timer.start("total");

		std::ostringstream out;
		size_t framesCounter = 0;
		bool frameReadStatus;
		bool isLastFrame;
		cv::Mat prev_frame, next_frame;

		// Detect all faces on the first frame and read the next one.
		timer.start("detection");
		faceDetector.enqueue(frame);
		faceDetector.submitRequest();
		timer.finish("detection");

		prev_frame = frame.clone();

		// Read next frame.
		timer.start("video frame decoding");
		frameReadStatus = cap.read(frame);
		timer.finish("video frame decoding");

		//dlib
		//dlib::image_window win, win_faces;

		FaceDetection::Result big_head;
		big_head.label = 0;
		big_head.confidence = 0;
		big_head.location = cv::Rect(0, 0, 0, 0);

		boost::circular_buffer<double> pitch = boost::circular_buffer<double>(5);
		bool headbutt = false;

		bool processing_finished = false;
		Player beep("beep.ogg");
		//std::thread beep_thread(beeping, &beep, &processing_finished);

		Aws::Crt::ApiHandle apiHandle;

		Aws::Crt::String endpoint("a1572pdc8tbdas-ats.iot.us-east-1.amazonaws.com");
		Aws::Crt::String certificatePath("bee7694a31-certificate.pem.crt");
		Aws::Crt::String keyPath("bee7694a31-private.pem.key");
		Aws::Crt::String caFile("AmazonRootCA1.pem");
		Aws::Crt::String topic("drivers/");
		Aws::Crt::String clientId("NEXCOM_device");

		Aws::Crt::Io::EventLoopGroup eventLoopGroup(1);
		if (!eventLoopGroup) {
			fprintf(stderr, "Event Loop Group Creation failed with error %s\n", Aws::Crt::ErrorDebugString(eventLoopGroup.LastError()));
			exit(-1);
		}
		Aws::Crt::Io::DefaultHostResolver defaultHostResolver(eventLoopGroup, 1, 5);
		Aws::Crt::Io::ClientBootstrap bootstrap(eventLoopGroup, defaultHostResolver);
		if (!bootstrap) {
			fprintf(stderr, "ClientBootstrap failed with error %s\n", Aws::Crt::ErrorDebugString(bootstrap.LastError()));
			exit(-1);
		}

		auto clientConfig = Aws::Iot::MqttClientConnectionConfigBuilder(certificatePath.c_str(), keyPath.c_str())
			.WithEndpoint(endpoint)
			.WithCertificateAuthority(caFile.c_str())
			.Build();

		if (!clientConfig) {
			fprintf(stderr, "Client Configuration initialization failed with error %s\n", Aws::Crt::ErrorDebugString(Aws::Crt::LastError()));
			exit(-1);
		}

		Aws::Iot::MqttClient mqttClient(bootstrap);
		if (!mqttClient) {
			fprintf(stderr, "MQTT Client Creation failed with error %s\n", Aws::Crt::ErrorDebugString(mqttClient.LastError()));
			exit(-1);
		}

		auto connection = mqttClient.NewConnection(clientConfig);
		if (!*connection) {
			fprintf(stderr, "MQTT Connection Creation failed with error %s\n", Aws::Crt::ErrorDebugString(connection->LastError()));
			exit(-1);
		}


		std::mutex mutex;
		std::condition_variable conditionVariable;
		bool connectionSucceeded = false;
		bool connectionClosed = false;
		bool connectionCompleted = false;
		bool connectionInterrupted = false;

		/*
		 * This will execute when an mqtt connect has completed or failed.
		 */
		auto onConnectionCompleted = [&](Aws::Crt::Mqtt::MqttConnection &, int errorCode, Aws::Crt::Mqtt::ReturnCode returnCode, bool) {
			if (errorCode)
			{
				fprintf(stdout, "Connection failed with error %s\n", Aws::Crt::ErrorDebugString(errorCode));
				std::lock_guard<std::mutex> lockGuard(mutex);
				connectionSucceeded = false;
			}
			else
			{
				fprintf(stdout, "Connection completed with return code %d\n", returnCode);
				connectionSucceeded = true;
			}
			{
				std::lock_guard<std::mutex> lockGuard(mutex);
				connectionCompleted = true;
			}
			conditionVariable.notify_one();
		};

		auto onInterrupted = [&](Aws::Crt::Mqtt::MqttConnection &, int error) {
			fprintf(stdout, "Connection interrupted with error %s\n", Aws::Crt::ErrorDebugString(error));
			connectionInterrupted = true;
		};

		auto onResumed = [&](Aws::Crt::Mqtt::MqttConnection &, Aws::Crt::Mqtt::ReturnCode, bool) { 
			fprintf(stdout, "Connection resumed\n");
			connectionInterrupted = false;
		};

		/*
		 * Invoked when a disconnect message has completed.
		 */
		auto onDisconnect = [&](Aws::Crt::Mqtt::MqttConnection &conn) {
			{
				fprintf(stdout, "Disconnect completed\n");
				std::lock_guard<std::mutex> lockGuard(mutex);
				connectionClosed = true;
			}
			conditionVariable.notify_one();
		};

		connection->OnConnectionCompleted = std::move(onConnectionCompleted);
		connection->OnDisconnect = std::move(onDisconnect);
		connection->OnConnectionInterrupted = std::move(onInterrupted); //I should set a flag here to try to reconnect, probably
		connection->OnConnectionResumed = std::move(onResumed);

		auto onPublish = [&](Aws::Crt::Mqtt::MqttConnection &, const Aws::Crt::String &topic, const Aws::Crt::ByteBuf &byteBuf) {
			fprintf(stdout, "Publish received on topic %s\n", topic.c_str());
			fprintf(stdout, "\n Message:\n");
			fwrite(byteBuf.buffer, 1, byteBuf.len, stdout);
			fprintf(stdout, "\n");
		};

		/*
		 * Subscribe for incoming publish messages on topic.
		 */
		auto onSubAck = [&](Aws::Crt::Mqtt::MqttConnection &, uint16_t packetId, const Aws::Crt::String &topic, Aws::Crt::Mqtt::QOS, int errorCode) {
			if (packetId)
			{
				fprintf(stdout, "Subscribe on topic %s on packetId %d Succeeded\n", topic.c_str(), packetId);
			}
			else
			{
				fprintf(stdout, "Subscribe failed with error %s\n", aws_error_debug_str(errorCode));
			}
			conditionVariable.notify_one();
		};


		/*
		 * Actually perform the connect dance.
		 * This will use default ping behavior of 1 hour and 3 second timeouts.
		 * If you want different behavior, those arguments go into slots 3 & 4.
		 */
		fprintf(stdout, "Connecting...\n");
		if (!connection->Connect(clientId.c_str(), false, 20)) {
			fprintf(stderr, "MQTT Connection failed with error %s\n", Aws::Crt::ErrorDebugString(connection->LastError()));
			exit(-1);
		}

		std::unique_lock<std::mutex> uniqueLock(mutex);
		conditionVariable.wait(uniqueLock, [&]() { return connectionCompleted; });

		if (connectionSucceeded) {
			
			connection->Subscribe(topic.c_str(), AWS_MQTT_QOS_AT_MOST_ONCE, onPublish, onSubAck);
			
			while (true)
			{	
				picojson::value v;
				picojson::value v1;
				framesCounter++;
				isLastFrame = !frameReadStatus;

				timer.start("detection");
				// Retrieve face detection results for previous frame.
				faceDetector.wait();
				faceDetector.fetchResults();
				auto prev_detection_results = faceDetector.results;
				if (!prev_detection_results.empty())
				{
					for (int i = 0; i < prev_detection_results.size(); i++)
					{
						if (big_head.location.area() < prev_detection_results[i].location.area())
						{
							big_head = prev_detection_results[i];
							biggest_head = i;
						}
					}
					prev_detection_results.clear();
					prev_detection_results.push_back(big_head);
					big_head.label = 0;
					big_head.confidence = 0;
					big_head.location = cv::Rect(0, 0, 0, 0);
				}
				// No valid frame to infer if previous frame is last.
				if (!isLastFrame)
				{
					faceDetector.enqueue(frame);
					faceDetector.submitRequest();
				}
				timer.finish("detection");

				timer.start("data preprocessing");
				// Fill inputs of face analytics networks.
				for (auto &&face : prev_detection_results)
				{
					if (isFaceAnalyticsEnabled)
					{
						auto clippedRect = face.location & cv::Rect(0, 0, width, height);
						cv::Mat face = prev_frame(clippedRect);
						face_save = frame(clippedRect);
						headPoseDetector.enqueue(face);
					}
				}
				timer.finish("data preprocessing");

				// Run age-gender recognition, head pose estimation and emotions recognition simultaneously.
				timer.start("face analytics call");
				if (isFaceAnalyticsEnabled)
				{
					headPoseDetector.submitRequest();
				}
				timer.finish("face analytics call");

				// Read next frame if current one is not last.
				if (!isLastFrame)
				{
					timer.start("video frame decoding");
					frameReadStatus = cap.read(next_frame);
					timer.finish("video frame decoding");
				}
				if (!frameReadStatus) {
					timer.finish("total");
					break;
				}

				timer.start("face analytics wait");
				if (isFaceAnalyticsEnabled)
				{
					headPoseDetector.wait();
				}
				timer.finish("face analytics wait");

				// Visualize results.
				if (!FLAGS_no_show)
				{
					TrackedObjects tracked_face_objects;
					timer.start("visualization");
					out.str("");
					out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
						<< (timer["video frame decoding"].getSmoothedDuration() +
								timer["visualization"].getSmoothedDuration())
						<< " ms";
					cv::putText(prev_frame, out.str(), cv::Point2f(10, 25), cv::FONT_HERSHEY_TRIPLEX, 0.4,
							cv::Scalar(255, 0, 0));

					out.str("");
					out << "Face detection time: " << std::fixed << std::setprecision(2)
						<< timer["detection"].getSmoothedDuration()
						<< " ms ("
						<< 1000.F / (timer["detection"].getSmoothedDuration())
						<< " fps)";
					cv::putText(prev_frame, out.str(), cv::Point2f(10, 45), cv::FONT_HERSHEY_TRIPLEX, 0.4,
							cv::Scalar(255, 0, 0));

					out.str("");
					out << "Total image throughput: "          
						<< framesCounter * (1000.F / timer["total"].getSmoothedDuration())
						<< " FPS";
					cv::putText(prev_frame, out.str(), cv::Point2f(10, 65), cv::FONT_HERSHEY_TRIPLEX, 0.4,
							cv::Scalar(0, 255, 0));

					if (isFaceAnalyticsEnabled)
					{
						out.str("");
						out << "Face Analysics Networks "
							<< "time: " << std::fixed << std::setprecision(2)
							<< timer["face analytics call"].getSmoothedDuration() +
							timer["face analytics wait"].getSmoothedDuration()
							<< " ms ";
						if (!prev_detection_results.empty())
						{
							out << "("
								<< 1000.F / (timer["face analytics call"].getSmoothedDuration() +
										timer["face analytics wait"].getSmoothedDuration())
								<< " fps)";
						}
						cv::putText(prev_frame, out.str(), cv::Point2f(10, 85), cv::FONT_HERSHEY_TRIPLEX, 0.4,
								cv::Scalar(255, 0, 0));

						
								
					}
					
					if ((truck.getEngine() && fSim) || !fSim) // Detect if Engine = ON and Simulator Flag
					{ 
						// Thread 1: Driver Recognition
						timer.start("land marks");
						std::thread thread_recognition(driver_recognition, prev_frame, prev_detection_results, landmarks_detector, face_reid, face_gallery, &driver_name, x_truck_i, y_driver_i);
						timer.finish("land marks");

						out.str("");
						out << "Test "
							<< "time: " << std::fixed << std::setprecision(2)
							<< timer["land marks"].getSmoothedDuration()
							<< " ms ";
						if (timer["land marks"].getSmoothedDuration()>0)
						{
							
							out << "("
								<< 1000.F / timer["land marks"].getSmoothedDuration()
								<< " fps)";
								
						}

						cv::putText(prev_frame, out.str(), cv::Point2f(10, 105), cv::FONT_HERSHEY_TRIPLEX, 0.4,
								cv::Scalar(255, 0, 0));
								
						// Driver Label (CHECK! -> Not here)
						cv::rectangle(prev_frame, cv::Rect(width - (x + 40), y_driver_i, x + 20, y_driver), cv::Scalar(0, 0, 0), -1);
						cv::rectangle(prev_frame, cv::Rect(width - (x + 40), y_driver_i, x + 20, y_driver), cv::Scalar(255, 255, 255), 2);

						// For every detected face.
						int ii = 0;
						std::vector<cv::Point2f> left_eye;
						std::vector<cv::Point2f> right_eye;
						std::vector<cv::Point2f> mouth;
						for (auto &result : prev_detection_results)
						{
							cv::Rect rect = result.location;

							out.str("");
							cv::rectangle(prev_frame, rect, cv::Scalar(255, 255, 255), 1);
							if (FLAGS_dlib_lm)
							{
								float scale_factor_x = 0.15;
								float scale_factor_y = 0.20;
								cv::Rect aux_rect = cv::Rect(rect.x + scale_factor_x * rect.width, rect.y + scale_factor_y * rect.height, rect.width * (1 - 2 * scale_factor_x), rect.height * (1 - scale_factor_y));
								//dlib facial landmarks
								dlib::array2d<dlib::rgb_pixel> img;
								dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(prev_frame));
								dlib::rectangle det = openCVRectToDlib(aux_rect);
								dlib::full_object_detection shape = sp(img, det);
                            for (int i = 0; i < shape.num_parts(); i++)
                            {
                                if (i >= 36 && i <= 41)
                                {
                                    left_eye.push_back(cv::Point2l(shape.part(i).x(), shape.part(i).y()));
                                    cv::circle(prev_frame, cv::Point2l(shape.part(i).x(), shape.part(i).y()), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);
                                }
                                if (i >= 42 && i <= 47)
                                {
                                    right_eye.push_back(cv::Point2l(shape.part(i).x(), shape.part(i).y()));
                                    cv::circle(prev_frame, cv::Point2l(shape.part(i).x(), shape.part(i).y()), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);
                                }
                                //48 - 54. 50 - 58. 52 - 56.

                                if (i == 48 || i == 54 || i == 50 || i == 58 || i == 52 || i == 56)
                                {
                                    mouth.push_back(cv::Point2l(shape.part(i).x(), shape.part(i).y()));
                                    cv::circle(prev_frame, cv::Point2l(shape.part(i).x(), shape.part(i).y()), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);
                                }
                            }
                            float ear_left = 0;
                            float ear_right = 0;
                            float ear = 0;
                            ear_left = (distanceAtoB(left_eye[1], left_eye[5]) + distanceAtoB(left_eye[2], left_eye[4])) / (2 * distanceAtoB(left_eye[0], left_eye[3]));
                            ear_right = (distanceAtoB(right_eye[1], right_eye[5]) + distanceAtoB(right_eye[2], right_eye[4])) / (2 * distanceAtoB(right_eye[0], right_eye[3]));
                            ear = (ear_left + ear_right) / 2;
                            ear_5.push_front(ear);
                            float ear_avg = 0;
                            for (auto &&i : ear_5)
                            {
                                ear_avg = ear_avg + i;
                            }
                            ear_avg = ear_avg / ear_5.size();
                            if (ear_avg < EYE_AR_THRESH)
                            {
                                // Blink Logic
                                vBlink = 1;
                                if (firstBlink)
                                {
                                    timeBlink = 0;
                                    firstBlink = false;
                                }
                                else
                                {
                                    timeBlink = timer["timeBlink"].getSmoothedDuration() - timeBlink;
                                }
                                timer.start("timeBlink");
                                // End Blink Logic

                                blink_counter += 1;
                                if (blink_counter >= 90)
                                    eye_closed = true;
                            }
                            else
                            {
                                if (blink_counter >= EYE_AR_CONSEC_FRAMES)
                                {
                                    blinl_total += 1;
                                    last_blink_counter = blink_counter;
                                }
                                blink_counter = 0;

                                // Blink Logic
                                vBlink = 0;
                                timeBlink = 0;
                                firstBlink = true;
                                // End Blink Logic
                            }

                            cv::putText(prev_frame, "Blinks: " + std::to_string(blinl_total), cv::Point2f(x_truck_i, y_driver_i + 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                            //Yawn detection
                            float ear_mouth = (distanceAtoB(mouth[1], mouth[5]) + distanceAtoB(mouth[2], mouth[4])) / (2 * distanceAtoB(mouth[0], mouth[3]));
                            ear_5_mouth.push_front(ear_mouth);
                            float ear_avg_mouth = 0;
                            for (auto &&i : ear_5_mouth)
                            {
                                ear_avg_mouth = ear_avg_mouth + i;
                            }
                            ear_avg_mouth = ear_avg_mouth / ear_5_mouth.size();
                            if (ear_avg_mouth > MOUTH_EAR_THRESH)
                            {
                                yawn_counter += 1;
                            }
                            else
                            {
                                if (yawn_counter >= MOUTH_EAR_CONSEC_FRAMES)
                                {
                                    vYawn = 1;
                                    yawn_total += 1;
                                }
                                yawn_counter = 0;
                            }
                            cv::putText(prev_frame, "Yawns: " + std::to_string(yawn_total), cv::Point2f(x_truck_i, y_driver_i + 80), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                            }

                        cv::putText(prev_frame,
                                    out.str(),
                                    cv::Point2f(result.location.x, result.location.y - 15),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.8,
                                    cv::Scalar(0, 0, 255));

                        if (headPoseDetector.enabled() && ii < headPoseDetector.maxBatch)
                        {
                            if (FLAGS_r)
                            {
                                std::cout << "Head pose results: yaw, pitch, roll = "
                                          << headPoseDetector[ii].angle_y << ";"
                                          << headPoseDetector[ii].angle_p << ";"
                                          << headPoseDetector[ii].angle_r << std::endl;
                            }
                            cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
                            headPoseDetector.drawAxes(prev_frame, center, headPoseDetector[ii], 50);
                            pitch.push_front(headPoseDetector[ii].angle_p);
                            headbutt = headbuttDetection(&pitch);

                            int is_dist = isDistracted(headPoseDetector[ii].angle_y, headPoseDetector[ii].angle_p, headPoseDetector[ii].angle_r);

                            // Alarm Label
                            int x_alarm = width - (x + 20) - 20;
                            int y_alarm = y_driver_i + y_driver + 10;
                            cv::rectangle(prev_frame, cv::Rect(x_alarm, y_alarm, x + 20, y + 100), cv::Scalar(0, 0, 0), -1);
                            cv::rectangle(prev_frame, cv::Rect(x_alarm, y_alarm, x + 20, y + 100), cv::Scalar(255, 255, 255), 2);

                            cv::putText(prev_frame, "Alarms", cv::Point2f(x_truck_i, y_alarm + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                            cv::putText(prev_frame, "Drowsiness | Distraction", cv::Point2f(x_truck_i, y_alarm + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                            cv::putText(prev_frame, "Description", cv::Point2f(x_truck_i, y_alarm + y_vum + 75), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                            // Thread: Drowsiness Alarm
                            std::thread thread_drowsiness(alarmDrowsiness, prev_frame, yawn_total, blinl_total, width, height, x_alarm, y_alarm, x_truck_i, headbutt);
                            std::thread thread_distraction(alarmDistraction, prev_frame, is_dist, y_alarm, x_truck_i, pid_da);
                            // Thread: Drowsiness Alarm
                            thread_drowsiness.join();
                            thread_distraction.join();
                        }
                        ii++;
                    }

                    // Truck Label
                    
                    cv::rectangle(prev_frame, cv::Rect(width - (x + 40), y_truck_i + 20, x + 20, y + 130), cv::Scalar(0, 0, 0), -1);
                    cv::rectangle(prev_frame, cv::Rect(width - (x + 40), y_truck_i + 20, x + 20, y + 130), cv::Scalar(255, 255, 255), 2);

                    cv::putText(prev_frame, "Truck Information", cv::Point2f(x_truck_i, y_truck_i + 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                    
                    if (truck.getEngine())
                        cv::putText(prev_frame, "Engine: ON", cv::Point2f(x_truck_i, y_truck_i + 60), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1.8);
                    else
                        cv::putText(prev_frame, "Engine: OFF", cv::Point2f(x_truck_i, y_truck_i + 60), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.8);
                    
                    if (truck.getParkingBrake())
                        cv::putText(prev_frame, "GearStatus: Parking", cv::Point2f(x_truck_i, y_truck_i + 75), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.2);
                    else if (truck.getSpeed() < -0.03)
                        cv::putText(prev_frame, "GearStatus: Reverse", cv::Point2f(x_truck_i, y_truck_i + 75), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    else if (truck.getSpeed() > 0.03)
                        cv::putText(prev_frame, "GearStatus: Driving", cv::Point2f(x_truck_i, y_truck_i + 75), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    else
                        cv::putText(prev_frame, "GearStatus: Stopped", cv::Point2f(x_truck_i, y_truck_i + 75), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    
                    if (truck.getTrailer())
                        cv::putText(prev_frame, "Trailer: ON", cv::Point2f(x_truck_i, y_truck_i + 90), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1.2);
                    else
                        cv::putText(prev_frame, "Trailer: OFF", cv::Point2f(x_truck_i, y_truck_i + 90), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255), 1.2);
                    
                    cv::putText(prev_frame, cv::format("Speed (Km/h): %3.2f", (truck.getSpeed())), cv::Point2f(x_truck_i, y_truck_i + 105), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, "RPM: " + std::to_string(truck.getRpm()), cv::Point2f(x_truck_i, y_truck_i + 120), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, "Gear: " + std::to_string(truck.getGear()), cv::Point2f(x_truck_i, y_truck_i + 135), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    
                    if (truck.getCruiseControl() > 0.03)
                        cv::putText(prev_frame, cv::format("Cruice (Km/h): %3.2f", truck.getCruiseControl() * 3.6), cv::Point2f(x_truck_i, y_truck_i + 150), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    else
                        cv::putText(prev_frame, "Cruice: OFF", cv::Point2f(x_truck_i, y_truck_i + 150), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);

                    cv::putText(prev_frame, cv::format("Air Pressure (psi): %3.2f", truck.getAirPressure()), cv::Point2f(x_truck_i, y_truck_i + 165), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Battery (V): %3.2f", truck.getBattery()), cv::Point2f(x_truck_i, y_truck_i + 180), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Fuel (l): %3.2f", truck.getFuel()), cv::Point2f(x_truck_i, y_truck_i + 195), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Fuel Average (l/km): %3.2f", truck.getFuelAverage()), cv::Point2f(x_truck_i, y_truck_i + 210), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Cargo Mass (Kg): %3.2f", truck.getCargoMass()), cv::Point2f(x_truck_i, y_truck_i + 225), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Wheel Wear: %3.2f", truck.getWearWheels() * 100), cv::Point2f(x_truck_i, y_truck_i + 240), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Trailer Wear: %3.2f", truck.getWearChassis() * 100), cv::Point2f(x_truck_i, y_truck_i + 255), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Engine Wear: %3.2f", truck.getWearEngine() * 100), cv::Point2f(x_truck_i, y_truck_i + 270), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    cv::putText(prev_frame, cv::format("Transmission Wear: %3.2f", truck.getWearTransmission() * 100), cv::Point2f(x_truck_i, y_truck_i + 285), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1.2);
                    
                    // End Thread 1: Driver Recognition
                    thread_recognition.join();

                }

                // Sample of Results
                cv::imshow("Detection results", prev_frame);
                timer.finish("visualization");
            }

            // Thread: Send Data to AWS
            //std::thread senddata_thread(send2aws, topic, connection);

            if (timer["send2aws"].getSmoothedDuration() > 500.0){
                timer.start("send2aws");
                picojson::value v;
                picojson::value v1;
                
                v.set<picojson::object>(picojson::object());
                v1.set<picojson::object>(picojson::object());

                unsigned long milliseconds_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                v1.get<picojson::object>()["timestamp"] = picojson::value((double)milliseconds_time);
                v1.get<picojson::object>()["name"] = picojson::value(driver_name);
                v1.get<picojson::object>()["drowsiness"] = picojson::value(tDrowsiness);
                v1.get<picojson::object>()["distraction"] = picojson::value(tDistraction);
                double dangMap = 0; // This variable shows the highest value.
                if (tDrowsiness >= tDistraction) dangMap = tDrowsiness;
                    else dangMap = tDistraction;
                v1.get<picojson::object>()["dangMap"] = picojson::value(dangMap);
                //          v.get<picojson::object>()["driver"].set<picojson::array>(picojson::array());
                //  	    v.get<picojson::object>()["driver"].get<picojson::array>().push_back(v1);
                //  	    v.get<picojson::object>()["driver"] = v1;
                
                // Truck Information 
                //v1.get<picojson::object>()["location"] = picojson::value(std::to_string(pos_lat)+","+std::to_string(pos_lon));
		        v1.get<picojson::object>()["engine"] = picojson::value(truck.getEngine());
                v1.get<picojson::object>()["trailer_connected"] = picojson::value(truck.getTrailer());
                //v1.get<picojson::object>()["speed"] = picojson::value(std::to_string(100.0));
                v1.get<picojson::object>()["rpm"] = picojson::value(std::to_string(truck.getRpm()));
                v1.get<picojson::object>()["gear"] = picojson::value(std::to_string(truck.getGear()));
                v1.get<picojson::object>()["cruise_control"] = picojson::value(truck.getCruiseControl());
                v1.get<picojson::object>()["air_pressure"] = picojson::value(truck.getAirPressure());
                v1.get<picojson::object>()["battery_voltage"] = picojson::value(truck.getBattery());
                v1.get<picojson::object>()["fuel"] = picojson::value(truck.getFuel());
                v1.get<picojson::object>()["fuel_average_consumption"] = picojson::value(truck.getFuelAverage());
                v1.get<picojson::object>()["cargo_mass"] = picojson::value(truck.getCargoMass());
                v1.get<picojson::object>()["wear_wheels"] = picojson::value(truck.getWearWheels() * 100);
                v1.get<picojson::object>()["wear_chassis"] = picojson::value(truck.getWearChassis() * 100);
                v1.get<picojson::object>()["wear_engine"] = picojson::value(truck.getWearEngine() * 100);
                v1.get<picojson::object>()["wear_transmission"] = picojson::value(truck.getWearTransmission() * 100);


                std::string input = picojson::value(v1).serialize();
                Aws::Crt::ByteBuf payload = Aws::Crt::ByteBufNewCopy(Aws::Crt::DefaultAllocator(), (const uint8_t *)input.data(), input.length());
                Aws::Crt::ByteBuf *payloadPtr = &payload;


                auto onPublishComplete = [payloadPtr](Aws::Crt::Mqtt::MqttConnection &, uint16_t packetId, int errorCode) {
                    aws_byte_buf_clean_up(payloadPtr);

                    if (packetId)
                    {
                        fprintf(stdout, "Operation on packetId %d Succeeded\n", packetId);
                    }
                    else
                    {
                        fprintf(stdout, "Operation failed with error %s\n", aws_error_debug_str(errorCode));
                    }
                };
				if(connectionInterrupted == false){
					connection->Publish(topic.c_str(), AWS_MQTT_QOS_AT_MOST_ONCE, false, payload, onPublishComplete); 
				}
 
            }

            // End of file (or a single frame file like an image). We just keep last frame displayed to let user check what was shown
            if (isLastFrame)
            {
                timer.finish("total");
                if (!FLAGS_no_wait)
                {
                    std::cout << "No more frames to process. Press any key to exit" << std::endl;
                    cv::waitKey(0);
                }
                break;
            }
            else if (!FLAGS_no_show && -1 != cv::waitKey(1))
            {
                timer.finish("total");
                break;
            }

            prev_frame = frame;
            frame = next_frame;
            next_frame = cv::Mat();

            //senddata_thread.join();

        }

        connection->Unsubscribe(
            topic.c_str(), [&](Aws::Crt::Mqtt::MqttConnection &, uint16_t, int) { conditionVariable.notify_one(); });
        conditionVariable.wait(uniqueLock);
        }
        processing_finished = true;
        //beep_thread.join();
        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.F / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        if (connection->Disconnect())
        {
            conditionVariable.wait(uniqueLock, [&]() { return connectionClosed; });
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception &error)
    {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...)
    {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;

#ifdef SIMULATOR
    std::terminate();
#endif

    return 0;
}
