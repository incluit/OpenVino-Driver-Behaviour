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

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "customflags.hpp"
#include "detectors.hpp"

#include <ie_iextension.h>
#include <ext_list.hpp>

#include <opencv2/opencv.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <boost/circular_buffer.hpp>

using namespace InferenceEngine;
static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

float distanceAtoB(cv::Point2f A, cv::Point2f B){
    float distance_l = sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y));
    return distance_l;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_n_ag < 1) {
        throw std::logic_error("Parameter -n_ag cannot be 0");
    }

    if (FLAGS_n_hp < 1) {
        throw std::logic_error("Parameter -n_hp cannot be 0");
    }

    // no need to wait for a key press from a user if an output image/video file is not shown.
    FLAGS_no_wait |= FLAGS_no_show;

    return true;
}

enum distractionLevel {
	NOT_DISTRACTED = 0,
	DISTRACTED,
	PHONE,
};

int isDistracted (float y, float p, float r)
{
	int result = 0;
	if (abs(y) > 30 || abs(p) > 30) {
		if (abs(y) > 20 && p > 10 && r < 0)
			result = PHONE;
		else
			result = DISTRACTED;
	}
	return result;
}

int main(int argc, char *argv[]) {
    try {

        dlib::shape_predictor sp;
        dlib::deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;
        std::vector<dlib::full_object_detection> shapes;
        float EYE_AR_THRESH = 0.195;
        float MOUTH_EAR_THRESH = 0.65;
        float EYE_AR_CONSEC_FRAMES = 3;
        float MOUTH_EAR_CONSEC_FRAMES = 5;

        std::chrono::high_resolution_clock::time_point slp1,slp2;
        bool eye_closed = false;

        int blink_counter = 0;
        int yawn_counter = 0;
        int last_blink_counter = 0;
        int last_yawn_counter = 0;
        int blinl_total = 0;
        int yawn_total = 0;
        boost::circular_buffer<float> ear_5(5);
        boost::circular_buffer<float> ear_5_mouth(5);

        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        const bool isCamera = FLAGS_i == "cam";
        if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // read input (video) frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        std::map<std::string, InferencePlugin> pluginsForDevices;
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_ag, FLAGS_m_ag}, {FLAGS_d_hp, FLAGS_m_hp},
            {FLAGS_d_em, FLAGS_m_em}, {FLAGS_d_lm, FLAGS_m_lm}
        };
        FaceDetection faceDetector(FLAGS_m, FLAGS_d, 1, false, FLAGS_async, FLAGS_t, FLAGS_r);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async);

        for (auto && option : cmdOptions) {
            auto deviceName = option.first;
            auto networkName = option.second;

            if (deviceName == "" || networkName == "") {
                continue;
            }

            if (pluginsForDevices.find(deviceName) != pluginsForDevices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << deviceName << slog::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(deviceName);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            /** Load extensions for the CPU plugin **/
            if ((deviceName.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            pluginsForDevices[deviceName] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForDevices) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        // Disable dynamic batching for face detector as long it processes one image at a time.
        Load(faceDetector).into(pluginsForDevices[FLAGS_d], false);
        Load(headPoseDetector).into(pluginsForDevices[FLAGS_d_hp], FLAGS_dyn_hp);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        // Start inference & calc performance.
        slog::info << "Start inference " << slog::endl;
        if (!FLAGS_no_show) {
            std::cout << "Press any key to stop" << std::endl;
        }

        bool isFaceAnalyticsEnabled = headPoseDetector.enabled();

        Timer timer;
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

        while (true) {
            framesCounter++;
            isLastFrame = !frameReadStatus;

            timer.start("detection");
            // Retrieve face detection results for previous frame.
            faceDetector.wait();
            faceDetector.fetchResults();
            auto prev_detection_results = faceDetector.results;

            // No valid frame to infer if previous frame is last.
            if (!isLastFrame) {
                faceDetector.enqueue(frame);
                faceDetector.submitRequest();
            }
            timer.finish("detection");

            timer.start("data preprocessing");
            // Fill inputs of face analytics networks.
            for (auto &&face : prev_detection_results) {
                if (isFaceAnalyticsEnabled) {
                    auto clippedRect = face.location & cv::Rect(0, 0, width, height);
                    cv::Mat face = prev_frame(clippedRect);
                    headPoseDetector.enqueue(face);
                }
            }
            timer.finish("data preprocessing");

            // Run age-gender recognition, head pose estimation and emotions recognition simultaneously.
            timer.start("face analytics call");
            if (isFaceAnalyticsEnabled) {
                headPoseDetector.submitRequest();
            }
            timer.finish("face analytics call");

            // Read next frame if current one is not last.
            if (!isLastFrame) {
                timer.start("video frame decoding");
                frameReadStatus = cap.read(next_frame);
                timer.finish("video frame decoding");
            }

            timer.start("face analytics wait");
            if (isFaceAnalyticsEnabled) {
                headPoseDetector.wait();
            }
            timer.finish("face analytics wait");

            // Visualize results.
            if (!FLAGS_no_show) {
                timer.start("visualization");
                out.str("");
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (timer["video frame decoding"].getSmoothedDuration() +
                       timer["visualization"].getSmoothedDuration())
                    << " ms";
                cv::putText(prev_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                            cv::Scalar(255, 0, 0));

                out.str("");
                out << "Face detection time: " << std::fixed << std::setprecision(2)
                    << timer["detection"].getSmoothedDuration()
                    << " ms ("
                    << 1000.f / (timer["detection"].getSmoothedDuration())
                    << " fps)";
                cv::putText(prev_frame, out.str(), cv::Point2f(0, 45), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                            cv::Scalar(255, 0, 0));

                if (isFaceAnalyticsEnabled) {
                    out.str("");
                    out << "Face Analysics Networks "
                        << "time: " << std::fixed << std::setprecision(2)
                        << timer["face analytics call"].getSmoothedDuration() +
                           timer["face analytics wait"].getSmoothedDuration()
                        << " ms ";
                    if (!prev_detection_results.empty()) {
                        out << "("
                            << 1000.f / (timer["face analytics call"].getSmoothedDuration() +
                               timer["face analytics wait"].getSmoothedDuration())
                            << " fps)";
                    }
                    cv::putText(prev_frame, out.str(), cv::Point2f(0, 65), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
                }

                // For every detected face.
                int i = 0;
                std::vector<cv::Point2f> left_eye, right_eye, mouth;
                for (auto &result : prev_detection_results) {
                    cv::Rect rect = result.location;

                    out.str("");
                    if(FLAGS_dlib_lm){
                        float scale_factor_x = 0.15;
                        float scale_factor_y = 0.20;
                        cv::Rect aux_rect = cv::Rect(rect.x + scale_factor_x *rect.width, rect.y + scale_factor_y * rect.height, rect.width * (1- 2 * scale_factor_x), rect.height * (1 - scale_factor_y));
                        //dlib facial landmarks
                        dlib::array2d<dlib::rgb_pixel> img;
                        dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(prev_frame));
                        dlib::rectangle det = openCVRectToDlib(aux_rect);
                        dlib::full_object_detection shape = sp(img, det);
                        for(int i = 0; i < shape.num_parts(); i++){
                            if(i >= 36 && i <= 41)
                                left_eye.push_back(cv::Point2l(shape.part(i).x(), shape.part(i).y()));
                            if(i >= 42 && i <= 47)
                                right_eye.push_back(cv::Point2l(shape.part(i).x(), shape.part(i).y()));
                            //cv::circle(prev_frame, cv::Point2l(shape.part(i).x(), shape.part(i).y()), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);
                            //48 - 54. 50 - 58. 52 - 56.

                            if(i == 48 || i == 54 || i ==  50 || i == 58 || i == 52 || i == 56){
                                mouth.push_back(cv::Point2l(shape.part(i).x(), shape.part(i).y()));
                                cv::circle(prev_frame, cv::Point2l(shape.part(i).x(), shape.part(i).y()), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);
                            }
                        }
                        cv::rectangle(prev_frame, aux_rect, cv::Scalar(255,255,255), 1);
                        float ear_left = 0;
                        float ear_right = 0;
                        float ear = 0;
                        ear_left = (distanceAtoB(left_eye[1], left_eye[5]) + distanceAtoB(left_eye[2], left_eye[4])) / (2 * distanceAtoB(left_eye[0],left_eye[3]));
                        ear_right = (distanceAtoB(right_eye[1], right_eye[5]) + distanceAtoB(right_eye[2], right_eye[4])) / (2 * distanceAtoB(right_eye[0],right_eye[3]));
                        ear = (ear_left + ear_right) / 2;
                        ear_5.push_front(ear);
                        float ear_avg = 0;
                        for(auto && i : ear_5){
                            ear_avg = ear_avg + i;
                        }
                        ear_avg = ear_avg / ear_5.size();
                        if(ear_avg < EYE_AR_THRESH){
                            blink_counter += 1;
                            if(blink_counter >= 90)                            
                                eye_closed = true;
                        }else {
                            if(blink_counter >= EYE_AR_CONSEC_FRAMES){
                                blinl_total += 1;
                                last_blink_counter = blink_counter;
                            }
                            blink_counter = 0; 
                        }
                        if(eye_closed)
                            cv::putText(frame, "DANGER", cv::Point2f(50, 250), cv::FONT_HERSHEY_SIMPLEX, 5, cv::Scalar(0, 0, 255), 5);
                        cv::putText(frame, "Blink time: " + std::to_string(last_blink_counter) + " frames", cv::Point2f(10, 130),cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
                        cv::putText(frame, "Blinks: " + std::to_string(blinl_total), cv::Point2f(10, 100),cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
                        //cv::putText(frame, "EAR: " + std::to_string(ear_avg), cv::Point2f(300, 100), cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);  

                        //Yawn detection
                        float ear_mouth = (distanceAtoB(mouth[1], mouth[5]) + distanceAtoB(mouth[2], mouth[4])) / (2 * distanceAtoB(mouth[0],mouth[3]));
                        ear_5_mouth.push_front(ear_mouth);
                        float ear_avg_mouth = 0;
                        for(auto && i : ear_5_mouth){
                            ear_avg_mouth = ear_avg_mouth + i;
                        }
                        ear_avg_mouth = ear_avg_mouth / ear_5_mouth.size();
                        if(ear_avg_mouth > MOUTH_EAR_THRESH){
                            yawn_counter += 1;
                        }else {
                            if(yawn_counter >= MOUTH_EAR_CONSEC_FRAMES){
                                yawn_total += 1;
                                last_yawn_counter = yawn_counter;
                            }
                            yawn_counter = 0; 
                        }
                        cv::putText(frame, "Yawn time: " + std::to_string(last_yawn_counter) + " frames", cv::Point2f(500, 130),cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
                        cv::putText(frame, "Yawns: " + std::to_string(yawn_total), cv::Point2f(10, 160),cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
                        cv::putText(frame, "EAR: " + std::to_string(ear_avg_mouth), cv::Point2f(10, 190), cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);  
                    }

                    cv::putText(prev_frame,
                                out.str(),
                                cv::Point2f(result.location.x, result.location.y - 15),
                                cv::FONT_HERSHEY_COMPLEX_SMALL,
                                0.8,
                                cv::Scalar(0, 0, 255));

                    if (headPoseDetector.enabled() && i < headPoseDetector.maxBatch) {
                        if (FLAGS_r) {
                            std::cout << "Head pose results: yaw, pitch, roll = "
                                      << headPoseDetector[i].angle_y << ";"
                                      << headPoseDetector[i].angle_p << ";"
                                      << headPoseDetector[i].angle_r << std::endl;
                        }
                        cv::Point3f center(rect.x + rect.width / 2, rect.y + rect.height / 2, 0);
                        headPoseDetector.drawAxes(prev_frame, center, headPoseDetector[i], 50);
                        int is_dist = isDistracted(headPoseDetector[i].angle_y, headPoseDetector[i].angle_p, headPoseDetector[i].angle_r);
                        if (is_dist) {
                            std::string distracted_str = "";
                            switch (is_dist) {
                            case DISTRACTED:
                                distracted_str = "WATCH THE ROAD!";
                                break;
                            case PHONE:
                                distracted_str = "STOP LOOKING AT YER PHONE!";
                                break;
                            default:
                                break;
                            }
                            cv::putText(frame, distracted_str, cv::Point2f(50, 200),cv::FONT_HERSHEY_SIMPLEX, 2,cv::Scalar(0, 0, 255), 5);
                        }
                    }
                    i++;
                }
                cv::imshow("Detection results", prev_frame);
                timer.finish("visualization");
            }

            // End of file (or a single frame file like an image). We just keep last frame displayed to let user check what was shown
            if (isLastFrame) {
                timer.finish("total");
                if (!FLAGS_no_wait) {
                    std::cout << "No more frames to process. Press any key to exit" << std::endl;
                    cv::waitKey(0);
                }
                break;
            } else if (!FLAGS_no_show && -1 != cv::waitKey(1)) {
                timer.finish("total");
                break;
            }

            prev_frame = frame;
            frame = next_frame;
            next_frame = cv::Mat();
        }

        slog::info << "Number of processed frames: " << framesCounter << slog::endl;
        slog::info << "Total image throughput: " << framesCounter * (1000.f / timer["total"].getTotalDuration()) << " fps" << slog::endl;

        // Show performace results.
        if (FLAGS_pc) {
            faceDetector.printPerformanceCounts();
            headPoseDetector.printPerformanceCounts();
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}


