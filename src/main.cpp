/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* \brief The entry point for the Inference Engine interactive_face_detection demo application
* \file interactive_face_detection_demo/main.cpp
* \example interactive_face_detection_demo/main.cpp
*/
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


int main(int argc, char *argv[]) {
    try {

        dlib::shape_predictor sp;
        dlib::deserialize("../data/shape_predictor_68_face_landmarks.dat") >> sp;
        std::vector<dlib::full_object_detection> shapes;
        float EYE_AR_THRESH = 0.195;
        float EYE_AR_CONSEC_FRAMES = 3;
        std::chrono::high_resolution_clock::time_point slp1,slp2;

        int blink_counter = 0;
        int blinl_total = 0;
        boost::circular_buffer<float> ear_5(5);
        //dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

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
        AgeGenderDetection ageGenderDetector(FLAGS_m_ag, FLAGS_d_ag, FLAGS_n_ag, FLAGS_dyn_ag, FLAGS_async);
        HeadPoseDetection headPoseDetector(FLAGS_m_hp, FLAGS_d_hp, FLAGS_n_hp, FLAGS_dyn_hp, FLAGS_async);
        EmotionsDetection emotionsDetector(FLAGS_m_em, FLAGS_d_em, FLAGS_n_em, FLAGS_dyn_em, FLAGS_async);
        FacialLandmarksDetection facialLandmarksDetector(FLAGS_m_lm, FLAGS_d_lm, FLAGS_n_lm, FLAGS_dyn_lm, FLAGS_async);

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
        Load(ageGenderDetector).into(pluginsForDevices[FLAGS_d_ag], FLAGS_dyn_ag);
        Load(headPoseDetector).into(pluginsForDevices[FLAGS_d_hp], FLAGS_dyn_hp);
        Load(emotionsDetector).into(pluginsForDevices[FLAGS_d_em], FLAGS_dyn_em);
        Load(facialLandmarksDetector).into(pluginsForDevices[FLAGS_d_lm], FLAGS_dyn_lm);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        // Start inference & calc performance.
        slog::info << "Start inference " << slog::endl;
        if (!FLAGS_no_show) {
            std::cout << "Press any key to stop" << std::endl;
        }

        bool isFaceAnalyticsEnabled = ageGenderDetector.enabled() || headPoseDetector.enabled() ||
                                      emotionsDetector.enabled() || facialLandmarksDetector.enabled();

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
        dlib::image_window win, win_faces;

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
                    ageGenderDetector.enqueue(face);
                    headPoseDetector.enqueue(face);
                    emotionsDetector.enqueue(face);
                    facialLandmarksDetector.enqueue(face);
                }
            }
            timer.finish("data preprocessing");

            // Run age-gender recognition, head pose estimation and emotions recognition simultaneously.
            timer.start("face analytics call");
            if (isFaceAnalyticsEnabled) {
                ageGenderDetector.submitRequest();
                headPoseDetector.submitRequest();
                emotionsDetector.submitRequest();
                facialLandmarksDetector.submitRequest();
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
                ageGenderDetector.wait();
                headPoseDetector.wait();
                emotionsDetector.wait();
                facialLandmarksDetector.wait();
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

                // dlib face detection
                /*
                dlib::array2d<dlib::rgb_pixel> img;
                dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(prev_frame));
                std::vector<dlib::rectangle> dets = detector(img);
                std::vector<dlib::full_object_detection> shapes;
                for (unsigned long j = 0; j < dets.size(); ++j)
                {
                    dlib::full_object_detection shape = sp(img, dets[j]);
                    // You get the idea, you can get all the face part locations if
                    // you want them.  Here we just store them in shapes so we can
                    // put them on the screen.
                    shapes.push_back(shape);
                }
                win.clear_overlay();
                win.set_image(img);
                win.add_overlay(render_face_detections(shapes));
                win.add_overlay(shapes);*/

                // For every detected face.
                int i = 0;
                std::vector<cv::Point2f> left_eye, right_eye;
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
                            cv::circle(prev_frame, cv::Point2l(shape.part(i).x(), shape.part(i).y()), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);
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
                        }else {
                            if(blink_counter >= EYE_AR_CONSEC_FRAMES)
                                blinl_total += 1;
                            blink_counter = 0; 
                        }
                        std::string blink_str = "Blinks: " + std::to_string(blinl_total);
                        std::string ear_str = "EAR: " + std::to_string(ear_avg);
                        cv::putText(frame, blink_str, cv::Point2f(10, 70),cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
                        cv::putText(frame, ear_str, cv::Point2f(300, 70), cv::FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);  
                    }

                    if (ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch) {
                        out << (ageGenderDetector[i].maleProb > 0.5 ? "M" : "F");
                        out << std::fixed << std::setprecision(0) << "," << ageGenderDetector[i].age;
                        if (FLAGS_r) {
                            std::cout << "Predicted gender, age = " << out.str() << std::endl;
                        }
                    } else {
                        out << (result.label < faceDetector.labels.size() ? faceDetector.labels[result.label] :
                                std::string("label #") + std::to_string(result.label))
                            << ": " << std::fixed << std::setprecision(3) << result.confidence;
                    }

                    if (emotionsDetector.enabled() && i < emotionsDetector.maxBatch) {
                        std::string emotion = emotionsDetector[i];
                        if (FLAGS_r) {
                            std::cout << "Predicted emotion = " << emotion << std::endl;
                        }
                        out << "," << emotion;
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
                    }
                    cv::Point2f eye_l_0;
                    cv::Point2f eye_l_1;
                    if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
                        auto normed_landmarks = facialLandmarksDetector[i];
                        auto n_lm = normed_landmarks.size();
                        if (FLAGS_r)
                            std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
                        for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                            if(i_lm < 4 || (i_lm >=12 && i_lm<=17)){
                                float normed_x = normed_landmarks[2 * i_lm];
                                float normed_y = normed_landmarks[2 * i_lm + 1];
                                if (FLAGS_r) {
                                    std::cout << normed_x << ", "
                                            << normed_y << std::endl;
                                }
                                int x_lm = rect.x + rect.width * normed_x;
                                int y_lm = rect.y + rect.height * normed_y;
                                if(i_lm == 1)
                                    eye_l_0=cv::Point2f(x_lm, y_lm);
                                if(i_lm == 0)
                                    eye_l_1=cv::Point2f(x_lm, y_lm);
                                // Draw facial landmarks on the frame
                                cv::circle(prev_frame, cv::Point(x_lm, y_lm), 1 + static_cast<int>(0.0012 * rect.width), cv::Scalar(0, 255, 255), -1);    
                            }
                        }/*
                        //Eye detection
                        float distance_l = sqrt((eye_l_1.x - eye_l_0.x)*(eye_l_1.x - eye_l_0.x) + (eye_l_1.y - eye_l_0.y)*(eye_l_1.y - eye_l_0.y));
                        float safety_factor = 0.20 * distance_l;
                        cv::Point2f frame_p0(eye_l_0.x - safety_factor, (eye_l_0.y + distance_l/3) + safety_factor);
                        cv::Point2f frame_p1(eye_l_1.x + safety_factor, (eye_l_1.y - distance_l/3) - safety_factor);
                        cv::Rect frame_eye_l(frame_p0,frame_p1);

                        cv::Mat l_eye;
                        prev_frame(frame_eye_l).copyTo(l_eye);
                        cv::cvtColor(l_eye, l_eye, cv::COLOR_BGR2GRAY);
                        cv::namedWindow("Left Eye gray");
                        cv::imshow("Left Eye gray", l_eye);
                        cv::adaptiveThreshold(l_eye, l_eye, 130, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, 2);
                        //cv::threshold(l_eye, l_eye, 70, 255, cv::THRESH_BINARY);
                        cv::namedWindow("Left Eye");
                        cv::imshow("Left Eye", l_eye);
                        int count = cv::countNonZero(l_eye);
                        std::cout << count << std::endl;
                        cv::rectangle(prev_frame, frame_eye_l, cv::Scalar(147, 20, 255), 2);
                        */
                    }

                    auto genderColor = (ageGenderDetector.enabled() && (i < ageGenderDetector.maxBatch)) ?
                                ((ageGenderDetector[i].maleProb < 0.5) ? cv::Scalar(147, 20, 255) : cv::Scalar(255, 0,
                                                                                                               0))
                              : cv::Scalar(100, 100, 100);
                    cv::rectangle(prev_frame, result.location, genderColor, 1);

                    i++;
                }

                cv::imshow("Detection results", prev_frame);
                timer.finish("visualization");
            } else if (FLAGS_r) {
                // For every detected face.
                for (int i = 0; i < prev_detection_results.size(); i++) {
                    if (ageGenderDetector.enabled() && i < ageGenderDetector.maxBatch) {
                        out.str("");
                        out << (ageGenderDetector[i].maleProb > 0.5 ? "M" : "F");
                        out << std::fixed << std::setprecision(0) << "," << ageGenderDetector[i].age;
                        std::cout << "Predicted gender, age = " << out.str() << std::endl;
                    }

                    if (emotionsDetector.enabled() && i < emotionsDetector.maxBatch) {
                        std::cout << "Predicted emotion = " << emotionsDetector[i] << std::endl;
                    }

                    if (headPoseDetector.enabled() && i < headPoseDetector.maxBatch) {
                        std::cout << "Head pose results: yaw, pitch, roll = "
                                  << headPoseDetector[i].angle_y << ";"
                                  << headPoseDetector[i].angle_p << ";"
                                  << headPoseDetector[i].angle_r << std::endl;
                    }

                    if (facialLandmarksDetector.enabled() && i < facialLandmarksDetector.maxBatch) {
                        auto normed_landmarks = facialLandmarksDetector[i];
                        auto n_lm = normed_landmarks.size();
                        std::cout << "Normed Facial Landmarks coordinates (x, y):" << std::endl;
                        for (auto i_lm = 0UL; i_lm < n_lm / 2; ++i_lm) {
                            float normed_x = normed_landmarks[2 * i_lm];
                            float normed_y = normed_landmarks[2 * i_lm + 1];
                            std::cout << normed_x << ", "
                                      << normed_y << std::endl;
                        }
                    }
                }
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
            ageGenderDetector.printPerformanceCounts();
            headPoseDetector.printPerformanceCounts();
            emotionsDetector.printPerformanceCounts();
            facialLandmarksDetector.printPerformanceCounts();
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


