// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "svo/config.h"
#include "svo/frame_handler_mono.h"
#include "svo/map.h"
#include "svo/frame.h"
#include "svo/feature.h"
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <iostream>
#include "test_utils.h"

#include "sense/videocapture_dispatch.h"
#include "sense/tracking_events.h"

namespace svo {

using Matrix34f = Eigen::Matrix < float, 3, 4 > ;
using Vec4f = Eigen::Matrix < float, 4, 1 >;
using Vec3f = Eigen::Matrix < float, 3, 1 >;

// w, h, fx, fy, cx, cy
static std::array<float, 6> cam_params{640, 480, 500, 500, 320, 240};

// draw features
void draw_points(cv::Mat& bgr_img, std::vector<cv::Point2f> pts,
  cv::Scalar bgr, float offset = 3) {
  for (auto pt : pts) {
    cv::line(bgr_img, cv::Point(pt.x-offset, pt.y), cv::Point(pt.x+offset, pt.y),
      bgr);
    cv::line(bgr_img, cv::Point(pt.x, pt.y-offset), cv::Point(pt.x, pt.y+offset),
      bgr);
  }
}


// draw features
void draw_features(cv::Mat& bgr_img, std::list<svo::Feature*> features) {
  std::array<cv::Scalar, 5> colors{cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
    cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255)};
  for (auto f : features) {
    cv::circle(bgr_img, cv::Point2f(f->px[0], f->px[1]), 4*(f->level+1), colors[f->level], 1);
  }
}


// Draws desirable target in world coordinate to current color image
void draw_target(cv::Mat& rgb_img, const Matrix34f& mvp) {
  const Vec4f point_target(0, 0, 1, 1);
  Vec3f point_cam = mvp * point_target;
  point_cam /= point_cam[2];
  Vec3f pointx_cam = mvp * (point_target + Vec4f(0.1, 0, 0, 0));
  pointx_cam /= pointx_cam[2];
  Vec3f pointy_cam = mvp * (point_target + Vec4f(0, 0.1, 0, 0));
  pointy_cam /= pointy_cam[2];
  Vec3f pointz_cam = mvp * (point_target + Vec4f(0, 0, 0.1, 0));
  pointz_cam /= pointz_cam[2];
  cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]),
           cv::Point(pointx_cam[0], pointx_cam[1]), cv::Scalar(0, 0, 255), 3);
  cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]),
           cv::Point(pointy_cam[0], pointy_cam[1]), cv::Scalar(0, 255, 0), 3);
  cv::line(rgb_img, cv::Point(point_cam[0], point_cam[1]),
           cv::Point(pointz_cam[0], pointz_cam[1]), cv::Scalar(255, 0, 0), 3);
}

class BenchmarkNode
{
  vk::AbstractCamera* cam_;
  svo::FrameHandlerMono* vo_;

public:
  BenchmarkNode();
  ~BenchmarkNode();
  void runFromFolder();
};

BenchmarkNode::BenchmarkNode()
{
  cam_ = new vk::PinholeCamera(cam_params[0], cam_params[1], cam_params[2], 
    cam_params[3], cam_params[4], cam_params[5]);
  vo_ = new svo::FrameHandlerMono(cam_);
  vo_->start();
}

BenchmarkNode::~BenchmarkNode()
{
  delete vo_;
  delete cam_;
}

void BenchmarkNode::runFromFolder()
{

  sense::VideoCaptureDispatch videodispatch(0, sense::CameraFamily::GENERIC,
                                          std::string(SVO_ROOT) +
                                              "/data/ueye-640x480-60fps.ini");

  sense::SenseSubscriber video_rec;
  video_rec.Register(videodispatch);

  Eigen::Matrix3f projection;
  projection << cam_params[2], 0.0, cam_params[4],
                0.0, cam_params[3], cam_params[5],
                0.0, 0.0, 1.0;
  bool stop = false;
  bool tracking = false;
  int img_id = 0;
  while (!stop) {
    cv::Mat bgr;
    while (!video_rec.empty()) {
      auto fr = std::dynamic_pointer_cast<sense::ImageEvent>(video_rec.pop());
      if (fr) cv::cvtColor(fr->frame(), bgr, CV_RGB2BGR);
    }
    if (bgr.empty()) continue;

    // // load image
    // char tmp[200];
    // sprintf(tmp, "%s/data/two-boxes/sensor_recorder_%06d-color.png", SVO_ROOT, img_id);
    // sprintf(tmp, "E:/Data/structural_modeling/recording/boxes-1/sensor_recorder_%06d-color.png", img_id);
    // img_id++;
    // if(img_id == 2)
    //   printf("reading image (%s)\n", tmp);
    // cv::Mat color = cv::imread(tmp);
    // if (color.empty()) {
    //   img_id = 0;
    //   continue;
    // }
    cv::Mat img;
    cv::cvtColor(bgr, img, CV_BGR2GRAY);

    auto t = std::chrono::high_resolution_clock::now();
    if (tracking) // process frame
      vo_->addImage(img, 0.01*img_id);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1-t).count();
    double fps = 1.e+6 / dur;

    static int kf_count = 0;
    static int frame_count = 0;
    frame_count += 1;

    // display tracking quality
    if(vo_->lastFrame() != nullptr)
    {
      kf_count += vo_->lastFrame()->is_keyframe_ ? 1 : 0;
      std::cout << "Frame-Id: " << vo_->lastFrame()->id_ << " \t"
                  << "#Features: " << vo_->lastNumObservations() << " \t"
                  << "Proc. Time: " << vo_->lastProcessingTime()*1000 << "ms \n";

      // access the pose of the camera via vo_->lastFrame()->T_f_w_.

      Eigen::Matrix4d w2c_pose = vo_->lastFrame()->T_f_w_.matrix();
      Matrix34f proj = projection * w2c_pose.block<3, 4>(0, 0).cast<float>();

      if (vo_->stage() == FrameHandlerMono::Stage::STAGE_FIRST_FRAME ||
        vo_->stage() == FrameHandlerMono::Stage::STAGE_SECOND_FRAME) {
        draw_points(bgr, vo_->initFeatureTrackRefPx(), cv::Scalar(255, 0, 0));
        draw_points(bgr, vo_->initFeatureTrackCurPx(), cv::Scalar(0, 255, 0));
      } else {
        draw_features(bgr, vo_->lastFrame()->fts_);
      }

      cv::Scalar green(0, 255, 0);

      cv::putText(bgr, vo_->ToString(vo_->stage()), cv::Point(1, 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, green);
      cv::putText(bgr, vo_->ToString(vo_->trackingQuality()),
                  cv::Point(1, 25), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  green);
      char tmp[50];
      sprintf(tmp, "Keyframe: #%d in %d frames", kf_count, frame_count);
      cv::putText(bgr, tmp,
                  cv::Point(1, 45), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  green);

      sprintf(tmp, "FPS: %.3f", fps);
      cv::putText(bgr, tmp,
                  cv::Point(1,65), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  green);

      draw_target(bgr, proj);

    } else {
      cv::putText(bgr, "Not tracking", cv::Point(1, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0));
    }

    cv::imshow("SVO Monocular", bgr);
    auto const c = cv::waitKey(10);
    switch (c) {
    case 27:
    stop = true;
    break;
    case 't':
    tracking = !tracking;
    break;
    case 'r':
    vo_->reset();
    vo_->start();
    break;
    }
  }
}

} // namespace svo

int main(int argc, char** argv)
{
  {
    svo::BenchmarkNode benchmark;
    benchmark.runFromFolder();
  }
  printf("BenchmarkNode finished.\n");
  return 0;
}

