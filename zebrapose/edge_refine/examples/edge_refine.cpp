#include "rbot_evaluator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/eigen.h>

#include <time.h>

using namespace srt3d;
using namespace cv;

std::vector<double> modify(const std::vector<double>& input)
{
   std::vector<double> output(input.size());

   for ( size_t i = 0 ; i < input.size() ; ++i )
     output[i] = 2. * input[i];

  return output;
}

void edge_refine(Eigen::Matrix3f &R, Eigen::Vector3f &t, const std::vector<float> visible_contours, std::string ply_path_, std::string save_path_prefix){
  clock_t start = clock(); //TODO 耗时测试
  // ply的路径
  std::filesystem::path ply_path{ply_path_};
  std::string body_name{"not_important"};
  // 生成相机到世界pose，世界到物体坐标系pose
  Transform3fA body2camera_pose;
  body2camera_pose = Eigen::Translation<float, 3>{t};
  body2camera_pose.rotate(R);
  Transform3fA camera2body_pose = body2camera_pose.inverse();
  Transform3fA geometry2body_pose{Transform3fA::Identity()};
  Transform3fA body2world_pose{Transform3fA::Identity()};
  // 读取ply并设置各种pose
  auto body_ptr{std::make_shared<srt3d::Body>(
      body_name, ply_path, 0.001f, true, false, 0.3f,
      geometry2body_pose)};
  body_ptr->set_body2world_pose(body2world_pose);
  // 设置renderer_geometry
  auto renderer_geometry_ptr = std::make_shared<RendererGeometry>("rg");
  renderer_geometry_ptr->SetUp(); // Set up GLFW
  renderer_geometry_ptr->AddBody(body_ptr);
  // 设置renderer
// for lmo
//  Intrinsics intrinsics{572.41140, 573.57043, 325.26110,
//                        242.04899, 640, 480};
// for ycbv
  Intrinsics intrinsics{1066.778, 1067.487, 312.9869, 241.3109, 640, 480};
  float z_min = 0.001f;
  float z_max = 50.0f;
  auto renderer_ptr= std::make_shared<NormalRenderer>(
      "renderer", renderer_geometry_ptr, Transform3fA::Identity(), intrinsics,
      z_min, z_max);
  renderer_ptr->SetUp();  //检查参数，设置投影矩阵和创建buffer
  // 开始优化
  float tikhonov_parameter_rotation_ = 5000.0f;
  float tikhonov_parameter_translation_ = 500000.0f;
  Eigen::Matrix<float, 6, 6> tikhonov_matrix_ = Eigen::Matrix<float, 6, 6>::Zero();
  tikhonov_matrix_.diagonal().head<3>().array() = tikhonov_parameter_rotation_;
  tikhonov_matrix_.diagonal().tail<3>().array() =
      tikhonov_parameter_translation_;
  clock_t before_iter=clock();
  float last_cost = 1e8;
  for (int iter = 0; iter < 10; iter++)
  {
    // 渲染图片
    renderer_ptr->set_camera2world_pose(camera2body_pose);
    renderer_ptr->StartRendering();
    renderer_ptr->FetchNormalImage();
    renderer_ptr->FetchDepthImage();
    // Compute silhouette
    std::vector<cv::Mat> normal_image_channels(4);
    cv::split(renderer_ptr->normal_image(), normal_image_channels);
    cv::Mat &silhouette_image{normal_image_channels[3]};
//    cv::imwrite(save_path_prefix + "/silhouette_image_" + std::to_string(iter) + ".jpg", silhouette_image);
    // Generate contour
    std::vector<std::vector<cv::Point2i>> contours;
    cv::findContours(silhouette_image, contours, cv::RetrievalModes::RETR_EXTERNAL,
                    cv::ContourApproximationModes::CHAIN_APPROX_NONE);
    // 除去小的轮廓
    contours.erase(std::remove_if(begin(contours), end(contours),
                                  [](const std::vector<cv::Point2i> &contour){return contour.size() < 20;}), end(contours));

    // Contour 可视化
    cv::Mat contour_image=Mat::zeros(silhouette_image.size(),CV_8UC3);
    // red color represents the estimated pose contour, green color represents the visible contour
    cv::drawContours(contour_image, contours, -1, cv::Scalar(0, 0, 255));
    for (size_t i = 0; i < visible_contours.size() / 2; i++){
      contour_image.at<cv::Vec3b>(visible_contours[2 * i + 1], visible_contours[2 * i + 0])[1] += 255;
    }
//    cv::imwrite(save_path_prefix + "/R_t_contour_image_" + std::to_string(iter) + ".jpg", contour_image);
    if (contours.size()==0){
      std::cout << "find contours number (should be at least 1):" << contours.size() << std::endl;
      cv::imwrite(save_path_prefix + "/R_t_contour_image_" + std::to_string(iter) + "_cant_find_contours.jpg", silhouette_image);
      return;
    }
    float cost = 0;
    Eigen::Matrix<float, 6, 6> H = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
    for (size_t i = 0; i < visible_contours.size() / 2; i++)
    {
      // get the point in visible_contours
      int x = visible_contours[2 * i + 0];
      int y = visible_contours[2 * i + 1];
      // find the closest point in contours
      int closest_y = -1;
      int closest_x = -1;
      int closest_distance = 1e7;
      for (size_t j = 0; j < contours[0].size(); j++)
      {
        int tmp_y = contours[0][j].y;
        int tmp_x = contours[0][j].x;
        int tmp_distance = (tmp_y - y) * (tmp_y - y) + (tmp_x - x) * (tmp_x - x);
        if (tmp_distance < closest_distance){
          closest_distance = tmp_distance;
          closest_y = tmp_y;
          closest_x = tmp_x;
        }
      }
      if (i % 5 == 0){
        cv::line(contour_image, Point(x, y), Point(closest_x, closest_y), cv::Scalar(255, 255, 255));
      }
      // 计算目标函数 
      Eigen::Vector3f center_f_camera{renderer_ptr->GetPointVector(Point(closest_x, closest_y))}; // 根据这个点的深度信息得到相机系下的空间坐标
      Eigen::Vector3f center_f_body = camera2body_pose * center_f_camera; //这个点在物体坐标系下的坐标
      float fx = intrinsics.fu;
      float fy = intrinsics.fv;
      float cx = intrinsics.ppu;
      float cy = intrinsics.ppv;
      float inv_z = 1.0 / center_f_camera[2];
      float inv_z2 = inv_z * inv_z;
      Eigen::Vector2f proj(fx * center_f_camera[0] * inv_z + cx, fy * center_f_camera[1] * inv_z + cy);
      Eigen::Vector2f e(x - proj[0], y - proj[1]);
      cost += e.squaredNorm();

      Eigen::Matrix<float, 2, 3> de_dcenter_f_camera;
      de_dcenter_f_camera << -fx * inv_z, 0, fx * center_f_camera[0] * inv_z2,
          0, -fy * inv_z, fy * center_f_camera[1] * inv_z2;
      Eigen::Matrix<float, 2, 3> de_dtranslation{de_dcenter_f_camera * body2camera_pose.rotation()};
      Eigen::Matrix3f center_f_body_hat;
      center_f_body_hat <<  0, -center_f_body(2), center_f_body(1),
                            center_f_body(2), 0, -center_f_body(0),
                            -center_f_body(1), center_f_body(0), 0;
      Eigen::Matrix<float, 2, 3> de_drotation{-de_dtranslation * center_f_body_hat};
      Eigen::Matrix<float, 2, 6> de_dtheta;  // 左边旋转，右边平移
      de_dtheta.leftCols(3) = de_drotation;
      de_dtheta.rightCols(3) = de_dtranslation;
      // J = de_dtheta, H = J^T*J, b = -J^T*e
      H += de_dtheta.transpose() * de_dtheta;
      b -= de_dtheta.transpose() * e;
    }
    if(cost < last_cost){
      last_cost = cost;
    }
    // Optimize and update pose
    Eigen::FullPivLU<Eigen::Matrix<float, 6, 6>> lu{tikhonov_matrix_ + H};
    if (lu.isInvertible()) {
      Eigen::Matrix<float, 6, 1> theta{lu.solve(b)};
      Transform3fA pose_variation{Transform3fA::Identity()};
      pose_variation.rotate(Vector2Skewsymmetric(theta.head<3>()).exp());
      pose_variation.translate(theta.tail<3>());
      body2camera_pose = body2camera_pose * pose_variation;
      camera2body_pose = body2camera_pose.inverse();
      renderer_ptr->set_camera2world_pose(camera2body_pose);
    }

//    cv::imwrite(save_path_prefix + "/R_t_contour_match_" + std::to_string(iter) + ".jpg", contour_image);
//    std::cout << "cost for iter" << iter << ":" << cost << std::endl;
  }
  R = body2camera_pose.rotation();
  t = body2camera_pose.translation();
  clock_t after_iter=clock();
//  std::cout<<std::endl<<"cost time before iter:"<<double(before_iter-start)/ CLOCKS_PER_SEC<<std::endl;
//  std::cout<<"cost time in iter:"<<double(after_iter-before_iter)/ CLOCKS_PER_SEC<<std::endl;
  return ;
}

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::tuple py_edge_refine(Eigen::Matrix3f R, Eigen::Vector3f t,
                    py::array_t<float, py::array::c_style | py::array::forcecast> contours_, std::string ply_path, std::string path_for_save){
  // check input dimensions
  if ( contours_.ndim()     != 2 )
    throw std::runtime_error("Input should be 2-D NumPy array");
  if ( contours_.shape()[1] != 2 )
    throw std::runtime_error("Input should have size [N,2]");
  // allocate and copy data to std::vector (to pass to the C++ function)
  std::vector<float> contours(contours_.size());
  std::memcpy(contours.data(),contours_.data(),contours_.size()*sizeof(float));

  // call pure C++ function
  edge_refine(R, t, contours, ply_path, path_for_save);
  py::tuple outputs = py::make_tuple(R, t);
  return outputs;
}

PYBIND11_MODULE(edge_refine,m)
{
  m.doc() = "pybind11 to get contour of R, t";

  m.def("py_edge_refine", &py_edge_refine);
  m.def("modify", &modify, "Multiply all entries of a list by 2.0");
}
