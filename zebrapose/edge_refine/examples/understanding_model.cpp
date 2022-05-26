#include "rbot_evaluator.h"

using namespace srt3d;
using namespace std;
using namespace cv;

int main(int argc, char** argv){
  std::filesystem::path dataset_directory{"/home/lyl/git/cosypose/local_data/linemod/original_dataset"};
  std::string body_name{"cat"};
  std::filesystem::path ply_path{dataset_directory / body_name / "mesh.ply"};
  
  // 生成camera2body_pose
  Eigen::Vector3f t{0.0662f,-0.1163f,1.0878f};
  Eigen::Matrix3f R;
  R << 0.0951, -0.9833, 0.1551, 0.7416, 0.1739, 0.6479, -0.6641, 0.0534, 0.7458;
  Transform3fA body2camera_pose;
  body2camera_pose = Eigen::Translation<float, 3>{t};
  body2camera_pose.rotate(R);
  Transform3fA camera2body_pose = body2camera_pose.inverse();


  // 读取obj并设置body2world_pose
  Transform3fA geometry2body_pose{Transform3fA::Identity()};
  auto body_ptr{std::make_shared<srt3d::Body>(
      body_name, ply_path, 0.001f, true, false, 0.3f,
      geometry2body_pose)};
  Transform3fA body2world_pose{Transform3fA::Identity()};
  body_ptr->set_body2world_pose(body2world_pose);
  std::cout << "geometry2body_pose:" << std::endl << geometry2body_pose.matrix() << std::endl;
  std::cout << "body2world_pose:" << std::endl << body2world_pose.matrix() << std::endl;
  std::cout << "camera2body_pose:" << std::endl << camera2body_pose.matrix() << std::endl;
  // 设置renderer_geometry
  auto renderer_geometry_ptr = std::make_shared<RendererGeometry>("rg");
  renderer_geometry_ptr->SetUp(); // Set up GLFW
  renderer_geometry_ptr->AddBody(body_ptr);
  
  // 设置renderer
  Intrinsics intrinsics{572.41140, 573.57043, 325.26110,
                        242.04899, 640, 480};
  float z_min = 0.001f;
  float z_max = 50.0f;
  auto renderer_ptr= std::make_shared<NormalRenderer>(
      "renderer", renderer_geometry_ptr, Transform3fA::Identity(), intrinsics,
      z_min, z_max);
  renderer_ptr->SetUp();  //检查参数，设置投影矩阵和创建buffer

  // 渲染图片
  renderer_ptr->set_camera2world_pose(camera2body_pose);
  renderer_ptr->StartRendering();
  renderer_ptr->FetchNormalImage();
  renderer_ptr->FetchDepthImage();

  // //生成数据
  // Model::TemplateView template_view;
  // template_view.orientation = camera2body_pose.matrix().col(2).segment(0, 3);
  // template_view.data_points.resize(200);

  // Compute silhouette
  std::vector<cv::Mat> normal_image_channels(4);
  cv::split(renderer_ptr->normal_image(), normal_image_channels);
  cv::Mat &silhouette_image{normal_image_channels[3]};

  // 图片可视化
  cout << "renderer_ptr->normal_image():" << silhouette_image.size() << endl;
  cv::namedWindow("renderer_ptr->normal_image()", cv::WINDOW_FREERATIO);
  cv::imshow("renderer_ptr->normal_image()", renderer_ptr->normal_image());
  cv::waitKey(0);

  // Generate contour
  std::vector<std::vector<cv::Point2i>> contours;
  cv::findContours(silhouette_image, contours, cv::RetrievalModes::RETR_LIST,
                   cv::ContourApproximationModes::CHAIN_APPROX_NONE);

  // 除去小的轮廓
  contours.erase(std::remove_if(begin(contours), end(contours),
                                [](const std::vector<cv::Point2i> &contour){return contour.size() < 20;}), end(contours));

  // // Contour 可视化
  // cv::Mat contour_image=Mat::zeros(silhouette_image.size(),CV_8UC1);
  // cv::drawContours(contour_image, contours, -1, cv::Scalar(255));
  // cv::namedWindow("contour", cv::WINDOW_FREERATIO);
  // cv::imshow("contour", contour_image);
  // cv::waitKey(0);

  // imformation
  if (contours.size()!=1){
    std::cerr << "共有轮廓:" << contours.size() << std::endl;
  }

  //随机生成器
  std::mt19937 generator{7};

  // Calculate data for contour points
  for (int i = 0; i < int(contours[0].size()); i++)
  {
    cv::Point2i center{contours[0][i]};
    Eigen::Vector3f center_f_camera{renderer_ptr->GetPointVector(center)}; // 根据这个点的深度信息得到相机系下的空间坐标
    Eigen::Vector3f center_f_body = camera2body_pose * center_f_camera; //这个点在物体坐标系下的坐标
    // 计算这个center的 contour_segment和近似的法向量
    std::vector<cv::Point2i> contour_segment;
    int start_idx = i - 2;
    int end_idx = i + 2;
    if (start_idx < 0){
      contour_segment.insert(end(contour_segment), end(contours[0]) + start_idx, end(contours[0]));
      start_idx = 0;
    }
    else if (end_idx >= int(contours[0].size())){
      contour_segment.insert(end(contour_segment), begin(contours[0]) + start_idx, end(contours[0]));
      start_idx = 0;
      end_idx = end_idx - int(contours[0].size());
    }
    contour_segment.insert(end(contour_segment), begin(contours[0]) + start_idx, begin(contours[0]) + end_idx + 1);
    
    //计算这个点的法向量
    Eigen::Vector2f normal{-float(contour_segment.back().y - contour_segment.front().y),
                    float(contour_segment.back().x - contour_segment.front().x)};
    normal = normal.normalized();
    Eigen::Vector3f normal_f_camera{normal.x(), normal.y(), 0.0f};
    Eigen::Vector3f normal_f_body = camera2body_pose.rotation() * normal_f_camera;

  }

  return 0;
}
