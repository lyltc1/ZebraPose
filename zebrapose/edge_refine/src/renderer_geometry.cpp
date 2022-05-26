// SPDX-License-Identifier: MIT
// Copyright (c) 2021 Manuel Stoiber, German Aerospace Center (DLR)

#include <srt3d/renderer_geometry.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

namespace srt3d {

int RendererGeometry::n_instances_ = 0;

RendererGeometry::RendererGeometry(const std::string &name) : name_{name} {}

RendererGeometry::~RendererGeometry() {
  if (initial_set_up_) {
    glfwMakeContextCurrent(window_);
    for (auto &render_data_body : render_data_bodies_) {
      DeleteGLVertexObjects(&render_data_body);
    }
    glfwMakeContextCurrent(0);
    glfwDestroyWindow(window_);
    window_ = nullptr;
    n_instances_--;
    if (n_instances_ == 0) glfwTerminate();
  }
}

bool RendererGeometry::SetUp() {
  // Set up GLFW
  if (!initial_set_up_) {
    if (!glfwInit()) {
      std::cerr << "Failed to initialize GLFW" << std::endl;
      return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    window_ = glfwCreateWindow(640, 480, "window", nullptr, nullptr);
    if (window_ == nullptr) {
      std::cerr << "Failed to create GLFW window" << std::endl;
      glfwTerminate();
      return false;
    }

    glfwMakeContextCurrent(window_);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
      std::cerr << "Failed to initialize GLEW" << std::endl;
      glfwDestroyWindow(window_);
      window_ = nullptr;
      glfwTerminate();
      return false;
    }
    glfwMakeContextCurrent(nullptr);

    n_instances_++;
    initial_set_up_ = true;
  }

  // Start with normal setup
  set_up_ = false;

  // Set up bodies
  glfwMakeContextCurrent(window_);
  for (auto &render_data_body : render_data_bodies_) {
    // Load vertices of mesh
    std::vector<float> vertices;
    if (!LoadPLYIntoVertices(*render_data_body.body_ptr.get(), &vertices))
      return false;
    render_data_body.n_vertices = unsigned(vertices.size()) / 6;

    // Create GL Vertex objects
    if (set_up_) DeleteGLVertexObjects(&render_data_body);
    CreateGLVertexObjects(vertices, &render_data_body);
  }
  glfwMakeContextCurrent(nullptr);

  set_up_ = true;
  return true;
}

bool RendererGeometry::AddBody(std::shared_ptr<Body> body_ptr, bool verbose) {
  const std::lock_guard<std::mutex> lock{mutex_}; //直到对象作用域结束的自动解锁

  // Check if renderer geometry for body already exists
  for (auto &render_data_body : render_data_bodies_) {
    if (body_ptr->name() == render_data_body.body_ptr->name()) {
      if (verbose)
        std::cerr << "Body data " << body_ptr->name() << " already exists"
                  << std::endl;
      return false;
    }
  }

  // Create data for body and assign parameters
  RenderDataBody render_data_body;
  render_data_body.body_ptr = std::move(body_ptr);
  if (set_up_) {
    // Load vertices of mesh
    std::vector<float> vertices;
    if (!LoadPLYIntoVertices(*render_data_body.body_ptr.get(), &vertices))
      return false;
    render_data_body.n_vertices = unsigned(vertices.size()) / 6;

    // Create GL Vertex objects
    glfwMakeContextCurrent(window_);
    CreateGLVertexObjects(vertices, &render_data_body);
    glfwMakeContextCurrent(nullptr);
  }

  // Add body data
  render_data_bodies_.push_back(std::move(render_data_body));
  return true;
}

bool RendererGeometry::DeleteBody(const std::string &name, bool verbose) {
  const std::lock_guard<std::mutex> lock{mutex_};
  for (size_t i = 0; i < render_data_bodies_.size(); ++i) {
    if (name == render_data_bodies_[i].body_ptr->name()) {
      if (set_up_) {
        glfwMakeContextCurrent(window_);
        DeleteGLVertexObjects(&render_data_bodies_[i]);
        glfwMakeContextCurrent(nullptr);
      }
      render_data_bodies_.erase(begin(render_data_bodies_) + i);
      return true;
    }
  }
  if (verbose)
    std::cerr << "Body data \"" << name << "\" not found" << std::endl;
  return false;
}

void RendererGeometry::ClearBodies() {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (set_up_) {
    glfwMakeContextCurrent(window_);
    for (auto &render_data_body : render_data_bodies_) {
      DeleteGLVertexObjects(&render_data_body);
    }
    glfwMakeContextCurrent(nullptr);
  }
  render_data_bodies_.clear();
}

void RendererGeometry::MakeContextCurrent() {
  mutex_.lock();
  glfwMakeContextCurrent(window_);
}

void RendererGeometry::DetachContext() {
  glfwMakeContextCurrent(nullptr);
  mutex_.unlock();
}

const std::string &RendererGeometry::name() const { return name_; }

const std::vector<RendererGeometry::RenderDataBody>
    &RendererGeometry::render_data_bodies() const {
  return render_data_bodies_;
}

bool RendererGeometry::set_up() const { return set_up_; }

bool RendererGeometry::LoadMeshIntoVertices(const Body &body,
                                            std::vector<float> *vertices) {
  tinyobj::attrib_t attributes;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warning;
  std::string error;

  if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error,
                        body.geometry_path().string().c_str(), nullptr, true,
                        false)) {
    std::cerr << "TinyObjLoader failed to load data from "
              << body.geometry_path() << std::endl;
    return false;
  }
  if (!error.empty()) std::cerr << error << std::endl;

  vertices->clear();
  for (auto &shape : shapes) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
      if (shape.mesh.num_face_vertices[f] != 3) {
        std::cerr << "Mesh contains non triangle shapes" << std::endl;
        index_offset += shape.mesh.num_face_vertices[f];
        continue;
      }

      // Extract triangle points
      std::array<Eigen::Vector3f, 3> points;
      for (int v = 0; v < 3; ++v) {
        int idx = 3 * shape.mesh.indices[index_offset + v].vertex_index;
        if (body.geometry_counterclockwise()) {
          points[v](0) = float(attributes.vertices[idx + 0]);
          points[v](1) = float(attributes.vertices[idx + 1]);
          points[v](2) = float(attributes.vertices[idx + 2]);
          points[v] *= body.geometry_unit_in_meter();
        } else {
          points[2 - v](0) = float(attributes.vertices[idx + 0]);
          points[2 - v](1) = float(attributes.vertices[idx + 1]);
          points[2 - v](2) = float(attributes.vertices[idx + 2]);
          points[2 - v] *= body.geometry_unit_in_meter();
        }
      }

      // Calculate normal vector
      Eigen::Vector3f normal{
          (points[2] - points[1]).cross(points[0] - points[1]).normalized()};

      // Save data in vertices vector
      for (auto point : points) {
        vertices->insert(end(*vertices), point.data(), point.data() + 3);
        vertices->insert(end(*vertices), normal.data(), normal.data() + 3);
      }

      index_offset += 3;
    }
  }
  return true;
}
/* 将每个面对应的三个点以及法向存入vertices */
bool RendererGeometry::LoadPLYIntoVertices(const Body &body,
                                            std::vector<float> *vertices) {
  vertices->clear();
  /* 读取ply文件 */
  std::ifstream ifs(body.geometry_path().string().c_str());
  if (!ifs.is_open()) {
    std::cerr << "Cannot open file [" << body.geometry_path() << "]" << std::endl;
    return -1;
  }
  std::string linebuf;
  // 载入点、面信息
  int vertex_count = -1;
  int face_count = -1;
  while (getline(ifs, linebuf))
  {
    if (linebuf.find("element vertex")!=std::string::npos){
      vertex_count = stoi(linebuf.substr(15));  //从第15个字符开始为数字
    }
    else if (linebuf.find("element face")!=std::string::npos){
      face_count = stoi(linebuf.substr(13));  //从第13个字符开始为数字
    }
    else if (linebuf.find("end_header")!=std::string::npos){
      break;
    }
  }
  //载入点及点的法向
  std::vector<float> vertices_;
  std::vector<float> verNormals_;
  vertices_.clear();
  verNormals_.clear();
  int pos_ = 0;
  for (int i = 0; i < vertex_count; i++)
  {
    getline(ifs, linebuf);
    for (int j = 0; j < 3; j++){
      pos_ = linebuf.find(" ");
      vertices_.push_back(stof(linebuf.substr(0, pos_))*body.geometry_unit_in_meter());
      linebuf.erase(0, pos_ + 1);
    }
    for (int j = 0; j < 3; j++){
      pos_ = linebuf.find(" ");
      verNormals_.push_back(stof(linebuf.substr(0, pos_)));
      linebuf.erase(0, pos_ + 1);
      }
    }
  // 按照每个面的顺序存入点及对应的法向
  int numPointPerFace, verIndex_1, verIndex_2, verIndex_3;
  for (int i = 0; i < face_count; i++){
    getline(ifs, linebuf);
    pos_ = linebuf.find(" ");
    numPointPerFace = stoi(linebuf.substr(0, pos_));
    linebuf.erase(0, pos_ + 1);
    if (numPointPerFace != 3){
      std::cerr << "The point numbers of the face is not 3" << std::endl;
      continue;
    }
    pos_ = linebuf.find(" ");
    verIndex_1 = stoi(linebuf.substr(0, pos_));
    linebuf.erase(0, pos_ + 1);
    pos_ = linebuf.find(" ");
    verIndex_2 = stoi(linebuf.substr(0, pos_));
    linebuf.erase(0, pos_ + 1);
    pos_ = linebuf.find(" ");
    verIndex_3 = stoi(linebuf.substr(0, pos_));
    // 面上的第一个点
    for (int j = 0; j < 3; j++) {
      vertices->push_back(vertices_[verIndex_1 * 3 + j]);
    }
    // 面上第一个法向
    for (int j = 0; j < 3; j++) {
      vertices->push_back(verNormals_[verIndex_1 * 3 + j]);
    }
    // 面上的第二个点
    for (int j = 0; j < 3; j++) {
      vertices->push_back(vertices_[verIndex_2 * 3 + j]);
    }
    // 面上第二个法向
    for (int j = 0; j < 3; j++) {
      vertices->push_back(verNormals_[verIndex_2 * 3 + j]);
    }
    // 面上的第三个点
    for (int j = 0; j < 3; j++) {
      vertices->push_back(vertices_[verIndex_3 * 3 + j]);
    }
    // 面上第三个法向
    for (int j = 0; j < 3; j++) {
      vertices->push_back(verNormals_[verIndex_3 * 3 + j]);
    }
  }
  return true;
}
void RendererGeometry::CreateGLVertexObjects(const std::vector<float> &vertices,
                                             RenderDataBody *render_data_body) {
  glGenVertexArrays(1, &render_data_body->vao);  // 创建VAO
  glBindVertexArray(render_data_body->vao);  // 设置为当前操作的VAO

  glGenBuffers(1, &render_data_body->vbo);  // 创建VBO
  glBindBuffer(GL_ARRAY_BUFFER, render_data_body->vbo);  // 设置VBO
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float),
               &vertices.front(), GL_STATIC_DRAW); //设置VBO中的数据

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr); // 设置索引为0顶点属性
  glEnableVertexAttribArray(0); // 设置开启顶点属性
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                        (void *)(3 * sizeof(float))); //设置索引为1顶点属性
  glEnableVertexAttribArray(1); //设置开启顶点属性

  glBindBuffer(GL_ARRAY_BUFFER, 0); // 解绑VBO
  glBindVertexArray(0); // 解绑VAO
}

void RendererGeometry::DeleteGLVertexObjects(RenderDataBody *render_data_body) {
  glDeleteBuffers(1, &render_data_body->vbo);
  glDeleteVertexArrays(1, &render_data_body->vao);
}

}  // namespace srt3d
