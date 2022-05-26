#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <string>

std::vector<double> modify(const std::vector<double>& input)
{
   std::string body_name{"cat"};
   std::cout << body_name << std::endl;




   std::vector<double> output(input.size());

   for ( size_t i = 0 ; i < input.size() ; ++i )
     output[i] = 2. * input[i];

  return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(example,m)
{
  m.doc() = "pybind11 example plugin";

  m.def("modify", &modify, "Multiply all entries of a list by 2.0");
}
