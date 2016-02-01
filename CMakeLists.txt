# CMakeLists.txt has to be located in the project folder and cmake has to be
# executed from 'project/build' with 'cmake ../'.
cmake_minimum_required(VERSION 2.6)
find_package(Rock)
rock_init(object_recognition 0.1)
rock_find_cmake(Boost 1.54 REQUIRED COMPONENTS system)
rock_standard_layout()
