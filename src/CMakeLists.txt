rock_library(object_recognition
    SOURCES Dummy.cpp
    HEADERS Dummy.hpp)

rock_executable(object_recognition_bin Main.cpp
    DEPS object_recognition)

