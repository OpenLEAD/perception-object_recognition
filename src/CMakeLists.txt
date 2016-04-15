find_package(VTK)
rock_library(object_recognition
    SOURCES ORPointCloud.cpp
    HEADERS ORPointCloud.hpp
    DEPS_PKGCONFIG pcl_registration-1.7 pcl_io-1.7 base-types pcl_visualization-1.7  pcl_recognition-1.7 pcl_features-1.7 pcl_keypoints-1.7 pcl_filters-1.7
    DEPS_CMAKE Boost VTK)

rock_executable(narf_identifier_bin narf_identifier.cpp
    DEPS object_recognition)

rock_executable(pcd_opener_bin pcd_opener.cpp
    DEPS object_recognition)

rock_executable(correspondence_grouping_bin correspondence_grouping.cpp
    DEPS object_recognition)

rock_executable(ply2pcd_bin ply2pcd.cpp
    DEPS object_recognition)

rock_executable(ply2pcd2_bin ply2pcd2.cpp
    DEPS object_recognition)

rock_executable(pcd_concatenator_bin pcd_concatenator.cpp
    DEPS object_recognition)

rock_executable(pcd_resize_bin pcd_resize.cpp
    DEPS object_recognition)

rock_executable(keypoints_bin keypoints.cpp
    DEPS object_recognition)

rock_executable(hypothesis_bin hypothesis.cpp
    DEPS object_recognition)

rock_executable(identification_bin identification.cpp
    DEPS object_recognition)

rock_executable(save_bin save_model.cpp
    DEPS object_recognition)
