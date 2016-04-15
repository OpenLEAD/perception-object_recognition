#include <pcl/io/pcd_io.h>
#include "ORPointCloud.hpp"
#include <pcl/console/parse.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;


std::string model_filename_;

int 
main (int argc, char *argv[])
{

    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

    model_filename_ = argv[filenames[0]];
    
    ORPointCloud modelOR;
      
    modelOR.importCloud(model_filename_, false);
    modelOR.extractNormals();
    pcl::io::savePCDFile("normals.pcd", *modelOR.normals);
    return 0;

}
