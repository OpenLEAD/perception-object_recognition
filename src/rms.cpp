#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include "ORPointCloud.hpp"
#include <pcl/point_cloud.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>    
    
typedef pcl::PointXYZRGBA PointType;
int
main (int argc, char** argv)
{
    //Model & scene filenames
    std::vector<int> filenames;
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
    if (filenames.size () != 2)
    {
        std::cout << "Filenames missing.\n";
        exit (-1);
    }
    
    std::string model_filename_ = argv[filenames[0]];
    std::string scene_filename_ = argv[filenames[1]];

    ORPointCloud modelOR;
    ORPointCloud sceneOR;

    modelOR.importCloud(model_filename_, false);
    sceneOR.importCloud(scene_filename_, false);
    
    double rms = ORPointCloud::computeCloudRMS(&sceneOR, modelOR.cloud, 5.0);
    std::cout << "RMS " << rms << std::endl;

    
    pcl::PointCloud<PointType>::Ptr registered (new pcl::PointCloud<PointType>);
    
    pcl::GeneralizedIterativeClosestPoint<PointType,PointType> gicp;
    gicp.setInputCloud(modelOR.cloud);
    gicp.setInputTarget(sceneOR.cloud);
    gicp.setTransformationEpsilon(1e-12);
    gicp.setEuclideanFitnessEpsilon(1e-6);
    gicp.align(*registered);
    
    if(gicp.hasConverged())
    {
        rms = ORPointCloud::computeCloudRMS(&sceneOR, registered, 5.0);
        std::cout << "RMS " << rms << std::endl;

        std::cout << "Fitness Score: "<< gicp.getFitnessScore() <<std::endl;
    }    
    
    
    
    
    
    pcl::visualization::PCLVisualizer viewer ("PCD OPENER");
    viewer.addPointCloud (sceneOR.cloud, "cloud");
    viewer.addPointCloud (modelOR.cloud, "model");
    viewer.setBackgroundColor (1, 1, 1);
 
    while (!viewer.wasStopped ())
    {
       viewer.spinOnce ();
       pcl_sleep (0.01);
     }
    
    
    return (0);
}

