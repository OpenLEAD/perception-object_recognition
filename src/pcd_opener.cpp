#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

int
main (int argc, char** argv)
{
    std::cout << "Opening " << argv[1] << std::endl;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (argv[1], *cloud) == -1) //* load the file
    {
       //PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
       return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height
        << " data points from test_pcd.pcd with the following fields: "<< std::endl;
    pcl::visualization::PCLVisualizer viewer ("PCD OPENER");

    Eigen::Vector3f trans; 
    Eigen::Matrix3f rot;
    //tf between the model and the blade
    Eigen::Affine3f t_mb = Eigen::Affine3f::Identity();
    //for the model montagem_pa_0001
    trans = Eigen::Vector3f(.0,.0,-5.0);
    rot = Eigen::AngleAxisf(M_PI,  Eigen::Vector3f::UnitY());
    t_mb.translation() = trans;
    t_mb.linear()= rot;

    for (pcl::PointCloud<pcl::PointXYZRGBA>::iterator cloud_it = cloud->begin (); cloud_it != cloud->end (); ++cloud_it)
    {
        cloud_it->r=0;
        cloud_it->g=0;
        cloud_it->b=0;
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::io::loadPCDFile<pcl::PointXYZRGBA>("models/montagem_pa_corrigida00000.pcd", *cloud2);
 
    viewer.addPointCloud (cloud, "cloud");
    //viewer.addPointCloud (cloud2, "cloud2");
    viewer.setBackgroundColor (1, 1, 1);
   
    
    //viewer.addCoordinateSystem(100.0);


    while (!viewer.wasStopped ())
    {
       viewer.spinOnce ();
       pcl_sleep (0.01);
     }
    
    
    return (0);
}

