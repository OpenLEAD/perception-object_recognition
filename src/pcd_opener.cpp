#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

int
main (int argc, char** argv)
{
    std::cout << "Opening " << argv[1] << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud) == -1) //* load the file
    {
       //PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
       return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height
        << " data points from test_pcd.pcd with the following fields: "<< std::endl;

    pcl::visualization::PCLVisualizer viewer ("PCD OPENER");
    viewer.addPointCloud (cloud, "cloud");

    while (!viewer.wasStopped ())
    {
       viewer.spinOnce ();
       pcl_sleep (0.01);
     }
    
    
    return (0);
}