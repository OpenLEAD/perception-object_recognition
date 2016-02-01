#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <pcl/registration/transforms.h> 



int main (int argc, char** argv)
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud);

    Eigen::Matrix4f T2,Ts;
    float s1= atof(argv[3]);
    Ts << s1, 0, 0, 0,
          0, s1, 0, 0,
          0, 0, s1, 0,
          0, 0, 0,  1;

    transformPointCloud(*cloud,*cloud,Ts); 	

    pcl::io::savePCDFileASCII (argv[2], *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points" << std::endl;
    return (0);
}
