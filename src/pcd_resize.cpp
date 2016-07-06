#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <pcl/registration/transforms.h> 


void resize(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float s)
{

    Eigen::Matrix4f Ts;
    Ts << s, 0, 0, 0,
          0, s, 0, 0,
          0, 0, s, 0,
          0, 0, 0,  1;

    pcl::transformPointCloud(*cloud,*cloud,Ts); 	

}


int main (int argc, char** argv)
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud);

    float s1= atof(argv[3]);

    resize(cloud,s1);
    pcl::io::savePCDFileASCII (argv[2], *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points" << std::endl;
    return (0);
}
