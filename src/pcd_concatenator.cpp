#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Geometry>
#include <pcl/registration/transforms.h> 



int main (int argc, char** argv)
{


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);


    pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud1);
    pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *cloud2);

    Eigen::Matrix4f T2,Ts;
    float s1= 1;
    Ts << s1, 0, 0, 0,
          0, s1, 0, 0,
          0, 0, s1, 0,
          0, 0, 0,  1;

    transformPointCloud(*cloud1,*cloud1,Ts); 	

    Eigen::Vector4f c1,c2;
    pcl::compute3DCentroid (*cloud1, c1);
    pcl::compute3DCentroid (*cloud2, c2);

  
    Eigen::Affine3f T1 = Eigen::Affine3f::Identity();

    // Define a translation 
    T1.translation() << -c1[0]-1,-c1[1],-c1[2];

    double theta = M_PI/2;

    // The same rotation matrix as before; tetha radians arround Z axis
    T1.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

    // Print the transformation
    printf ("\nMethod #2: using an Affine3f\n");
    std::cout << T1.matrix() << std::endl;

   // T1 << 1, 0, 0, -c1[0]+4,
   //       0, 1, 0, -c1[1]+3,
   //       0, 0, 1, -c1[2],
   //       0, 0, 0, 1;

    T2 << 1, 0, 0, -c2[0],
          0, 1, 0, -c2[1],
          0, 0, 1, -c2[2],
          0, 0, 0, 1;

    transformPointCloud(*cloud1,*cloud1,T1);
    transformPointCloud(*cloud2,*cloud2,T2); 	
    *cloud+=*cloud1+*cloud2;

    pcl::io::savePCDFileASCII (argv[3], *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points" << std::endl;
    return (0);
}
     

