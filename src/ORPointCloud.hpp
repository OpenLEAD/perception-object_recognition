#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

class ORPointCloud 
{
  public:
    pcl::PointCloud<PointType>::Ptr cloud;
    pcl::PointCloud<PointType>::Ptr keypoints;
    pcl::PointCloud<NormalType>::Ptr normals;
    pcl::PointCloud<DescriptorType>::Ptr descriptors;
    pcl::PointCloud<RFType>::Ptr reference_frames;
    double cloud_resolution;
    std::string file_name;
    std::string file_name_raw;    
    
    ORPointCloud();
    void extractKeypoints(float cloud_ss, bool use_cloud_resolution);
    void extractNormals();
    void extractDescriptors(float descr_rad, bool use_cloud_resolution);
    void extractRF(float rf_rad);
    void importCloud(const std::string &file_name, bool model);
    void saveModel();
    void computeCloudResolution();
    static pcl::CorrespondencesPtr correspondences(const ORPointCloud* model, ORPointCloud* scene);    
    static double computeCloudRMS(const ORPointCloud* target,
            pcl::PointCloud<PointType>::ConstPtr source, double max_range); 
    

};
