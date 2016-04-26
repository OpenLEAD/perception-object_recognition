#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include "ORPointCloud.hpp"
#include <pcl/filters/normal_space.h>  
#include <pcl/filters/random_sample.h>
#include <pcl/filters/covariance_sampling.h>


typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

std::string model_filename_;

//Algorithm params
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
bool use_uniform_(false);
bool use_normal_(false);
bool use_random_(true);

float model_ss_ (5.0f);
float scene_ss_ (3.0f);
float rf_rad_ (0.115f);
float descr_rad_ (30.0f);
float cg_size_ (0.1f);
float cg_thresh_ (5.0f);

void
parseCommandLine (int argc, char *argv[])
{

  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 1)
  {
    std::cout << "Filenames missing.\n";
    exit (-1);
  }

  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }

  model_filename_ = argv[filenames[0]];

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}

double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

int
main (int argc, char *argv[])
{
  parseCommandLine (argc, argv);

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  
  //
  //  Load clouds
  //
  if (pcl::io::loadPCDFile (model_filename_, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    return (-1);
  }

  //
  //  Set up resolution invariance
 
  float resolution = static_cast<float> (computeCloudResolution (model));
  if (use_cloud_resolution_)
  { 
    if (resolution != 0.0f)
    {
      model_ss_   *= resolution;
      rf_rad_     *= resolution;
      descr_rad_  *= resolution;
      cg_size_    *= resolution;
    
      std::cout << "models_ss: " << model_ss_ << std::endl;
      std::cout << "rf_rad: " << rf_rad_ << std::endl;
      std::cout << "descr_rad: " << descr_rad_ << std::endl;
      std::cout << "cg_size: " << cg_size_ << std::endl;
    }
  }
  //
  //  Compute Normals
  //
  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);
  std::cout << "Model total normals: " << model_normals->size () << std::endl;
  //
  //  Downsample Clouds to Extract keypoints
  //
// if(use_random_||use_random_)
// {
//     std::vector<int> sampled_indices;
// } else
//      pcl::IndicesPtr sampled_indices(new std::vector<int>());/
  
 use_uniform_=true; 
// if(use_uniform_)
//  {
//      //std::vector<int> sampled_indices;
//      pcl::PointCloud<int> sampled_indices;
//      pcl::UniformSampling<PointType> uniform_sampling;
//      uniform_sampling.setInputCloud (model);
//      uniform_sampling.setRadiusSearch (model_ss_);
//      std::cout << "model_ss " << model_ss_ <<  std::endl;
//      uniform_sampling.compute (sampled_indices);
//      pcl::copyPointCloud (*model, sampled_indices.points, *model_keypoints);
//      std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
//  }
// }
//  if(use_normal_)
//  {
//      pcl::NormalSpaceSampling<PointType,NormalType> normal_space_sampling;
//      normal_space_sampling.setInputCloud(model);
//      normal_space_sampling.setNormals(model_normals);
//      normal_space_sampling.setBins(1000,1000,1000);
//      normal_space_sampling.setSeed(0);
//      normal_space_sampling.setSample(static_cast<unsigned int>(model_normals->size())/4);
//      normal_space_sampling.filter(*sampled_indices);
//      pcl::copyPointCloud (*model, *sampled_indices, *model_keypoints);
//      std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
//
// }
//if(use_random_)
//  {
//      std::vector<int> sampled_indices;
//      pcl::RandomSample<PointType> random_sample(true);
//      random_sample.setInputCloud(model);
//      int num_points = static_cast<unsigned int>(model->size()*0.1);
//      std::cout<<num_points<<std::endl;
//      random_sample.setSample(num_points);
//      random_sample.filter(sampled_indices);
//      pcl::copyPointCloud (*model, sampled_indices, *model_keypoints);
//      std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
//  }
//  
  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
  
  
   // Compute model_resolution
  
   double iss_salient_radius_ = 6 * resolution;
   
   double iss_non_max_radius_ = 4 * resolution;
  
   //
   // Compute keypoints
   //
   pcl::ISSKeypoint3D<PointType, PointType> iss_detector;
  
   iss_detector.setSearchMethod (tree);
   iss_detector.setSalientRadius (iss_salient_radius_);
   iss_detector.setNonMaxRadius (iss_non_max_radius_);
   iss_detector.setThreshold21 (0.975);
   iss_detector.setThreshold32 (0.975);
   iss_detector.setMinNeighbors (5);
   iss_detector.setNumberOfThreads (4);
   iss_detector.setInputCloud (model);
   iss_detector.compute (*model_keypoints);
  
  
  ////
  ////  Compute Descriptor for keypoints
  ////
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);
  
  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);
  std::cout << "Model total descriptors: " << model_descriptors->size () << std::endl;
  
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Keypoints");

  pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr off_scene_model_descriptors (new pcl::PointCloud<PointType> ());

  //We are translating the model so that it doesn't end in the middle of the scene representation
  pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (0,0,0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (0,0,0), Eigen::Quaternionf (1, 0, 0, 0));
  //pcl::transformPointCloud (*model_descriptors, *off_scene_model_descriptors, Eigen::Vector3f (0,0,0), Eigen::Quaternionf (1, 0, 0, 0));

  pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 255);
  viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");

 
  pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 255, 0, 0);
  viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 11, "off_scene_model_keypoints");

  //pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_descriptors_color_handler (off_scene_model_descriptors, 255, 0, 0);
  //viewer.addPointCloud (off_scene_model_descriptors, off_scene_model_descriptors_color_handler, "off_scene_model_descriptors");
  //viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_descriptors");



  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}

