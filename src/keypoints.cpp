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
bool use_normal_(true);
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
    pcl::PointCloud<PointType>::Ptr model_keypoints_uniform (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_keypoints_random (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_keypoints_normal (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr model_keypoints_iss (new pcl::PointCloud<PointType> ());
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
 
    
    //************UNIFORM SAMPLING **************
    pcl::PointCloud<int> sampled_indices_uniform;
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (model_ss_);
    uniform_sampling.compute (sampled_indices_uniform);
    pcl::copyPointCloud (*model, sampled_indices_uniform.points, *model_keypoints_uniform);
    std::cout << "Uniform Sampling Selected Keypoints: " << model_keypoints_uniform->size () << std::endl;
   

    //********NORMAL SPACE SAMPLING *************
    pcl::IndicesPtr sampled_indices_normal(new std::vector<int>());
    pcl::NormalSpaceSampling<PointType,NormalType> normal_space_sampling;
    normal_space_sampling.setInputCloud(model);
    normal_space_sampling.setNormals(model_normals);
    normal_space_sampling.setSeed(0);
    normal_space_sampling.setSample(static_cast<unsigned int>(model_normals->size())/10);


    normal_space_sampling.setBins(100,100,100);
    normal_space_sampling.filter(*sampled_indices_normal);
    pcl::copyPointCloud (*model, *sampled_indices_normal, *model_keypoints_normal);
    std::cout << "Normal Sampling Selected Keypoints: " << model_keypoints_normal->size () << std::endl;

    ////*********RANDOM SAMPLING *****************
    std::vector<int> sampled_indices_random;
    pcl::RandomSample<PointType> random_sample(true);
    random_sample.setInputCloud(model);
    int num_points = static_cast<unsigned int>(model->size()*0.1);
    random_sample.setSample(num_points);
    random_sample.filter(sampled_indices_random);
    pcl::copyPointCloud (*model, sampled_indices_random, *model_keypoints_random);
    std::cout << "Random Sampling Selected Keypoints: " << model_keypoints_random->size () << std::endl;
   
    ////***********ISS KEYPOINT ESTIMATOR**********
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
    double iss_salient_radius_ = 6 * resolution;
    double iss_non_max_radius_ = 4 * resolution;
    pcl::ISSKeypoint3D<PointType, PointType> iss_detector;
    iss_detector.setSearchMethod (tree);
    iss_detector.setSalientRadius (iss_salient_radius_);
    iss_detector.setNonMaxRadius (iss_non_max_radius_);
    iss_detector.setThreshold21 (0.975);
    iss_detector.setThreshold32 (0.975);
    iss_detector.setMinNeighbors (5);
    iss_detector.setNumberOfThreads (4);
    iss_detector.setInputCloud (model);
    iss_detector.compute (*model_keypoints_iss);
  
    std::cout << "ISS Sampling Selected Keypoints: " << model_keypoints_iss->size () << std::endl;
  
  ////
  ////  Compute Descriptor for keypoints
  ////
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);
  
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  
  
  
  //descr_est.setInputCloud (model_keypoints_uniform);
  //descr_est.compute (*model_descriptors);
  //std::cout << "Model total descriptors Uniform: " << model_descriptors->size () << std::endl;
  //
  //    
  //descr_est.setInputCloud (model_keypoints_random);
  //descr_est.compute (*model_descriptors);
  //std::cout << "Model total descriptors Random: " << model_descriptors->size () << std::endl;
  //
  //descr_est.setInputCloud (model_keypoints_iss);
  //descr_est.compute (*model_descriptors);
  //std::cout << "Model total descriptors ISS: " << model_descriptors->size () << std::endl;
  
  //descr_est.setInputCloud (model_keypoints_normal);
  //descr_est.compute (*model_descriptors);
  //std::cout << "Model total descriptors Normal: " << model_descriptors->size () << std::endl;
  //
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer ("Keypoints");

  //We are translating the model so that it doesn't end in the middle of the scene representation
  pcl::transformPointCloud (*model_keypoints_uniform, *model_keypoints_uniform, Eigen::Vector3f (0,0,0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_keypoints_normal, *model_keypoints_normal, Eigen::Vector3f (3,0,0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_keypoints_random, *model_keypoints_random, Eigen::Vector3f (6,0,0), Eigen::Quaternionf (1, 0, 0, 0));
  pcl::transformPointCloud (*model_keypoints_iss, *model_keypoints_iss, Eigen::Vector3f (9,0,0), Eigen::Quaternionf (1, 0, 0, 0));



  pcl::visualization::PointCloudColorHandlerCustom<PointType> uniform_color (model_keypoints_uniform, 255, 255, 255);
  viewer.addPointCloud (model_keypoints_uniform, uniform_color, "uniform_sampling");
 
  pcl::visualization::PointCloudColorHandlerCustom<PointType> normal_color (model_keypoints_normal, 0, 0, 255);
  viewer.addPointCloud (model_keypoints_normal, normal_color, "normal_sampling");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> random_color (model_keypoints_random, 255, 0, 0);
  viewer.addPointCloud (model_keypoints_random, random_color, "random_sampling");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> iss_color (model_keypoints_iss, 0, 255, 0);
  viewer.addPointCloud (model_keypoints_iss, iss_color, "iss_sampling");
  
  
  viewer.addCoordinateSystem (1.0); 
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce ();
  }

  return (0);
}

