#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include "ORPointCloud.hpp"
#include <boost/graph/graph_concepts.hpp>
//Standard ICP
#include <pcl/registration/icp.h>
//Generalized ICP
#include <pcl/registration/gicp.h>
#include <algorithm>

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

std::string model_filename_;
std::string scene_filename_;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool save_alingment_(false);
bool use_hough_ (true);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);

void
showHelp (char *filename)
{
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_filename.pcd scene_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                     Show this help." << std::endl;
  std::cout << "     -k:                     Show used keypoints." << std::endl;
  std::cout << "     -c:                     Show used correspondences." << std::endl;
  std::cout << "     -s:                     Save Alignment." << std::endl;
  std::cout << "     -r:                     Compute the model cloud resolution and multiply" << std::endl;
  std::cout << "                             each radius given by that value." << std::endl;
  std::cout << "     --algorithm (Hough|GC): Clustering algorithm used (default Hough)." << std::endl;
  std::cout << "     --model_ss val:         Model uniform sampling radius (default 0.01)" << std::endl;
  std::cout << "     --scene_ss val:         Scene uniform sampling radius (default 0.03)" << std::endl;
  std::cout << "     --rf_rad val:           Reference frame radius (default 0.015)" << std::endl;
  std::cout << "     --descr_rad val:        Descriptor radius (default 0.02)" << std::endl;
  std::cout << "     --cg_size val:          Cluster size (default 0.01)" << std::endl;
  std::cout << "     --cg_thresh val:        Clustering threshold (default 5)" << std::endl << std::endl;
}

void
parseCommandLine (int argc, char *argv[])
{
  //Show help
  if (pcl::console::find_switch (argc, argv, "-h"))
  {
    showHelp (argv[0]);
    exit (0);
  }

  //Model & scene filenames
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  if (filenames.size () != 2)
  {
    std::cout << "Filenames missing.\n";
    showHelp (argv[0]);
    exit (-1);
  }

  model_filename_ = argv[filenames[0]];
  scene_filename_ = argv[filenames[1]];

  //Program behavior
  if (pcl::console::find_switch (argc, argv, "-k"))
  {
    show_keypoints_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-c"))
  {
    show_correspondences_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-r"))
  {
    use_cloud_resolution_ = true;
  }
  if (pcl::console::find_switch (argc, argv, "-s"))
  {
    save_alingment_ = true;
  }
  std::string used_algorithm;
  if (pcl::console::parse_argument (argc, argv, "--algorithm", used_algorithm) != -1)
  {
    if (used_algorithm.compare ("Hough") == 0)
    {
      use_hough_ = true;
    }else if (used_algorithm.compare ("GC") == 0)
    {
      use_hough_ = false;
    }
    else
    {
      std::cout << "Wrong algorithm name.\n";
      showHelp (argv[0]);
      exit (-1);
    }
  }

  //General parameters
  pcl::console::parse_argument (argc, argv, "--model_ss", model_ss_);
  pcl::console::parse_argument (argc, argv, "--scene_ss", scene_ss_);
  pcl::console::parse_argument (argc, argv, "--rf_rad", rf_rad_);
  pcl::console::parse_argument (argc, argv, "--descr_rad", descr_rad_);
  pcl::console::parse_argument (argc, argv, "--cg_size", cg_size_);
  pcl::console::parse_argument (argc, argv, "--cg_thresh", cg_thresh_);
}


void localizeModel(const ORPointCloud* model, const ORPointCloud* scene, 
        pcl::PointCloud<PointType>::Ptr model_aligned)
{
    pcl::CorrespondencesPtr model_scene_corrs = ORPointCloud::correspondences(model,scene);
    std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;
    //
    //  Actual Clustering
    //
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;


    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model->keypoints);
    clusterer.setInputRf (model->reference_frames);
    clusterer.setSceneCloud (scene->keypoints);
    clusterer.setSceneRf (scene->reference_frames);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    clusterer.recognize (rototranslations, clustered_corrs);
    
    /**
    * Stop if no instances
    */
    if (rototranslations.size () <= 0)
    {
        cout << "*** No instances found! ***" << endl;
        //return (0);
    }
    else
    {
        cout << "Recognized Instances: " << rototranslations.size () << endl << endl;
    }
     //Find the instance with the larger number of correspondences
    
    double rms_min=5*model->cloud_resolution;  
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    
    int max =0;
    int i_max=-1;

    std::vector<int> decrescent_corrs;
    for(size_t i=0;i<clustered_corrs.size();i++)
    {
        if(clustered_corrs[i].size()>max)
        {
            max=clustered_corrs[i].size();
            i_max=i;
        }
    }

    //for (size_t i = 0; i < rototranslations.size(); i++)
    //{
    //    pcl::transformPointCloud (*model->cloud, *rotated_model, rototranslations[i]);
    //    
    //    double rms = ORPointCloud::computeCloudRMS(&scene-> rotated_model, 10.0);
    //    std::cout<< "Instance " << i << " RMS: " << rms << std::endl;
    //    
    //    if(rms<rms_min)
    //    {
    //        i_min=i;
    //        rms_min = rms;
    //    }
    //}

    //if(i_min==-1)
    //{
    //    cout << "*** No instance passed the RMS criteria ***" << endl;
    //    return(0);
    //}
    
    pcl::transformPointCloud (*model->cloud, *rotated_model, rototranslations[i_max]);
    double rms = ORPointCloud::computeCloudRMS(scene, rotated_model, 10.0);
    std::cout<< "Instance RMS: " << rms << std::endl;

    /**
     * ICP
     */
    cout << "--- ICP ---------" << endl;

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance (0.005f);
    icp.setInputTarget (scene->cloud);
    icp.setInputCloud (rotated_model);
    icp.align (*model_aligned);

    if(save_alingment_)
    {
        std::string alignament_name = scene->file_name_raw + "_aligned.pcd";
        pcl::io::savePCDFile(alignament_name,*model_aligned); 
        std::cout << "alingment model saved "<< alignament_name << std::endl; 
    }   

    if (icp.hasConverged ())
    {
      std::cout << "ICP has converged" << std::endl;  
  
    }
    else
    {
        cout << "ICP couldn't converge, aborting" << endl;
    //    return(0);
    }
    
    // Output results
    //
 
    Eigen::Affine3f t_icp(icp.getFinalTransformation()); 
    
    double icp_rms = ORPointCloud::computeCloudRMS(scene, model_aligned, 10.0);
    std::cout<< "RMS after ICP alingment " <<  icp_rms << std::endl;
}


int
main (int argc, char *argv[])
{
    parseCommandLine (argc, argv);

    ORPointCloud modelOR;
    ORPointCloud sceneOR;

    modelOR.importCloud(model_filename_, false);
    modelOR.computeCloudResolution();

    sceneOR.importCloud(scene_filename_, false);
    sceneOR.computeCloudResolution();
    if (use_cloud_resolution_)
    {  
        if (modelOR.cloud_resolution != 0.0f)
        {
          model_ss_   *= modelOR.cloud_resolution;
          scene_ss_   *= modelOR.cloud_resolution;
          rf_rad_     *= modelOR.cloud_resolution;
          descr_rad_  *= modelOR.cloud_resolution;
          cg_size_    *= modelOR.cloud_resolution;
        }

        std::cout << "Model resolution:       " << modelOR.cloud_resolution << std::endl;
        std::cout << "Model sampling size:    " << model_ss_ << std::endl;
        std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
        std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
        std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
        std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
    }



    modelOR.extractNormals();
    sceneOR.extractNormals();
    std::cout << "Normals Computed" << std::endl;    
    
    modelOR.extractKeypoints(model_ss_,2);
    sceneOR.extractKeypoints(scene_ss_,2);
    std::cout << "Keypoints Computed" << std::endl;
    
    modelOR.extractDescriptors(descr_rad_,use_cloud_resolution_);
    sceneOR.extractDescriptors(descr_rad_,use_cloud_resolution_);
    std::cout << "Descriptors Computed" << std::endl;

    modelOR.extractRF(rf_rad_);
    sceneOR.extractRF(rf_rad_);    
    std::cout << "RF Computed" << std::endl;


    pcl::CorrespondencesPtr model_scene_corrsOR = ORPointCloud::correspondences(&modelOR,&sceneOR);
    std::cout << "Correspondences found: " << model_scene_corrsOR->size () << std::endl;



    pcl::PointCloud<PointType>::Ptr model_aligned (new pcl::PointCloud<PointType>);
   
    localizeModel(&modelOR,&sceneOR,model_aligned);
  
    //
    //  Visualization
    //
    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    viewer.addPointCloud (sceneOR.cloud, "scene_cloud");

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    if (show_correspondences_ || show_keypoints_)
    {
         //  We are translating the model so that it doesn't end in the middle of the scene representation
         pcl::transformPointCloud (*modelOR.cloud, *off_scene_model, Eigen::Vector3f (-5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
         pcl::transformPointCloud (*modelOR.keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-5,0,0), Eigen::Quaternionf (1, 0, 0, 0));
         pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
         viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
    }

    if (show_keypoints_)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (sceneOR.keypoints, 0, 0, 255);
        viewer.addPointCloud (sceneOR.keypoints, scene_keypoints_color_handler, "scene_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene_keypoints");

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
        viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "off_scene_model_keypoints");
    }


   std::stringstream ss_cloud;
   ss_cloud << "instance";

   pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (model_aligned, 255, 0, 0);
   viewer.addPointCloud (model_aligned, rotated_model_color_handler, ss_cloud.str ());
  // if (show_correspondences_)
  // {
  //     for (size_t j = 0; j < clustered_corrs[i_max].size (); ++j)
  //     {
  //         std::stringstream ss_line;
  //         ss_line << "correspondence_line" << "_" << j;
  //         PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i_max][j].index_query);
  //         PointType& scene_point = sceneOR.keypoints->at (clustered_corrs[i_max][j].index_match);

  //         //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
  //         viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
  //     }
  //  }
  
    
    viewer.setBackgroundColor (1, 1, 1);
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
    return (0);
}
