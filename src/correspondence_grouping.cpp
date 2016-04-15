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
#include <pcl/registration/icp.h>

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

int
main (int argc, char *argv[])
{
    parseCommandLine (argc, argv);

    ORPointCloud modelOR;
    ORPointCloud sceneOR;

    modelOR.importCloud(model_filename_, false);
    modelOR.computeCloudResolution();

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
    modelOR.extractKeypoints(model_ss_,use_cloud_resolution_);
    modelOR.extractDescriptors(descr_rad_,use_cloud_resolution_);
    modelOR.extractRF(rf_rad_);

    sceneOR.importCloud(scene_filename_, false);
    sceneOR.computeCloudResolution();
    sceneOR.extractNormals();
    sceneOR.extractKeypoints(scene_ss_,use_cloud_resolution_);
    sceneOR.extractDescriptors(descr_rad_,use_cloud_resolution_);
    sceneOR.extractRF(rf_rad_);    

    pcl::CorrespondencesPtr model_scene_corrsOR = ORPointCloud::correspondences(&modelOR,&sceneOR);
    std::cout << "Correspondences found: " << model_scene_corrsOR->size () << std::endl;

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

    clusterer.setInputCloud (modelOR.keypoints);
    clusterer.setInputRf (modelOR.reference_frames);
    clusterer.setSceneCloud (sceneOR.keypoints);
    clusterer.setSceneRf (sceneOR.reference_frames);
    clusterer.setModelSceneCorrespondences (model_scene_corrsOR);

    clusterer.recognize (rototranslations, clustered_corrs);

    /**
    * Stop if no instances
    */
    if (rototranslations.size () <= 0)
    {
        cout << "*** No instances found! ***" << endl;
        return (0);
    }
    else
    {
        cout << "Recognized Instances: " << rototranslations.size () << endl << endl;
    }

    //Find the instance with the larger number of correspondences
    float min = 1000;
    int i_min=0;
    bool testing_model = true;
    std::cout << "Model instances found: " << rototranslations.size () << std::endl;

    Eigen::Vector3f trans; 
    Eigen::Matrix3f rot;
    //tf between the scene and rotor
    Eigen::Affine3f t_sr = Eigen::Affine3f::Identity();
    //for the model montagem_turbina_0001
    trans = Eigen::Vector3f(.0,2.0,-7.0);
    rot =  Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitY());

    t_sr.translation() = trans;  
    t_sr.linear()= rot;
    
    //tf between the rotor and the blade
    Eigen::Affine3f t_rb = Eigen::Affine3f::Identity();
    //tf given by the solid works
    trans = Eigen::Vector3f(0,0,-1265.741e-3);
    rot = Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitY())
                   *Eigen::AngleAxisf(0.5*M_PI, Eigen::Vector3f::UnitZ())
                       *Eigen::AngleAxisf(-31/180.0*M_PI, Eigen::Vector3f::UnitY());
    t_rb.linear() = rot;
    t_rb.translation() = trans;
    
    //the original tf calculated from the was with the model without any
    //rotation 
    Eigen::Affine3f t_r_rotation = Eigen::Affine3f::Identity();
    rot = Eigen::AngleAxisf(-0.25*M_PI, Eigen::Vector3f::UnitX());

    t_r_rotation.linear() = rot;  
    t_rb = t_r_rotation*t_rb;

    //tf between the model and the blade
    Eigen::Affine3f t_mb = Eigen::Affine3f::Identity();
    //for the model montagem_pa_0001
    trans = Eigen::Vector3f(.0,.0,-5.0);
    rot = Eigen::AngleAxisf(M_PI,  Eigen::Vector3f::UnitY());

    t_mb.translation() = trans;
    t_mb.linear()= rot;

    Eigen::Affine3f t_sb = t_sr*t_rb;
    
    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        //tf between the scene and the model (estimated by the calib. alg.)
    
        Eigen::Affine3f t_sm(rototranslations[i]); 

        Eigen::Affine3f t_error = t_sb.inverse()*t_sm*t_mb;
        Eigen::AngleAxisf rot_error(t_error.rotation());
        if( rot_error.angle() < min)
        {    
            i_min = i;
            min =  rot_error.angle();
        }
    }

    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*modelOR.cloud, *rotated_model, rototranslations[i_min]);
    
    Eigen::Affine3f t_sm_min(rototranslations[i_min]); 

    Eigen::Affine3f t_error_min = t_sb.inverse()*t_sm_min*t_mb;
    Eigen::AngleAxisf rot_error_min(t_error_min.rotation());

    Eigen::Vector3f trans_error = t_error_min.translation();

    std::cout << "Rotation Error before ICP: " << rot_error_min.angle() << std::endl;
    std::cout << "Translation Error before ICP: " << trans_error.norm() << std::endl;
   

    /**
     * ICP
     */
    cout << "--- ICP ---------" << endl;

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance (0.005f);
    icp.setInputTarget (sceneOR.cloud);
    icp.setInputSource (rotated_model);
    pcl::PointCloud<PointType>::Ptr registered (new pcl::PointCloud<PointType>);
    icp.align (*registered);

    if (icp.hasConverged ())
    {
      std::cout << "ICP has converged" << std::endl;
      
  
    } 
        //  Output results
    //
 
    Eigen::Affine3f t_icp(icp.getFinalTransformation()); 
    Eigen::Affine3f t_sm_icp = t_sm_min*t_icp;

    Eigen::AngleAxisf rot_icp(t_icp.rotation());
    Eigen::Vector3f trans_icp = t_icp.translation();
    
    std::cout << "Rotation Error ICP: " << rot_icp.angle() << std::endl;
    std::cout << "Translation Error ICP: " << trans_icp.norm() << std::endl;
   


    Eigen::Affine3f  t_error_icp = t_sb.inverse()*t_sm_icp*t_mb;
    Eigen::AngleAxisf rot_error_icp(t_error_icp.rotation());

    Eigen::Vector3f trans_error_icp = t_error_icp.translation();

    std::cout << "Rotation Error after ICP: " << rot_error_icp.angle() << std::endl;
    std::cout << "Translation Error after ICP: " << trans_error_icp.norm() << std::endl;
   

    // Print the rotation matrix and translation vector
    Eigen::Matrix3f rotation = rototranslations[i_min].block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslations[i_min].block<3,1>(0, 3);

    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
  
    printf("Max correspondences found %d \n ",min);
  
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

  
    if(min!=1000)
    {
       pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
       pcl::transformPointCloud (*modelOR.cloud, *rotated_model, rototranslations[i_min]);

       std::stringstream ss_cloud;
       ss_cloud << "instance" << i_min;

       pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
       viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
       double rms = ORPointCloud::computeCloudRMS(&sceneOR, rotated_model, 10.0);
       std::cout << "RMS " << rms << std::endl;
       if (show_correspondences_)
       {
           for (size_t j = 0; j < clustered_corrs[i_min].size (); ++j)
           {
               std::stringstream ss_line;
               ss_line << "correspondence_line" << i_min << "_" << j;
               PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i_min][j].index_query);
               PointType& scene_point = sceneOR.keypoints->at (clustered_corrs[i_min][j].index_match);

               //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
               viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
           }
      }
  
    }
   if(testing_model)
    {
    
            //origin of the scene
        viewer.addCoordinateSystem (1.0);
     
        if(show_correspondences_)
        {
            //Model is translated (-5,0,0) by default in the visualization. 
            viewer.addCoordinateSystem(1.0,-5.0,0.0,0.0);
            //origin of the blade in the correspondence model
            viewer.addCoordinateSystem(1.0, t_mb.translate(Eigen::Vector3f(-5.0,.0,.0)));     
        }
   
        viewer.addCoordinateSystem(1.0,t_sm_min*t_mb);
        viewer.addCoordinateSystem(1.0,t_sr);
        viewer.addCoordinateSystem(1.0,t_sr*t_rb);
     
    }

    viewer.setBackgroundColor (1, 1, 1);
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }
    return (0);
}
