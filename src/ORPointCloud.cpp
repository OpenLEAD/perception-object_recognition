#include"ORPointCloud.hpp"
#include <stdexcept>


ORPointCloud::ORPointCloud():
cloud (new pcl::PointCloud<PointType> ()),keypoints (new pcl::PointCloud<PointType> ()),
    normals (new pcl::PointCloud<NormalType> ()), descriptors (new pcl::PointCloud<DescriptorType> ()), reference_frames (new pcl::PointCloud<RFType>())
{
  cloud_resolution = 0.0;
}

void ORPointCloud::importCloud(const std::string &cloud_name, bool model=false)
{
    std::string file_name;
    std::string file_name_raw;

    size_t last_dot = cloud_name.find_last_of(".");
    if (last_dot == std::string::npos)
   {
        file_name = cloud_name+".pcd";
        file_name_raw = cloud_name;
    }
    else
    {
        file_name =  cloud_name;
        file_name_raw = cloud_name.substr(0,last_dot);
    }

    if (pcl::io::loadPCDFile (file_name, *cloud) < 0)
    {
        throw std::runtime_error("Cannot open Point cloud with name " + cloud_name);
    }
    if(model)
    {   
        std::string keypoints_name = file_name_raw + "_keypoints.pcd";
        std::string normals_name = file_name_raw + "_normals.pcd";
        std::string descriptors_name = file_name_raw + "_descriptors.pcd";     
        std::string rf_name = file_name_raw + "_rf.pcd";     
        if (pcl::io::loadPCDFile (keypoints_name, *keypoints) < 0)
        {
            throw std::runtime_error("Cannot open Model with name " + cloud_name);
        }
        if (pcl::io::loadPCDFile (normals_name, *normals) < 0)
        {
            throw std::runtime_error("Cannot open Model with name " + cloud_name);
        }
        if (pcl::io::loadPCDFile (descriptors_name, *descriptors) < 0)
        {
            throw std::runtime_error("Cannot open Model with name " + cloud_name);
        }
        if (pcl::io::loadPCDFile (rf_name, *reference_frames) < 0)
        {
            throw std::runtime_error("Cannot open Model with name " + cloud_name);
        }
   }
}  


void ORPointCloud::saveModel()
{
    
    pcl::io::savePCDFile(file_name_raw + "_keypoints.pcd", *keypoints);
    pcl::io::savePCDFile(file_name_raw + "_normals.pcd", *normals);
    pcl::io::savePCDFile(file_name_raw + "_descriptors.pcd", *descriptors);
    pcl::io::savePCDFile(file_name_raw + "_rf.pcd", *reference_frames);
}

void ORPointCloud::extractKeypoints(float cloud_ss , bool use_cloud_resolution)
{
     //if(use_cloud_resolution)
       //  cloud_ss *= cloud_resolution;
     
     pcl::PointCloud<int> sampled_indices;
     pcl::UniformSampling<PointType> uniform_sampling;
     uniform_sampling.setInputCloud (cloud);
     uniform_sampling.setRadiusSearch (cloud_ss);
     uniform_sampling.compute (sampled_indices);
     pcl::copyPointCloud (*cloud, sampled_indices.points, *keypoints);
     if(!keypoints) 
        std::cout << "No keypoints found!!!" << std::endl;
}


void ORPointCloud::extractNormals()
{
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (cloud);
    norm_est.compute (*normals);
    if(!normals)
        std::cout << "No normals found!!!" << std::endl;
}

void ORPointCloud::extractDescriptors(float descr_rad, bool use_cloud_resolution)
{
    //if(use_cloud_resolution)
      //descr_rad *= cloud_resolution;

    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (descr_rad);
    descr_est.setInputCloud (keypoints);
    descr_est.setInputNormals (normals);
    descr_est.setSearchSurface (cloud);
    descr_est.compute (*descriptors);

    if(!descriptors)
        std::cout << "No descriptors found!!!" << std::endl;
}


void ORPointCloud::computeCloudResolution ()
{
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
            cloud_resolution += sqrt (sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        cloud_resolution /= n_points;
    }
}

pcl::CorrespondencesPtr ORPointCloud::correspondences(const ORPointCloud* model, ORPointCloud* scene)
{
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());
    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud (model->descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene->descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        if (!pcl_isfinite (scene->descriptors->at (i).descriptor[0])) //skipping NaNs
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch (scene->descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back(corr);
        }
    }
    return model_scene_corrs;
}

void ORPointCloud::extractRF(float rf_rad)
{
    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles(true);
    rf_est.setRadiusSearch(rf_rad);

    rf_est.setInputCloud(keypoints);
    rf_est.setInputNormals(normals);
    rf_est.setSearchSurface(cloud);
    rf_est.compute(*reference_frames);
}

double ORPointCloud::computeCloudRMS(const ORPointCloud* target, pcl::PointCloud<PointType>::ConstPtr source, double max_range){

        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        tree->setInputCloud(target->cloud);
        double fitness_score = 0.0;

        std::vector<int> nn_indices (1);
        std::vector<float> nn_dists (1);

        // For each point in the source dataset
        int nr = 0;
        for (size_t i = 0; i < source->points.size (); ++i){
                //Avoid NaN points as they crash nn searches
                if(!pcl_isfinite((*source)[i].x)){
                        continue;
                }

                // Find its nearest neighbor in the target
                tree->nearestKSearch (source->points[i], 1, nn_indices, nn_dists);

                // Deal with occlusions (incomplete targets)
                if (nn_dists[0] <= max_range*max_range){
                        // Add to the fitness score
                        fitness_score += nn_dists[0];
                        nr++;
                }
        }

        if (nr > 0){
                return sqrt(fitness_score / nr)*1000.0;
        }else{
                return (std::numeric_limits<double>::max ());
        }
}                                                                                                                                                                                                                                                                                                          
