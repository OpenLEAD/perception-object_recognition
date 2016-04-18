/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2012, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *  Copyright (c) 2014, RadiantBlue Technologies, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 */

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace pcl::search;

typedef PointXYZ PointType;
typedef PointCloud<PointXYZ> Cloud;

void
printHelp (int, char **argv)
{
  print_error ("Syntax is: %s cloud_a.pcd cloud_b.pcd\n", argv[0]);
}

bool
loadCloud (const std::string &filename, Cloud &cloud)
{
  TicToc tt;
  print_highlight ("Loading "); print_value ("%s ", filename.c_str ());

  tt.tic ();
  if (loadPCDFile (filename, cloud) < 0)
    return (false);
  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : "); print_value ("%d", cloud.width * cloud.height); print_info (" points]\n");
  print_info ("Available dimensions: "); print_value ("%s\n", pcl::getFieldsList (cloud).c_str ());

  return (true);
}

void
compute (Cloud &cloud_a, Cloud &cloud_b)
{
  // Estimate
  TicToc tt;
  tt.tic ();

  print_highlight (stderr, "Computing ");

  int id_a, id_b, id_a2, id_b2 = 0;
  
  // compare A to B
  pcl::search::KdTree<PointType> tree_b;
  tree_b.setInputCloud (cloud_b.makeShared ());
  float max_dist_a = -std::numeric_limits<float>::max ();
  std::vector<double> distances; 
  
  for (size_t i = 0; i < cloud_a.points.size (); ++i)
  {
    std::vector<int> indices (1);
    std::vector<float> sqr_distances (1);

    tree_b.nearestKSearch (cloud_a.points[i], 1, indices, sqr_distances);
    if (sqr_distances[0] > max_dist_a)
    {
      max_dist_a = sqr_distances[0];
      id_a = i;
      id_b = indices[0];
    }
  }


  // compare B to A
  pcl::search::KdTree<PointType> tree_a;
  tree_a.setInputCloud (cloud_a.makeShared ());
  float max_dist_b = -std::numeric_limits<float>::max ();
  for (size_t i = 0; i < cloud_b.points.size (); ++i)
  {
    std::vector<int> indices (1);
    std::vector<float> sqr_distances (1);

    tree_a.nearestKSearch (cloud_b.points[i], 1, indices, sqr_distances);
    if (sqr_distances[0] > max_dist_b)
      max_dist_b = sqr_distances[0];
      id_a2=indices[0];
      id_b2=i;
  }

  max_dist_a = std::sqrt (max_dist_a);
  max_dist_b = std::sqrt (max_dist_b);

  float dist = std::max (max_dist_a, max_dist_b);

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms : ");
  print_info ("A->B: "); print_value ("%f", max_dist_a);
  print_info (", B->A: "); print_value ("%f", max_dist_b);
  print_info (", Hausdorff Distance: "); print_value ("%f", dist);
  print_info (" ]\n");



    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");

    pcl::PointCloud<PointType>::Ptr ptr_a(&cloud_a);
    pcl::PointCloud<PointType>::Ptr ptr_b(&cloud_b);
    viewer.addPointCloud<PointXYZ>(ptr_a, "cloud_a");
    viewer.addPointCloud<PointXYZ>(ptr_b, "cloud_b");

    PointXYZ& model_point_ab = cloud_a.points[id_a];
    PointXYZ& scene_point_ab = cloud_b.points[id_b];    
    PointXYZ& model_point_ba = cloud_a.points[id_a2];
    PointXYZ& scene_point_ba = cloud_b.points[id_b2];    
    std::stringstream ss_line;
    ss_line << "correspondence_line" << id_a << "_" << id_b;

    //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
    viewer.addLine<PointXYZ, PointXYZ> (model_point_ab, scene_point_ab, 0, 255, 0, ss_line.str ());
    ss_line << "correspondence_line" << id_a2 << "_" << id_b2;
    viewer.addLine<PointXYZ, PointXYZ> (model_point_ba, scene_point_ba, 255, 0, 0, ss_line.str ());
    while (!viewer.wasStopped ())
    {
        viewer.spinOnce ();
    }

}

/* ---[ */
int
main (int argc, char** argv)
{
  print_info ("Compute Hausdorff distance between point clouds. For more information, use: %s -h\n", argv[0]);

  if (argc < 3)
  {
    printHelp (argc, argv);
    return (-1);
  }

  // Parse the command line arguments for .pcd files
  std::vector<int> p_file_indices;
  p_file_indices = parse_file_extension_argument (argc, argv, ".pcd");
  if (p_file_indices.size () != 2)
  {
    print_error ("Need two PCD files to compute Hausdorff distance.\n");
    return (-1);
  }

  // Load the first file
  Cloud::Ptr cloud_a (new Cloud);
  if (!loadCloud (argv[p_file_indices[0]], *cloud_a))
    return (-1);

  // Load the second file
  Cloud::Ptr cloud_b (new Cloud);
  if (!loadCloud (argv[p_file_indices[1]], *cloud_b))
    return (-1);

  // Compute the Hausdorff distance
  compute (*cloud_a, *cloud_b);
}