#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/segmentation/sac_segmentation.h>


#include <cstdio>
#include <iostream>

const static std::string FRAME_ID = "camera_depth_optical_frame";
static ros::Publisher PubOutput;

void tf_broadcast(const std::string frame_id){
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;
    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "camera_realsense";
    transformStamped.child_frame_id = frame_id;
    transformStamped.transform.translation.x = 0.0;
    transformStamped.transform.translation.y = 0.0;
    transformStamped.transform.translation.z = 0.0;
    tf2::Quaternion q;
    q.setRPY(-1.5707, 0, -1.5707);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();
    br.sendTransform(transformStamped); 
}

void downsample(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
{
  //Conversion
  pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2());
  pcl_conversions::toPCL(*cloud_msg, *cloud);

  pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
  pcl::VoxelGrid<pcl::PCLPointCloud2> cloud_msg_sample;
  cloud_msg_sample.setInputCloud(cloud);
  cloud_msg_sample.setLeafSize(0.1f, 0.1f, 0.1f);
  cloud_msg_sample.filter(*cloud_filtered);

  sensor_msgs::PointCloud2 output;
  pcl_conversions::moveFromPCL(*cloud_filtered, output);
  output.header.frame_id = FRAME_ID;
  PubOutput.publish(output);

  //tf
  tf_broadcast(FRAME_ID);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "downsample");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe("/realsense/depth/color/points", 1000, downsample);
  PubOutput = nh.advertise<sensor_msgs::PointCloud2>("/realsense/downsample", 1);

  ros::spin();
}