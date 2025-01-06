#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

class PointCloudProcessor : public rclcpp::Node {
public:
    PointCloudProcessor() : Node("point_cloud_processor"), ground_plane_found(false) {
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/SD0452002/rslidar/points", rclcpp::SensorDataQoS(),
            std::bind(&PointCloudProcessor::processPointCloud, this, std::placeholders::_1)
        );

        filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_pointcloud", 10);

        RCLCPP_INFO(this->get_logger(), "Point Cloud Processor Node has been started.");
    }

private:
    void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);

        // Height-based filtering to remove irrelevant points
        pcl::PassThrough<pcl::PointXYZ> height_filter;
        height_filter.setInputCloud(cloud);
        height_filter.setFilterFieldName("z");
        height_filter.setFilterLimits(-0.9, 2.0);  // Height range relative to LiDAR height of 1.5m
        height_filter.filter(*cloud);

        // Ground plane segmentation using RANSAC
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02);
        seg.setInputCloud(cloud);
        seg.segment(*ground_indices, *coefficients);

        if (ground_indices->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No ground plane found.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Ground plane found with %ld points", ground_indices->indices.size());

        // Extract non-ground points
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(ground_indices);
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*cloud_no_ground);

        RCLCPP_INFO(this->get_logger(), "Filtered cloud contains %ld points after removing ground", cloud_no_ground->points.size());

        // Apply Voxel Grid Downsampling to reduce point density
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(cloud_no_ground);
        voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);  // Leaf size of 10cm
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_grid.filter(*cloud_downsampled);

        // Publish the filtered and downsampled point cloud
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*cloud_downsampled, output_msg);
        output_msg.header.stamp = cloud_msg->header.stamp;
        output_msg.header.frame_id = "rslidar";
        filtered_cloud_pub_->publish(output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;

    bool ground_plane_found;
    Eigen::Vector4f ground_coefficients;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
