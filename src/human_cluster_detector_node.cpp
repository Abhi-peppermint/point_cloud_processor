#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/filter.h>
#include <pcl/common/common.h>  // For getMinMax3D

class HumanClusterDetector : public rclcpp::Node {
public:
    HumanClusterDetector() : Node("human_cluster_detector") {
        filtered_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "filtered_pointcloud", rclcpp::SensorDataQoS(),
            std::bind(&HumanClusterDetector::processFilteredCloud, this, std::placeholders::_1)
        );

        clustered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("clustered_pointcloud", 10);
        RCLCPP_INFO(this->get_logger(), "Human Cluster Detector Node has been started.");
    }

private:
    void processFilteredCloud(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud_filtered);

        // Remove NaN or invalid points using a PassThrough filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_nan(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("z");  // Filter points based on the z-axis
        pass.setFilterLimits(-10.0, 10.0);  // Adjust these limits as per your environment
        pass.filter(*cloud_no_nan);

        // Check for NaN and remove them
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_no_nan, *cloud_no_nan, indices);

        // Height-based filtering relative to LiDAR height (1.5 m)
        pcl::PassThrough<pcl::PointXYZ> pass_height;
        pass_height.setInputCloud(cloud_no_nan);
        pass_height.setFilterFieldName("z");
        pass_height.setFilterLimits(-1.0, 2.0);  // More lenient height range to retain useful clusters
        pass_height.filter(*cloud_no_nan);

        // Perform clustering
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_no_nan);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.3);  // 30cm for cluster tolerance to detect human-sized clusters
        ec.setMinClusterSize(30);  // Reduced minimum cluster size to retain smaller clusters
        ec.setMaxClusterSize(3000);  // Increased maximum cluster size to handle groups of people
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_no_nan);
        ec.extract(cluster_indices);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        // Assign unique colors to each cluster and filter out walls and pillars
        int cluster_id = 0;
        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr single_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& index : indices.indices) {
                single_cluster->points.push_back(cloud_no_nan->points[index]);
            }

            // Evaluate cluster dimensions to identify potential humans
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*single_cluster, min_pt, max_pt);
            float height = max_pt[2] - min_pt[2];
            float width = std::max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]);

            // Adjusted filter conditions for human-like clusters
            if (height > 0.8 && height < 2.5 && width > 0.2 && width < 1.0) {
                // This cluster is likely a human, color it
                uint8_t r = (rand() % 256);
                uint8_t g = (rand() % 256);
                uint8_t b = (rand() % 256);

                for (const auto& index : indices.indices) {
                    pcl::PointXYZRGB point;
                    point.x = cloud_no_nan->points[index].x;
                    point.y = cloud_no_nan->points[index].y;
                    point.z = cloud_no_nan->points[index].z;
                    point.r = r;
                    point.g = g;
                    point.b = b;

                    clustered_cloud->points.push_back(point);
                }

                RCLCPP_INFO(this->get_logger(), "Cluster %d is likely a human with %ld points", cluster_id++, indices.indices.size());
            } else {
                RCLCPP_INFO(this->get_logger(), "Cluster %d filtered out (not human-like) with %ld points", cluster_id++, indices.indices.size());
            }
        }

        // Publish the clustered point cloud
        sensor_msgs::msg::PointCloud2 clustered_msg;
        pcl::toROSMsg(*clustered_cloud, clustered_msg);
        clustered_msg.header.stamp = cloud_msg->header.stamp;
        clustered_msg.header.frame_id = "rslidar";
        clustered_cloud_pub_->publish(clustered_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr clustered_cloud_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<HumanClusterDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
