/*
 * Multi-Actor Camera Capture Node
 *
 * Subscribes to RTSP streams (via GStreamer) and publishes to ROS 2 topics
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

class CameraCaptureNode : public rclcpp::Node
{
public:
  CameraCaptureNode() : Node("camera_capture_node")
  {
    // Declare parameters
    this->declare_parameter("camera_id", "camera_0");
    this->declare_parameter("rtsp_url", "rtsp://192.168.1.100:554/stream");
    this->declare_parameter("publish_rate", 30.0);

    // Get parameters
    camera_id_ = this->get_parameter("camera_id").as_string();
    rtsp_url_ = this->get_parameter("rtsp_url").as_string();
    publish_rate_ = this->get_parameter("publish_rate").as_double();

    // Create publisher
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "camera/" + camera_id_ + "/image_raw", 10);

    // Create timer
    timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_),
      std::bind(&CameraCaptureNode::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "Camera capture node started: %s", camera_id_.c_str());
  }

private:
  void timer_callback()
  {
    // TODO: Integrate GStreamer RTSP capture
    // For now, publish dummy image
    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp = this->now();
    msg.header.frame_id = camera_id_;
    msg.height = 480;
    msg.width = 640;
    msg.encoding = "rgb8";
    msg.step = msg.width * 3;
    msg.data.resize(msg.height * msg.step);

    image_pub_->publish(msg);
  }

  std::string camera_id_;
  std::string rtsp_url_;
  double publish_rate_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraCaptureNode>());
  rclcpp::shutdown();
  return 0;
}
