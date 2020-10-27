#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <iostream>

int main(){
  cv::VideoCapture cap("test_countryroad.mp4");
  if(!cap.isOpened())
    return -1;

  cv::Mat frame;
  cv::Mat grayframe;
  std::vector<cv::Point2f> corners;
  while(1){
    if(!cap.read(frame)){
      std::cout << "Can't read file" << std::endl;
      break;
    }
    cv::resize(frame, frame, cv::Size(640, 540), 0, 0, cv::INTER_AREA);
    cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY, 0);
    cv::goodFeaturesToTrack(grayframe, corners, 10000, 0.01, 10, grayframe, 3, 3, false, 0.04);
    for(uint i = 0; i < corners.size(); ++i){
      cv::circle(frame, corners.at(i), 3, cv::Scalar(255, 25, 25), cv::FILLED);
    }
    imshow("vid", frame);
    if(cv::waitKey(30) == 27){
      break;
    }
  }
  return 0;
}
