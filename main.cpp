#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main(){
  cv::VideoCapture cap("test_countryroad.mp4");
  if(!cap.isOpened())
    return -1;

  cv::Mat frame;
  cv::Mat grayframe;
  std::vector<cv::Point2f> corners;
  std::vector<cv::KeyPoint> oldkp;
  cv::Mat oldgrayframe;
  cv::Mat desc;
  cv::Mat olddesc;
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> bfmatch = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> matches;
  std::vector<cv::DMatch> goodmatches;
  cv::Mat imgmatches;
  while(1){
    if(!cap.read(frame)){
      std::cout << "Can't read file" << std::endl;
      break;
    }
    cv::resize(frame, frame, cv::Size(640, 540), 0, 0, cv::INTER_AREA);
    cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY, 0);
    cv::goodFeaturesToTrack(grayframe, corners, 1000, 0.01, 10, grayframe, 3, 3, false, 0.04);
    std::vector<cv::KeyPoint> kp;
    for(size_t i = 0; i < corners.size(); i++){
      kp.push_back(cv::KeyPoint(corners[i], 1.f));
    }
    orb->compute(grayframe, kp, desc);
    for(uint i = 0; i < corners.size(); ++i){
      cv::circle(frame, corners.at(i), 3, cv::Scalar(255, 25, 25), cv::FILLED);
    }
    if(!olddesc.empty())
      bfmatch->knnMatch(desc, olddesc, matches, 2); 
    goodmatches.clear();
    for(uint i = 0; i < matches.size(); ++i){
        if(matches[i][0].distance < 0.75 * matches[i][1].distance)
          goodmatches.push_back(matches[i][0]);
    }
    std::cout << goodmatches.size() << std::endl;
    cv::Mat test;
    if(!olddesc.empty())
      cv::drawMatches(oldgrayframe, oldkp, grayframe, kp, goodmatches, test);
    imshow("vid", frame);
    oldgrayframe = grayframe;
    olddesc = desc;
    oldkp = kp;
    if(cv::waitKey(30) == 27){
      break;
    }
  }
  return 0;
}
