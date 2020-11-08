#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

int main(){
  cv::VideoCapture cap("test_countryroad.mp4");
  if(!cap.isOpened())
    return -1;

  cv::Mat cameramatrix = cv::Mat::zeros(3, 3, CV_64F);
  cameramatrix.at<double>(0, 0) = 800;
  cameramatrix.at<double>(0, 2) = 320;
  cameramatrix.at<double>(1, 1) = 800;
  cameramatrix.at<double>(1, 2) = 270;
  cameramatrix.at<double>(2, 2) = 1;
  cv::Mat R;
  cv::Mat t;
  cv::Mat output;
  cv::Mat fundamental;
  cv::Mat essential;
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
  std::vector<cv::KeyPoint> kp;
  cv::Mat imgmatches;
  while(1){
    goodmatches.clear();
    matches.clear();
    kp.clear();
    if(!cap.read(frame)){
      std::cout << "Can't read file" << std::endl;
      break;
    }
    cv::resize(frame, frame, cv::Size(640, 540), 0, 0, cv::INTER_AREA);
    cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY, 0);
    cv::goodFeaturesToTrack(grayframe, corners, 1000, 0.01, 10, grayframe, 3, 3, false, 0.04);
    for(size_t i = 0; i < corners.size(); i++){
      kp.push_back(cv::KeyPoint(corners[i], 1.f));
    }
    orb->compute(grayframe, kp, desc);
    for(uint i = 0; i < corners.size(); ++i){
      cv::circle(frame, corners.at(i), 3, cv::Scalar(255, 25, 25), cv::FILLED);
    }
    if(!olddesc.empty())
      bfmatch->knnMatch(desc, olddesc, matches, 2); 
    for(uint i = 0; i < matches.size(); ++i){
        if(matches[i][0].distance < 0.75 * matches[i][1].distance)
          goodmatches.push_back(matches[i][0]);
    }
    std::vector<int> pointindex1;
    std::vector<int> pointindex2;
    for(std::vector<cv::DMatch>::const_iterator it = goodmatches.begin(); it != goodmatches.end(); ++it){
      pointindex1.push_back(it->queryIdx);
      pointindex2.push_back(it->trainIdx);
    }
    std::vector<cv::Point2f> selectpoints1;
    std::vector<cv::Point2f> selectpoints2;
    cv::KeyPoint::convert(kp, selectpoints1, pointindex1);
    cv::KeyPoint::convert(oldkp, selectpoints2, pointindex2);
    if(!olddesc.empty()){
      //fundamental = cv::findFundamentalMat(cv::Mat(selectpoints1), cv::Mat(selectpoints2), cv::FM_RANSAC);
      essential = cv::findEssentialMat(cv::Mat(selectpoints1), cv::Mat(selectpoints2), cameramatrix);
      std::vector<cv::Point3f> lines;
      //computeCorrespondEpilines(selectpoints1, 1, fundamental, lines);
      for(uint i = 0; i < selectpoints1.size(); ++i){
        //std::cout << selectpoints1.at(i) << std::endl;
        //std::cout << lines.at(i) << std::endl;
        //cv::line(frame, selectpoints1.at(i), selectpoints2.at(i), cv::Scalar(5,100, 200)); 
      }
      cv::recoverPose(essential, selectpoints1, selectpoints2, cameramatrix, R, t);
      std::cout << t.size() << std::endl;
      std::cout << t.at<double>(0) <<std::endl;
      std::cout << t.at<double>(1) <<std::endl;
      std::cout << t.at<double>(2) <<std::endl;
      std::cout << frame.size() << std::endl;
      for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
          //std::cout << essential.at<double>(i, j) << " ";
        }
        //std::cout << std::endl;
      }
        cv::line(frame, cv::Point(30, 30), cv::Point(50 + 100*t.at<double>(0), 50 + 100*t.at<double>(1)), cv::Scalar(200,100, 200)); 
      //cv::drawMatches(oldgrayframe, oldkp, grayframe, kp, goodmatches, output);
      imshow("vid", frame);
    }
    oldgrayframe = grayframe;
    olddesc = desc;
    oldkp = kp;
    if(cv::waitKey(30) == 27){
      break;
    }
  }
  return 0;
}
