#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/mat.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <sstream>
#include <fstream>

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	
{
  
  std::string line;
  int i = 0;
  std::ifstream myfile("/home/rahul/CPP/00/00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    std::cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}
void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
                     std::vector<uchar>& status)	
{ 

  std::vector<float> err;					
  cv::Size winSize = cv::Size(21,21);																								
  cv::TermCriteria termcrit= cv::TermCriteria(cv::TermCriteria::COUNT+ cv::TermCriteria::EPS, 30, 0.01);

  cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points2.at(i- indexCorrection);
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
     		  if((pt.x<0)||(pt.y<0))	{
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}

     }

}
void matchdescriptors(std::vector<cv::KeyPoint> &Srckyepoints, std::vector<cv::KeyPoint> &Refkeypoints,
                         cv::Mat &Srcdescriptors, cv::Mat &Refdescriptors, std::vector<cv::DMatch> matches)

{
    cv::Ptr<cv::DescriptorMatcher> matcher;
    matcher = cv::FlannBasedMatcher::create();
    matcher->match(Srcdescriptors,Refdescriptors,matches);
    std::cout << matches.size() << std::endl;

}

void keypointdescriptors(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors)
{
    cv::Ptr<cv::DescriptorExtractor> extractor;
    int threshold = 30;        
    int octaves = 3;           
    float patternScale = 1.0f;
    
    extractor = cv::BRISK::create(threshold, octaves, patternScale);
    extractor->compute(img,keypoints,descriptors);
}

void detectfeatures(cv::Mat &img, std::vector<cv::Point2f> &point1)
{
    std::vector<cv::KeyPoint> keypoints; 
    cv::Ptr<cv::FeatureDetector> detector;
    detector = cv::FastFeatureDetector::create();
    detector->detect(img, keypoints);  
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage,cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::KeyPoint::convert(keypoints,point1,std::vector<int>());
    // cv::namedWindow("keypoints", 2);
    // cv::imshow("keypoints", visImage);
    // cv::waitKey(0);  

}

int main(int argc, const char *argv[])
{
    double scale = 1.00;

    cv::Mat img1 = cv::imread("/home/rahul/CPP/00/image_0/000000.png");
    cv::Mat img2 = cv::imread("/home/rahul/CPP/00/image_0/000001.png");
    std::string windowname = "Image";
    
    cv::Mat img_gray1, img_gray2;
    cv::cvtColor(img1, img_gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img_gray2, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> kp; 
    std::vector<cv::Point2f> points1, points2;

    detectfeatures(img_gray1, points1);

    std::vector<uchar> status;

    featureTracking(img_gray1,img_gray2,points1,points2, status);

    cv::Mat cameraMatrix = (cv::Mat1d(3,3) << 718.8560, 0.0, 607.1928, 0.0, 718.8560, 185.2157, 0.0, 0.0, 1.0 );
    
    cv::Mat E;
    cv::Mat mask;
    E = cv::findEssentialMat(points2, points1, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);

    cv::Mat R, t;
    cv::recoverPose(E, points2, points1, cameraMatrix, R, t, mask);

    cv::Mat prevImage = img_gray2;
    cv::Mat currImage;

    std::vector<cv::Point2f> prevFeatures = points2;
    std::vector<cv::Point2f> currFeatures;

    cv::Mat R_f = R.clone();
    cv::Mat t_f = t.clone();

    cv::namedWindow("front camera", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    cv::Mat traj = cv::Mat::zeros(600,600,CV_8UC3);

    char filename[100];
    char text[100];
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    cv::Point textOrg(10, 50);
    double fontScale = 1;
    int thickness = 1;
    for(int numFrame = 2; numFrame < 1000; numFrame++)
    {
        std::sprintf(filename,"/home/rahul/CPP/00/image_0/%06d.png", numFrame);
        cv::Mat currImage_c = cv::imread(filename);
        cv::cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY);
        std::vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        E = cv::findEssentialMat(currFeatures, prevFeatures, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t, mask);

        cv::Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);


        for(int i = 0; i < prevFeatures.size();i++)
        {
            prevPts.at<double>(0,i) = prevFeatures.at(i).x;
            prevPts.at<double>(1,i) = prevFeatures.at(i).y;
            
            currPts.at<double>(0,i) = currFeatures.at(i).x;
            currPts.at<double>(1,i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        if((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
          t_f = t_f + scale * (R_f * t);

          R_f = R * R_f;
        }

        // t_f = t_f + (R_f * t);

        // R_f = R * R_f;

        if(prevFeatures.size() < 2000 )
        {
          detectfeatures(prevImage, prevFeatures);
          featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        cv::circle(traj, cv::Point(x,y), 1, CV_RGB(255, 0, 0), 2);
        
        cv::rectangle(traj, cv::Point(10,30), cv::Point(550, 50), CV_RGB(0,0,0), cv::FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        putText(traj, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        
        cv::imshow( "Road facing camera", currImage_c );
        cv::imshow( "Trajectory", traj );
        cv::waitKey(1);

    }
    
    // cv::Mat descriptors;
    // keypointdescriptors(kp, img_gray, descriptors);

    // matchdescriptors


    // cv::imshow(windowname, img_gray1);
    // cv::waitKey(0);
    return 0;

}
