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



double getAbsoluteScale(int fram_id, int sequence_id, double z_cal)
{
    string line;
    int i = 0;
    ifstream myfile ("")

} 
void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2,
                     std::vector<uchar>& status)	
{ 

//this function automatically gets rid of points for which tracking fails

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

    cv::Mat img1 = cv::imread("/home/rahul/CPP/2011_09_26/2011_09_26_drive_0035_sync/image_00/data/0000000000.png");
    cv::Mat img2 = cv::imread("/home/rahul/CPP/2011_09_26/2011_09_26_drive_0035_sync/image_00/data/0000000001.png");
    std::string windowname = "Image";
    
    cv::Mat img_gray1, img_gray2;
    cv::cvtColor(img1, img_gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img_gray2, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> kp; 
    std::vector<cv::Point2f> points1, points2;

    detectfeatures(img_gray1, points1);

    std::vector<uchar> status;

    featureTracking(img_gray1,img_gray2,points1,points2, status);

    cv::Mat cameraMatrix = (cv::Mat1d(3,3) << 984.24, 0.0, 690.0, 0.0, 980.81, 233.1, 0.0, 0.0, 1.0 );
    
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

    for(int numFrame = 2; numFrame < 123; numFrame++)
    {
        std::sprintf(filename,"/home/rahul/CPP/2011_09_26/2011_09_26_drive_0035_sync/image_00/data/%08d.png", numFrame);
        cv::Mat currImage_c = cv::imread(filename);
        cv::cvtColor(currImage_c, currImage, cv::COLOR_BGR2GRAY);
        vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures);
        E = cv::findEssentialMat(currFeatures, prevFeatures, cameraMatrix, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t, mask);

        cv::Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);


        for(int i = 0; prevFeatures,size();i++)
        {
            prevPts.at<double>(0,i) = prevFeatures.at(i).x;
            prevPts.at<double>(1,i) = prevFeatures.at(i).y;
            
            currPts.at<double>(0,i) = currFeatures.at(i).x;
            currPts.at<double>(1,i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale()






    }
    
    // cv::Mat descriptors;
    // keypointdescriptors(kp, img_gray, descriptors);

    // matchdescriptors


    // cv::imshow(windowname, img_gray1);
    // cv::waitKey(0);
    return 0;

}
