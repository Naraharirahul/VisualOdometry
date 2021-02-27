#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>



int main(int argc, const char *argv[])
{
    cv::Mat img1 = cv::imread("/home/rahul/CPP/2011_09_26/2011_09_26_drive_0035_sync/image_00/data/0000000000.png");
    cv::Mat img2 = cv::imread("/home/rahul/CPP/2011_09_26/2011_09_26_drive_0035_sync/image_00/data/0000000001.png");
    std::string windowname = "Image";
    cv::imshow(windowname, img2);
    cv::waitKey(0);
    return 0;

}
