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
    cv::Mat img = cv::imread("0000000000.png");
    std::cout << img.size() << std::endl;
    std::string windowname = "Image";
    cv::imshow(windowname, img);
    cv::waitKey(0);
    return 1;

}
