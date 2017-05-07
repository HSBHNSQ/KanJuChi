//
//  ViewController.m
//  去锯齿
//
//  Created by Mac_H on 2017/5/6.
//  Copyright © 2017年 何少博. All rights reserved.
//
#import "ViewController.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;



@interface ViewController ()
@property (weak, nonatomic) IBOutlet UIImageView *oriImageView;
@property (weak, nonatomic) IBOutlet UIImageView *resultImageView;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
        UIImage * image = [UIImage imageNamed:@"juchi.jpg"];
    NSData * data = UIImageJPEGRepresentation(image, 1);
    NSString * path = [NSTemporaryDirectory() stringByAppendingPathComponent:@"ceshi.jpg"];
    [data writeToFile:path atomically:YES];
    self.oriImageView.image = image;
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


-(void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event{
    
//    UIImage * resultImage =  [self ceshi2:self.oriImageView.image];
    UIImage * resultImage =  [self ceshi4:self.oriImageView.image];
    self.resultImageView.image = resultImage;
}


void ceshi()
{
    cv::namedWindow("result");
    Mat img=imread("TestImg.png");
    Mat whole_image=imread("D:\\ImagesForTest\\lena.jpg");
    whole_image.convertTo(whole_image,CV_32FC3,1.0/255.0);
    cv::resize(whole_image,whole_image,img.size());
    img.convertTo(img,CV_32FC3,1.0/255.0);
    
    Mat bg=Mat(img.size(),CV_32FC3);
    bg=Scalar(1.0,1.0,1.0);
    
    // Prepare mask
    Mat mask;
    Mat img_gray;
    cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);
    img_gray.convertTo(mask,CV_32FC1);
    threshold(1.0-mask,mask,0.9,1.0,cv::THRESH_BINARY_INV);
    
    
    
    cv::GaussianBlur(mask,mask,cv::Size(21,21),11.0);
    imshow("result",mask);
    cv::waitKey(0);
    
    
    // Reget the image fragment with smoothed mask
    Mat res;
    
    vector<Mat> ch_img(3);
    vector<Mat> ch_bg(3);
    cv::split(whole_image,ch_img);
    cv::split(bg,ch_bg);
    ch_img[0]=ch_img[0].mul(mask)+ch_bg[0].mul(1.0-mask);
    ch_img[1]=ch_img[1].mul(mask)+ch_bg[1].mul(1.0-mask);
    ch_img[2]=ch_img[2].mul(mask)+ch_bg[2].mul(1.0-mask);
    cv::merge(ch_img,res);
    cv::merge(ch_bg,bg);
    
    imshow("result",res);
    cv::waitKey(0);
    cv::destroyAllWindows();
}


-(UIImage *)ceshi2:(UIImage *)image;
{
//    UIImage * image = [UIImage imageNamed:@"juchi.jpg"];
//    Mat im = imread("dsds", 0);
    Mat im = [self cvMatFromUIImage:image];
//    NSString * path = [NSTemporaryDirectory() stringByAppendingPathComponent:@"juchi.jpg"];
//    
//     Mat im = imread([path UTF8String]);
    Mat cont = ~im;
    Mat original = Mat::zeros(im.rows, im.cols, CV_8UC3);
    Mat smoothed = Mat(im.rows, im.cols, CV_8UC3, Scalar(255,255,255));
//    您可以更改以下参数以获得不同的结果。
//    // contour smoothing parameters for gaussian filter
//    int filterRadius = 10; // you can try to change this value
//    int filterSize = 2 * filterRadius + 1;
//    double sigma = 20; // you can try to change this value
    // contour smoothing parameters for gaussian filter
    int filterRadius = 50;
    int filterSize = 2 * filterRadius + 1;
    double sigma = 80;
    
    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;
    // find contours and store all contour points
//    CV_RETR_EXTERNAL=0,
//    CV_RETR_LIST=1,
//    CV_RETR_CCOMP=2,
//    CV_RETR_TREE=3,
//    CV_RETR_FLOODFILL=4
    findContours(cont, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE,cv:: Point(0, 0));
    for(size_t j = 0; j < contours.size(); j++)
    {
        // extract x and y coordinates of points. we'll consider these as 1-D signals
        // add circular padding to 1-D signals
        size_t len = contours[j].size() + 2 * filterRadius;
        size_t idx = (contours[j].size() - filterRadius);
        vector<float> x, y;
        for (size_t i = 0; i < len; i++)
        {
            x.push_back(contours[j][(idx + i) % contours[j].size()].x);
            y.push_back(contours[j][(idx + i) % contours[j].size()].y);
        }
        // filter 1-D signals
        vector<float> xFilt, yFilt;
        GaussianBlur(x, xFilt, cv::Size(filterSize, filterSize), sigma, sigma);
        GaussianBlur(y, yFilt, cv::Size(filterSize, filterSize), sigma, sigma);
        // build smoothed contour
        vector<vector<cv::Point> > smoothContours;
        vector<cv::Point> smooth;
        for (size_t i = filterRadius; i < contours[j].size() + filterRadius; i++)
        {
            smooth.push_back(cv::Point(xFilt[i], yFilt[i]));
        }
        smoothContours.push_back(smooth);
        
        Scalar color;
        
        if(hierarchy[j][3] < 0 )
        {
            color = Scalar(0,0,0);
        }
        else
        {
            color = Scalar(255,255,255);
        }
        drawContours(smoothed, smoothContours, 0, color, -1);
    }
    UIImage * newImage = [self UIImageFromCVMat:smoothed];
    return newImage;
//    imshow( "result", smoothed );
//    waitKey(0);
}

-(UIImage *)ceshi3:(UIImage *)image{
//    Mat im = imread("4.png", 0);
    Mat im = [self cvMatFromUIImage:image];
    
    Mat cont = im.clone();
    Mat original = Mat::zeros(im.rows, im.cols, CV_8UC3);
    Mat smoothed = Mat::zeros(im.rows, im.cols, CV_8UC3);
    
    // contour smoothing parameters for gaussian filter
    int filterRadius = 5;
    int filterSize = 2 * filterRadius + 1;
    double sigma = 10;
    
    vector<vector<cv::Point> > contours;
    vector<Vec4i> hierarchy;
    // find external contours and store all contour points
    findContours(cont, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE,cv:: Point(0, 0));
    for(size_t j = 0; j < contours.size(); j++)
    {
        // draw the initial contour shape
        drawContours(original, contours, j, Scalar(0, 255, 0), 1);
        // extract x and y coordinates of points. we'll consider these as 1-D signals
        // add circular padding to 1-D signals
        size_t len = contours[j].size() + 2 * filterRadius;
        size_t idx = (contours[j].size() - filterRadius);
        vector<float> x, y;
        for (size_t i = 0; i < len; i++)
        {
            x.push_back(contours[j][(idx + i) % contours[j].size()].x);
            y.push_back(contours[j][(idx + i) % contours[j].size()].y);
        }
        // filter 1-D signals
        vector<float> xFilt, yFilt;
        GaussianBlur(x, xFilt, cv::Size(filterSize, filterSize), sigma, sigma);
        GaussianBlur(y, yFilt, cv::Size(filterSize, filterSize), sigma, sigma);
        // build smoothed contour
        vector<vector<cv::Point> > smoothContours;
        vector<cv::Point> smooth;
        for (size_t i = filterRadius; i < contours[j].size() + filterRadius; i++)
        {
            smooth.push_back(cv::Point(xFilt[i], yFilt[i]));
        }
        smoothContours.push_back(smooth);
        
        drawContours(smoothed, smoothContours, 0, Scalar(255, 0, 0), 1);
        
        cout << "debug contour " << j << " : " << contours[j].size() << ", " << smooth.size() << endl;
    }
    UIImage * newImage = [self UIImageFromCVMat:smoothed];
    return newImage;
}
-(UIImage *)ceshi4:(UIImage *)image{
        NSString * path = [NSTemporaryDirectory() stringByAppendingPathComponent:@"ceshi.jpg"];
    
         Mat vesselImage = imread([path UTF8String]);
    //
//  cv::Mat vesselImage = cv::imread("sds"); //the original image
//    Mat vesselImage = [self cvMatFromUIImage:image];
    cv::threshold(vesselImage, vesselImage, 125, 255, THRESH_BINARY);
    cv::Mat blurredImage; //output of the algorithm
    cv::pyrUp(vesselImage, blurredImage);
    
    for (int i = 0; i < 15; i++)
        cv::medianBlur(blurredImage, blurredImage, 7);
    
    cv::pyrDown(blurredImage, blurredImage);
    cv::threshold(blurredImage, blurredImage, 200, 255, THRESH_BINARY);
    return  [self UIImageFromCVMat:blurredImage];
}
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
//     cv::Mat cvMat(rows, cols, CV_32SC1);
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}



@end
