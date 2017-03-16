#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"  //SURF
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;






void getMatchPoint(Mat src1, Mat src2)

{

	vector<KeyPoint> keys1;
	vector<KeyPoint> keys2;

	Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(800);
	cv::BFMatcher matcher;
	Mat descriptorMat1, descriptorMat2;
	std::vector<DMatch> mathces;

	detector->detectAndCompute(src1, Mat(), keys1, descriptorMat1);
	detector->detectAndCompute(src2, Mat(), keys2, descriptorMat2);
	matcher.match(descriptorMat1, descriptorMat2, mathces);

	drawKeypoints(src1, keys1, src1, cv::Scalar::all(255),
			cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(src2, keys2, src2, cv::Scalar::all(255),
			cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	Mat matchMat;
	drawMatches(src1, keys1, src2, keys2, mathces, matchMat);
	cv::imshow("Mathces", matchMat);

	imshow("image1", src1);
	imshow("image2", src2);

#if 1
	double max_dist = 0;
	double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptorMat1.rows; i++) {
		double dist = mathces[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}
	cout << "-- Max dist :" << max_dist << endl;
	cout << "-- Min dist :" << min_dist << endl;

	//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )
	//-- PS.- radiusMatch can also be used here.
	vector<DMatch> good_matches;
	for (int i = 0; i < descriptorMat1.rows; i++) {
		if (mathces[i].distance < 0.6 * max_dist) {
			good_matches.push_back(mathces[i]);
		}
	}

	Mat img_matches;
	drawMatches(src1, keys1, src2, keys2, good_matches, img_matches,
			Scalar::all(-1), Scalar::all(-1), vector<char>(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("FASTResult.jpg", img_matches);
	imshow("goodMatch", img_matches);
#endif

#if 1
	// 分配空间
	int ptCount = (int) mathces.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);

	// 把Keypoint转换为Mat
	Point2f pt;
	for (int i = 0; i < ptCount; i++) {
		pt = keys1[mathces[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keys2[mathces[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// 用RANSAC方法计算 基本矩阵F
	Mat fundamental;
	vector<uchar> RANSACStatus;

	fundamental = findFundamentalMat(p1, p2, RANSACStatus, FM_RANSAC);
	// 计算野点个数
	int OutlinerCount = 0;
	for (int i = 0; i < ptCount; i++) {
		if (RANSACStatus[i] == 0) // 状态为0表示野点
				{
			OutlinerCount++;
		}
	}

	// 计算内点
	vector<Point2f> Inlier1;
	vector<Point2f> Inlier2;
	vector<DMatch> InlierMatches;
	// 上面三个变量用于保存内点和匹配关系
	int InlinerCount = ptCount - OutlinerCount;
	InlierMatches.resize(InlinerCount);
	Inlier1.resize(InlinerCount);
	Inlier2.resize(InlinerCount);
	InlinerCount = 0;
	for (int i = 0; i < ptCount; i++) {
		if (RANSACStatus[i] != 0) {
			Inlier1[InlinerCount].x = p1.at<float>(i, 0);
			Inlier1[InlinerCount].y = p1.at<float>(i, 1);
			Inlier2[InlinerCount].x = p2.at<float>(i, 0);
			Inlier2[InlinerCount].y = p2.at<float>(i, 1);
			InlierMatches[InlinerCount].queryIdx = InlinerCount;
			InlierMatches[InlinerCount].trainIdx = InlinerCount;
			cout << "index = " << i << ", distance=" << mathces.at(i).distance
					<< endl;
			InlinerCount++;
		}
	}

	// 把内点转换为drawMatches可以使用的格式
	vector<KeyPoint> key1(InlinerCount);
	vector<KeyPoint> key2(InlinerCount);
	KeyPoint::convert(Inlier1, key1);
	KeyPoint::convert(Inlier2, key2);

	// 显示计算F过后的内点匹配
	//Mat m_matLeftImage;
	//Mat m_matRightImage;
	// 以上两个变量保存的是左右两幅图像
	Mat OutImage;
	drawMatches(src1, key1, src2, key2, InlierMatches, OutImage);
	imwrite("FmatrixResult.jpg", OutImage);
	imshow("FMatch", OutImage);
#endif

}

int main( int argc, char** argv )
{
	Mat img1=imread(argv[1]);
	Mat img2=imread(argv[2]);
	getMatchPoint(img1,img2);
	waitKey(0);

}



