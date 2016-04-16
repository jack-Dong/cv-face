#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <cassert>

#define POINT_MATCH_THRESH 0.3
#define FACE_MATCH_THRESH 0.5
#define FACE_VALID_POINT_COUNT 100

using namespace std;
using namespace cv;

int main()
{
    //加载图片
    string sy1 = "sy1.jpeg";
    string sy2 = "sy2.jpeg";
    string sm ="sm.jpeg";
    Mat src1 = imread(sy1);
    Mat src2 = imread(sm);
    assert(!src1.empty() && !src2.empty());

    //加载模型
    string  faceCascadeName = "haarcascade_frontalface_alt.xml";
    CascadeClassifier faceCascasde;
    assert(true == faceCascasde.load(faceCascadeName));

    //灰度化
    Mat gray1,gray2;
    cvtColor(src1,gray1,CV_BGR2GRAY);
    cvtColor(src2,gray2,CV_BGR2GRAY);
    vector<Rect> faces1,faces2;
    //开始检测人脸
    faceCascasde.detectMultiScale(gray1,faces1);
    faceCascasde.detectMultiScale(gray2,faces2);

//    cout << faces1.size() <<" "<< faces2.size()<<endl;
    Mat faceDetect1 = src1.clone();
    Mat faceDetect2 = src2.clone();
    assert(faces1.size() > 0 && faces2.size() > 0);
    rectangle(faceDetect1,faces1[0],Scalar(0,255,255));
    rectangle(faceDetect2,faces2[0],Scalar(0,255,255));

    imshow("src1",faceDetect1);
    imshow("src2",faceDetect2);

    //取出人脸
    Mat face1 = gray1(faces1[0]);
    Mat face2 = gray2(faces2[0]);

    imshow("face1",face1);
    imshow("face2",face2);

    //提取特征点
    vector<KeyPoint> keypoints1,keypoints2;
    SurfFeatureDetector surfFeatureDetector(2500);

    surfFeatureDetector.detect(face1,keypoints1);
    surfFeatureDetector.detect(face2,keypoints2);

    //计算描述符
    Mat descriptors1,descriptors2;
    SurfDescriptorExtractor surfDescriptorExtractor;
    surfDescriptorExtractor.compute(face1,keypoints1,descriptors1);
    surfDescriptorExtractor.compute(face2,keypoints2,descriptors2);

    //匹配描述符
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors1,descriptors2,matches);

    //匹配可视化
    Mat imgMatches;
    drawMatches(face1,keypoints1,face2,keypoints2,matches,imgMatches);

    imshow("匹配",imgMatches);

    //计算前一百个点匹配率
    int size = matches.size() > FACE_VALID_POINT_COUNT ? FACE_VALID_POINT_COUNT :matches.size();
    int count;
    for (int i = 0; i < size; ++i)
    {
        if(matches[i].distance > POINT_MATCH_THRESH)
        {
            count++;
        }
    }
    float ratio = static_cast<float >(count) / FACE_VALID_POINT_COUNT;
    if (ratio> FACE_MATCH_THRESH)
    {
        cout << "匹配成功"<<endl;

    } else
    {
        cout <<"匹配失败"<<endl;
    }
    cout <<"匹配率:"<<ratio<<endl;

    waitKey(0);
    return 0;
}