#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include<opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <QDir>
#include <QtDebug>
#include<iostream>

/**
 *  Система	поиска	измененных	дубликатов	изображений
 *  Автоматизированная	система	поиска	похожих	изображений	на	диске	компьютера
 *  Система	должна	искать	изображения,	полученные	после	преобразований
 *  (изменение	разрешения,	ретушь,	изменение	яркости/контрастности,	вращение)
 *  оригинального	и	предоставлять	их	в	виде	списка.
 */

using namespace cv;
using namespace std;
QLatin1String s1("."), s2("..");
//папка где ищем похожие
QString dirPath = "D:/labs/my/1/";
//изображение по которому ищем похожие
QString etalonPath = "D:/labs/my/1.jpg";

//Метод сравнения двух изображений
void compareImage(QString &nameImg1, QString &nameImg2){
    cout << "---Compare: " << nameImg1.toStdString()<<" and " <<nameImg2.toStdString() << endl;

    Mat img_object = imread( nameImg1.toStdString(), IMREAD_GRAYSCALE );
    Mat img_scene = imread( nameImg2.toStdString(), IMREAD_GRAYSCALE );

    //С помощью SURF определяю ключевые точки на образце и изображении.
    int minHessian = 500;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect( img_object,keypoints_object );
    detector.detect( img_scene, keypoints_scene );

    // Для найденных точек рассчитываю значения дескрипторов
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    //Сопоставляю точки на целевом объекте и в сцене, получаю так называемые матчи ( FLANN, класс FlannBasedMatcher, метод match).
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //находим максимальное и минимальную дистанцию
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist )
            min_dist = dist;
        if( dist > max_dist )
            max_dist = dist;
    }

    //сохраняем только матчи расстояние которых составляет менее 3 * Min_Dist
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
        { good_matches.push_back( matches[i]); }
    }

    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //Получаям ключевые точки из хороших матчей
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    //с помощью findHomography() получаем матрицу масок, содержащую инлайны
    std::vector <uchar> mask;
    findHomography(obj, scene, CV_RANSAC, 3, mask);

    //считаем количество где инлайн равен 1
    int inliers_count = 0;
    for( unsigned i = 0; i < mask.size(); i++ )
        if (mask[i] == 1)
            inliers_count++;

    //находим процент совпадения
    float s= (inliers_count*100)/mask.size();
    cout << "----Image: " << nameImg2.toStdString() << "  --percent: "<< s << endl;

}

void processFile(QString &path) {
    compareImage(etalonPath, path);
}

void processDir(QString &path) {
    cout << "\n-dir " << path.toStdString() << endl;

    QDir dir(path);
    auto list = dir.entryList(QDir::Files);
    for (int i = 0; i < list.size(); i++) {
        QString filePath = path + "/" + list[i];
        processFile(filePath);
    }
    list = dir.entryList(QDir::Dirs);
    for (int i = 0; i < list.size(); i++) {
        if (list[i] == s1 || list[i] == s2)
            continue;
        QString np = path + "/" + list[i];
        processDir(np);
    }
}



int main()
{
    cout << "Etalon image: " << etalonPath.toStdString() << endl;
    processDir(dirPath);
    cout << "The End!" <<endl;
    return 0;
}
