#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>


using namespace std;
using namespace cv;


struct trainingDataEr {
    int number = {};
    vector<bool> data;
};

// 转01矩阵
bool toBinaryMat(const string &path, vector<bool> &binaryMatrix) {
    binaryMatrix.resize(0);
    Mat image = imread(path, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "无法加载图片" << endl;
        return false;
    }
    Mat binaryMat(image.size(), CV_8UC1);

    // 二值化处理
    threshold(image, binaryMat, 128, 255, THRESH_BINARY);
    for (int i = 0; i < binaryMat.rows; ++i) {
        for (int j = 0; j < binaryMat.cols; ++j) {
            auto pixel = binaryMat.at<uchar>(i, j);
            binaryMatrix.push_back(pixel > 0);
        }
    }
    return true;
}

struct pointDis {
    int number = {};
    float dis = {};
};


// 加载训练数据
bool loadingTraining(string &trainingPath, vector<trainingDataEr> &trainData) {
    trainData.resize(0);

    filesystem::path path = trainingPath;
    // 检查路径是否存在
    if (!filesystem::exists(path)) {
        cout << "目录不存在" << endl;
        return false;
    }

    // 遍历目录下文件
    filesystem::directory_iterator iterator(path);
    for (auto &x: iterator) {
        if (!filesystem::is_directory(x.path())) {
            continue;
        }
        filesystem::directory_iterator numSet(x.path());
        for (auto &y: numSet) {

            vector<bool> bMat;
            toBinaryMat(y.path().string(), bMat);
            trainData.push_back({stoi(x.path().filename().string()), bMat});
        }

    }
    return true;
}


int knn(const vector<bool> &bMat, const vector<trainingDataEr> &trainData, int k) {
    vector<pointDis> dis;
    vector<vector<bool>> resMat(trainData.size());
    for (auto &i: resMat) {
        i = bMat;
    }

    vector<vector<float>> diffMat(resMat.size());
    for (auto &i: diffMat) {
        i.resize(resMat[0].size());
    }
    for (int y = 0; y < resMat.size(); ++y) {
        float ans = 0;
        for (int x = 0; x < resMat[y].size(); ++x) {
            diffMat[y][x] = (float) pow(trainData[y].data[x] - resMat[y][x], 2);
            ans += diffMat[y][x];
        }
        dis.push_back({trainData[y].number, (float) sqrt(ans)});
    }

    sort(dis.begin(), dis.end(), [](pointDis a, pointDis b) { return a.dis < b.dis; });

    // 投票
    vector<int> vote(100,0);
    for (int i = 0; i < k; ++i) {
        vote[dis[i].number]++;
    }

    int res = (int) (max_element(vote.begin(), vote.end()) - vote.begin());
    return res;
}


// 函数用于生成唯一文件名
filesystem::path generateUniqueFilename(const filesystem::path& dir, const std::string& filename) {
    filesystem::path newFilename = dir / filename;
    int counter = 0;
    while (filesystem::exists(newFilename)) {
        newFilename = dir / (filename + std::to_string(counter) + ".jpg");
        counter++;
    }
    return newFilename;
}





int main() {
    // k值
    int k = 5;


    string n = "4";
    // 自动移动照片到测试文件夹开关， 必须填好对应数字
    bool onOff = false;

    vector<bool> bMat;
    vector<trainingDataEr> trainData;

    // 路径
    string testPath = R"(..\data\test\)" + n +".jpg";
    string trainingPath = R"(..\data\xl)";


    //加载测试图片和训练集
    if (!loadingTraining(trainingPath, trainData) || !toBinaryMat(testPath, bMat)) {
        return 1;
    }


    filesystem::path pathObj(testPath);
    string pictureName = pathObj.filename().string();

    cout << "测试图片:" << pictureName << endl;
    cout << "测试结果：" << knn(bMat, trainData, k) << endl;


    // 移动图片
    if (onOff) {
        if (n >= "0" && n <= "9") {
            trainingPath = trainingPath + "/" + n;

            filesystem::path p1(testPath);
            filesystem::path p2(trainingPath);

            
            // 移动文件
            
            // 生成唯一文件名
            filesystem::path uniqueFilename = generateUniqueFilename(p2, p1.filename().string());
            // 移动文件
            filesystem::rename(testPath, uniqueFilename);
//            filesystem::rename(p1, p2 / filesystem::path(  p1.filename()));
        }
    }
    return 0;
}

