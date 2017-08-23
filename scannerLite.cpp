/*
 * File: scannerLite.cpp
 * Author: daisygao
 * An OpenCV program implementing the recognition feature of the app "CamScanner".
 * It extracts the main document object from an image and adjust it to A4 size.
 */
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <iterator>

using namespace boost;
namespace po = boost::program_options;
using namespace cv;
using namespace std;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v) {
        copy(v.begin(), v.end(), ostream_iterator<T>(os, " "));
        return os;
}
/**
 * Get edges of an image
 * @param gray - grayscale input image
 * @param canny - output edge image
 */
void getCanny(Mat gray, Mat &canny) {
        Mat thres;
        double high_thres = threshold(gray, thres, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU), low_thres = high_thres * 0.5;
        cv::Canny(gray, canny, low_thres, high_thres);
}

struct Line {
        Point _p1;
        Point _p2;
        Point _center;

        Line(Point p1, Point p2) {
                _p1 = p1;
                _p2 = p2;
                _center = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        }
};

bool cmp_y(const Line &p1, const Line &p2) {
        return p1._center.y < p2._center.y;
}

bool cmp_x(const Line &p1, const Line &p2) {
        return p1._center.x < p2._center.x;
}

/**
 * Compute intersect point of two lines l1 and l2
 * @param l1
 * @param l2
 * @return Intersect Point
 */
Point2f computeIntersect(Line l1, Line l2) {
        int x1 = l1._p1.x, x2 = l1._p2.x, y1 = l1._p1.y, y2 = l1._p2.y;
        int x3 = l2._p1.x, x4 = l2._p2.x, y3 = l2._p1.y, y4 = l2._p2.y;
        if (float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) {
                Point2f pt;
                pt.x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
                pt.y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;
                return pt;
        }
        return Point2f(-1, -1);
}


// https://stackoverflow.com/questions/32848301/image-quality-improvement-in-opencv
// not yet used
void imadjust(const Mat1b& src, Mat1b& dst, int tol = 1, Vec2i in = Vec2i(0, 255), Vec2i out = Vec2i(0, 255))
{
        // src : input CV_8UC1 image
        // dst : output CV_8UC1 imge
        // tol : tolerance, from 0 to 100.
        // in  : src image bounds
        // out : dst image bounds

        dst = src.clone();

        tol = max(0, min(100, tol));

        if (tol > 0)
        {
                // Compute in and out limits

                // Histogram
                vector<int> hist(256, 0);
                for (int r = 0; r < src.rows; ++r) {
                        for (int c = 0; c < src.cols; ++c) {
                                hist[src(r, c)]++;
                        }
                }

                // Cumulative histogram
                vector<int> cum = hist;
                for (int i = 1; i < hist.size(); ++i) {
                        cum[i] = cum[i - 1] + hist[i];
                }

                // Compute bounds
                int total = src.rows * src.cols;
                int low_bound = total * tol / 100;
                int upp_bound = total * (100 - tol) / 100;
                in[0] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), low_bound));
                in[1] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), upp_bound));

        }

        // Stretching
        float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
        for (int r = 0; r < dst.rows; ++r)
        {
                for (int c = 0; c < dst.cols; ++c)
                {
                        int vs = max(src(r, c) - in[0], 0);
                        int vd = min(int(vs * scale + 0.5f) + out[0], out[1]);
                        dst(r, c) = saturate_cast<uchar>(vd);
                }
        }
}


void scan(String file, String out, bool debug = true, int ppi = 200, bool adjust = false) {

        /* get input image */
        Mat img = imread(file);
        // resize input image to img_proc to reduce computation
        Mat img_proc;
        int w = img.size().width, h = img.size().height, min_w = 200;
        double scale = min(10.0, w * 1.0 / min_w);
        int w_proc = w * 1.0 / scale, h_proc = h * 1.0 / scale;
        resize(img, img_proc, Size(w_proc, h_proc));
        Mat img_dis = img_proc.clone();

        /* get four outline edges of the document */
        // get edges of the image
        Mat gray, canny;
        cvtColor(img_proc, gray, CV_BGR2GRAY);
        getCanny(gray, canny);

        // extract lines from the edge image
        vector<Vec4i> lines;
        vector<Line> horizontals, verticals;
        HoughLinesP(canny, lines, 1, CV_PI / 180, w_proc / 3, w_proc / 3, 20);
        for (size_t i = 0; i < lines.size(); i++) {
                Vec4i v = lines[i];
                double delta_x = v[0] - v[2], delta_y = v[1] - v[3];
                Line l(Point(v[0], v[1]), Point(v[2], v[3]));
                // get horizontal lines and vertical lines respectively
                if (fabs(delta_x) > fabs(delta_y)) {
                        horizontals.push_back(l);
                } else {
                        verticals.push_back(l);
                }
                // for visualization only
                if (debug)
                        line(img_proc, Point(v[0], v[1]), Point(v[2], v[3]), Scalar(0, 0, 255), 1, CV_AA);
        }

        // edge cases when not enough lines are detected
        if (horizontals.size() < 2) {
                if (horizontals.size() == 0 || horizontals[0]._center.y > h_proc / 2) {
                        horizontals.push_back(Line(Point(0, 0), Point(w_proc - 1, 0)));
                }
                if (horizontals.size() == 0 || horizontals[0]._center.y <= h_proc / 2) {
                        horizontals.push_back(Line(Point(0, h_proc - 1), Point(w_proc - 1, h_proc - 1)));
                }
        }
        if (verticals.size() < 2) {
                if (verticals.size() == 0 || verticals[0]._center.x > w_proc / 2) {
                        verticals.push_back(Line(Point(0, 0), Point(0, h_proc - 1)));
                }
                if (verticals.size() == 0 || verticals[0]._center.x <= w_proc / 2) {
                        verticals.push_back(Line(Point(w_proc - 1, 0), Point(w_proc - 1, h_proc - 1)));
                }
        }
        // sort lines according to their center point
        sort(horizontals.begin(), horizontals.end(), cmp_y);
        sort(verticals.begin(), verticals.end(), cmp_x);
        // for visualization only
        if (debug) {
                line(img_proc, horizontals[0]._p1, horizontals[0]._p2, Scalar(0, 255, 0), 2, CV_AA);
                line(img_proc, horizontals[horizontals.size() - 1]._p1, horizontals[horizontals.size() - 1]._p2, Scalar(0, 255, 0), 2, CV_AA);
                line(img_proc, verticals[0]._p1, verticals[0]._p2, Scalar(255, 0, 0), 2, CV_AA);
                line(img_proc, verticals[verticals.size() - 1]._p1, verticals[verticals.size() - 1]._p2, Scalar(255, 0, 0), 2, CV_AA);
        }

        /* perspective transformation */

        // define the destination image size: A4 - 200 PPI
        int w_a4 = ppi*8.27, h_a4 = ppi*11.7;
        //int w_a4 = 595, h_a4 = 842;
        Mat dst = Mat::zeros(h_a4, w_a4, CV_8UC3);

        // corners of destination image with the sequence [tl, tr, bl, br]
        vector<Point2f> dst_pts, img_pts;
        dst_pts.push_back(Point(0, 0));
        dst_pts.push_back(Point(w_a4 - 1, 0));
        dst_pts.push_back(Point(0, h_a4 - 1));
        dst_pts.push_back(Point(w_a4 - 1, h_a4 - 1));

        // corners of source image with the sequence [tl, tr, bl, br]
        img_pts.push_back(computeIntersect(horizontals[0], verticals[0]));
        img_pts.push_back(computeIntersect(horizontals[0], verticals[verticals.size() - 1]));
        img_pts.push_back(computeIntersect(horizontals[horizontals.size() - 1], verticals[0]));
        img_pts.push_back(computeIntersect(horizontals[horizontals.size() - 1], verticals[verticals.size() - 1]));

        // convert to original image scale
        for (size_t i = 0; i < img_pts.size(); i++) {
                // for visualization only
                if (debug) {
                        circle(img_proc, img_pts[i], 10, Scalar(255, 255, 0), 3);
                }
                img_pts[i].x *= scale;
                img_pts[i].y *= scale;
        }

        // get transformation matrix
        Mat transmtx = getPerspectiveTransform(img_pts, dst_pts);

        // apply perspective transformation
        warpPerspective(img, dst, transmtx, dst.size());

        // adjust if necessary
        if (adjust) {
                vector<Mat1b> planes;
                split(dst, planes);
                for (int i = 0; i < 3; ++i)
                {
                        imadjust(planes[i], planes[i]);
                }

                Mat result;
                merge(planes, result);
                imwrite(out, result);

        } else {

                // save dst img
                imwrite(out, dst);

        }

        // for visualization only
        if (debug) {
                namedWindow("dst", CV_WINDOW_KEEPRATIO);
                imshow("src", img_dis);
                imshow("canny", canny);
                imshow("img_proc", img_proc);
                imshow("dst", dst);
                waitKey(0);
        }
}


int main(int argc, char** argv) {
        po::options_description desc("Allowed options");
        desc.add_options()
                ("help,h", "help message")
                ("adjust,a", "adjust image")
                ("verbose,v", po::value<int>()->implicit_value(1)->default_value(0),
                "enable verbosity (optionally specify level)")
                ("image,i", po::value< vector<string> >(),
                "input image")
                ("output,o", po::value<string>()->default_value("output"),
                "output folder")
                ("ppi,p", po::value<int>()->default_value(200),
                "pixel per inch")
        ;

        po::positional_options_description p;
        p.add("image,i", -1);

        po::variables_map args;
        try {
                po::store(
                        po::command_line_parser(argc, argv).options(desc).positional(p).run(),
                        args
                        );
        }
        catch (po::error const& e) {
                std::cerr << e.what() << '\n';
                exit( EXIT_FAILURE );
        }
        po::notify(args);

        if (args.count("help")) {
                cout << desc << "\n";
                return 1;
        }
        //if (args.count("verbose")) {
        //        cout << "Verbosity enabled.  Level is " << args["verbose"].as<int>()
        //             << "\n";
        //}
        if (args.count("image")) {
                BOOST_FOREACH( string file, args["image"].as< vector<string> >() ) {
                        cout << "Processing: " << file << "\n";
                        scan(file,
                             args["output"].as<string>() + "/" + file,
                             args["verbose"].as<int>() > 0,
                             args["ppi"].as<int>(),
                             args.count("adjust")
                             );
                }
                exit( EXIT_SUCCESS );
        }
        return 0;
}
