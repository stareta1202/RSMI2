#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include "utils/FileReader.h"
// #include "indices/ZM.h"
#include "indices/RSMI.h"
#include "utils/ExpRecorder.h"
#include "utils/Constants.h"
#include "utils/FileWriter.h"
#include "utils/util.h"
#include <torch/torch.h>

#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

// EDIT: Include cstdlib for hashing string
#include <cstdlib>
// END EDIT

using namespace std;

#ifndef use_gpu
// #define use_gpu

int ks[] = {1, 5, 25, 125, 625};
float areas[] = {0.000006, 0.000025, 0.0001, 0.0004, 0.0016};
float ratios[] = {0.25, 0.5, 1, 2, 4};
int Ns[] = {5000, 2500, 500};

int k_length = sizeof(ks) / sizeof(ks[0]);
int window_length = sizeof(areas) / sizeof(areas[0]);
int ratio_length = sizeof(ratios) / sizeof(ratios[0]);

int n_length = sizeof(Ns) / sizeof(Ns[0]);

int query_window_num = 1000;
int query_k_num = 1000;

long long cardinality = 10000;
long long inserted_num = cardinality / 10;
string distribution = Constants::DEFAULT_DISTRIBUTION;
int inserted_partition = 5;
int skewness = 1;

// EDIT: New function for hashing key
// hash the key of insert value
float hash_key(string s) {
    string res, tmp;
    const char *c = s.c_str();
    for (int i = 0; i < s.length(); i++) {
        tmp = to_string(c[i] - '0');
        res.append(tmp);
    }
    return stof(res);
}
// END EDIT

// EDIT: New function for parsing model name
string parse_title(char *arg) {
    vector<string> vec;
    string line = str(arg);

    boost::algorithm::split(vec, line, boost::is_any_of("./"));

    int vec_size = vec.size();

    return vec[vec_size - 2];
}
// END EDIT

double knn_diff(vector<Point> acc, vector<Point> pred)
{
    int num = 0;
    for (Point point : pred)
    {
        for (Point point1 : acc)
        {
            if (point.x == point1.x && point.y == point1.y)
            {
                num++;
            }
        }
    }
    return num * 1.0 / pred.size();
}

void exp_RSMI(FileWriter file_writer, ExpRecorder exp_recorder, vector<Point> points, map<string, vector<Mbr>> mbrs_map, vector<Point> query_poitns, vector<Point> insert_points, string model_path)
{
    exp_recorder.clean();
    exp_recorder.structure_name = "RSMI";
    RSMI::model_path_root = model_path;
    RSMI *partition = new RSMI(0,  Constants::MAX_WIDTH);
    auto start = chrono::high_resolution_clock::now();
    partition->model_path = model_path;
    partition->build(exp_recorder, points);
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    cout << "build time: " << exp_recorder.time << endl;
    exp_recorder.size = (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE + Constants::PAGESIZE + Constants::DIM * Constants::DIM) * Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();
    partition->point_query(exp_recorder, points);
    cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    cout << "finish point_query time: " << exp_recorder.time << endl;
    file_writer.write_point_query(exp_recorder);
    exp_recorder.clean();

    exp_recorder.window_size = areas[2];
    exp_recorder.window_ratio = ratios[2];
    partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    cout << "RSMI::acc_window_query time: " << exp_recorder.time << endl;
    cout << "RSMI::acc_window_query page_access: " << exp_recorder.page_access << endl;
    file_writer.write_acc_window_query(exp_recorder);
    partition->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    exp_recorder.accuracy = ((double)exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_qesult_size;
    cout << "window_query time: " << exp_recorder.time << endl;
    cout << "window_query page_access: " << exp_recorder.page_access << endl;
    cout<< "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    file_writer.write_window_query(exp_recorder);

    exp_recorder.clean();
    exp_recorder.k_num = ks[2];
    partition->acc_kNN_query(exp_recorder, query_poitns, ks[2]);
    cout << "exp_recorder.time: " << exp_recorder.time << endl;
    cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    file_writer.write_acc_kNN_query(exp_recorder);
    partition->kNN_query(exp_recorder, query_poitns, ks[2]);
    cout << "exp_recorder.time: " << exp_recorder.time << endl;
    cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    cout<< "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    file_writer.write_kNN_query(exp_recorder);
    exp_recorder.clean();

    partition->insert(exp_recorder, insert_points);
    cout << "exp_recorder.insert_time: " << exp_recorder.insert_time << endl;
    exp_recorder.clean();
    partition->point_query(exp_recorder, points);
    cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    cout << "finish point_query time: " << exp_recorder.time << endl;
    exp_recorder.clean();
}

string RSMI::model_path_root = "";
int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "Invalid arguments!" << endl;
        return -1;
    }

    ExpRecorder exp_recorder;

    double minx = 1000.0;
    double maxx = -1000.0;
    double miny = 1000.0;
    double maxy = -1000.0;
    double rangex, rangey;

    RSMI *partition = new RSMI(0, Constants::MAX_WIDTH);

    if (strcmp(argv[1], "sequential") != 0) {
        // TODO change filename
        string dataset_filename = Constants::DATASETS + exp_recorder.distribution + "_" + to_string(exp_recorder.dataset_cardinality) + "_" + to_string(exp_recorder.skewness) + "_2_.csv";
        // EDIT: Change dataset filename to "Tweet.csv"
        dataset_filename = str(argv[1]);
        // END EDIT
        FileReader filereader(dataset_filename, ",");
        // EDIT: Change function from get_points() to get_tweets()
        // vector<Point> points = filereader.get_points();
        
        vector<Point> points = filereader.get_tweets(&minx, &maxx, &miny, &maxy);
        rangex = maxx - minx;
        rangey = maxy - miny;

        for (int i = 0; i < points.size(); i++) {
            points[i].x = (points[i].x - minx) / rangex;
            points[i].y = (points[i].y - miny) / rangey;
        }
        // END EDIT
        string model_root_path = Constants::TORCH_MODELS + distribution + "_" + to_string(cardinality);
        // EDIT: Change model root path
        string model_title = parse_title(argv[1]);
        model_root_path = Constants::TORCH_MODELS + model_title;
        // END EDIT
        file_utils::check_dir(model_root_path);
        string model_path = model_root_path + "/";
        FileWriter file_writer(Constants::RECORDS);

        // EDITED FROM HERE
        exp_recorder.structure_name = "RSMI";
        RSMI::model_path_root = model_path;

        // build model
        auto start = chrono::high_resolution_clock::now();
        partition->model_path = model_path;
        partition->build(exp_recorder, points);
        auto finish = chrono::high_resolution_clock::now();
        exp_recorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        cout << "build time: " << exp_recorder.time << endl;

        exp_recorder.size = (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 + 1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num + (Constants::DIM * Constants::PAGESIZE + Constants::PAGESIZE + Constants::DIM * Constants::DIM) * Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;

        exp_recorder.clean();
    }
    else {
        string model_title = parse_title(argv[2]);

        string model_root_path = Constants::TORCH_MODELS + model_title;
        file_utils::check_dir(model_root_path);
        string model_path = model_root_path + "/";
        FileWriter file_writer(Constants::RECORDS);

        exp_recorder.structure_name = "RSMI";
        RSMI::model_path_root = model_path;

        partition->model_path = model_path;

        vector<Point> points;
        Point p(0.460476, 0.528138);
        points.push_back(p);
        partition->build(exp_recorder, points);

        exp_recorder.clean();
    }

    ifstream f;
    f.open(argv[2]);
    string l = "";
    long long elapsed = 0;
    long long range_elapsed = 0;
    long long count = 0;

    while (getline(f, l)) {
        vector<string> vec;
        cout << count++ << endl;
        
        boost::algorithm::split(vec, l, boost::is_any_of(", "));
        if (vec[0].compare("INSERT") == 0) {
            Point p((stod(vec[2]) - minx) / rangex, (stod(vec[3]) - miny) / rangey);
            vector<Point> pts;
            pts.push_back(p);
            cout << vec[1] << ")" << vec[2] << "," << vec[3] << endl;
            // if (partition->point_query(exp_recorder, pts)) {
            //     partition->remove(exp_recorder, pts);
            //     exp_recorder.clean();
            // }
            // partition->remove(exp_recorder, pts);
            // exp_recorder.clean();
            partition->insert(exp_recorder, pts);
            // if (count == 1962033) cout << "checkpoint 1" << endl;
            cout << "Insert Time: " << exp_recorder.insert_time << endl;
            elapsed += exp_recorder.insert_time;
            exp_recorder.clean();
        } else if (vec[0].compare("POINT") == 0) {
            Point p((stod(vec[1]) - minx) / rangex, (stod(vec[2]) - miny) / rangey);
            vector<Point> pts;
            pts.push_back(p);
            partition->point_query(exp_recorder, pts);
            cout << "Point Query Time: " << exp_recorder.time << endl;
            elapsed += exp_recorder.time;
            exp_recorder.clean();
        } else if (vec[0].compare("RANGE") == 0) {
            Mbr tmbr((stod(vec[1]) - minx) / rangex, (stod(vec[2]) - miny) / rangey, (stod(vec[3]) - minx) / rangex, (stod(vec[4]) - miny) / rangey);
            vector<Mbr> tmbrs;
            tmbrs.push_back(tmbr);
            partition->window_query(exp_recorder, tmbrs);
            cout << "Range Query Time: " << exp_recorder.time << "//" << vec[1] << "," << vec[2] << "," << vec[3] << "," << vec[4] << endl;
            elapsed += exp_recorder.time;
            range_elapsed += exp_recorder.time;
            exp_recorder.clean();
        } else if (vec[0].compare("KNN") == 0) {
            // KNN x y n
            Point p((stod(vec[1]) - minx) / rangex, (stod(vec[2]) - miny) / rangey);
            vector<Point> pts;
            pts.push_back(p);
            partition->kNN_query(exp_recorder, pts, stoi(vec[3]));
            cout << "KNN Query Time: " << exp_recorder.time << endl;
            elapsed += exp_recorder.time;
            exp_recorder.clean();
        } else if (vec[0].compare("SETPARAM") == 0) {
            minx = stod(vec[1]);
            miny = stod(vec[2]);
            maxx = stod(vec[3]);
            maxy = stod(vec[4]);
            rangex = maxx - minx;
            rangey = maxy - miny;
            exp_recorder.clean();
        } else if (vec[0].compare("DELETE") == 0) {
            Point p((stod(vec[1]) - minx) / rangex, (stod(vec[2]) - miny) / rangey);
            vector<Point> pts;
            pts.push_back(p);
            partition->remove(exp_recorder, pts);
            cout << "DELETE Query Time: " << exp_recorder.delete_time << endl;
            elapsed += exp_recorder.delete_time;
            exp_recorder.clean();
        }
    }

    f.close();

    cout << "Total Elapsed Time : " << elapsed << " nanoseconds" << endl;
    cout << "Total Range Query Time : " << range_elapsed << " nanoseconds" << endl;

}

#endif  // use_gpu