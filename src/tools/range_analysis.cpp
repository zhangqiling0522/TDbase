#include "himesh.h"
#include "tile.h"
#include "util.h"
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace tdbase;

struct RangeStats {
    int lod;
    int object_count;
    float min_range_min;
    float min_range_max;
    float max_range_min;
    float max_range_max;
    float avg_range_min;
    float avg_range_max;
};

void analyze_ranges(const char *data_file, const char *output_file) {
    // Load tile
    Tile *tile = new Tile(data_file);
    int num_objects = tile->num_objects();

    vector<RangeStats> stats_list;

    // Decode from LOD 10 to 100 in steps of 10
    for (int lod = 10; lod <= 100; lod += 10) {
        cout << "Processing LOD: " << lod << "%" << endl;

        tile->decode_all(lod);

        vector<float> range_mins;
        vector<float> range_maxs;

        // For each object, calculate range statistics
        for (int i = 0; i < num_objects; i++) {
            HiMesh *mesh = tile->get_mesh(i);
            if (!mesh) continue;

            // Get bounding box as a simple range metric
            aab bbox = mesh->get_mbb();

            // Calculate range span (max - min for each dimension)
            float x_range = bbox.maxx - bbox.minx;
            float y_range = bbox.maxy - bbox.miny;
            float z_range = bbox.maxz - bbox.minz;

            // Use the maximum range as the primary metric
            float range_span = max({x_range, y_range, z_range});
            range_maxs.push_back(range_span);

            // Also collect min ranges (could be min dimension)
            float min_dim = min({x_range, y_range, z_range});
            range_mins.push_back(min_dim);
        }

        if (range_maxs.empty()) continue;

        // Calculate statistics
        RangeStats stat;
        stat.lod = lod;
        stat.object_count = num_objects;

        // Min/Max of ranges
        stat.min_range_min = *min_element(range_mins.begin(), range_mins.end());
        stat.min_range_max = *max_element(range_mins.begin(), range_mins.end());
        stat.max_range_min = *min_element(range_maxs.begin(), range_maxs.end());
        stat.max_range_max = *max_element(range_maxs.begin(), range_maxs.end());

        // Average ranges
        stat.avg_range_min = accumulate(range_mins.begin(), range_mins.end(), 0.0f) / range_mins.size();
        stat.avg_range_max = accumulate(range_maxs.begin(), range_maxs.end(), 0.0f) / range_maxs.size();

        stats_list.push_back(stat);
    }

    // Write to CSV
    ofstream csv_file(output_file);
    if (!csv_file.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        delete tile;
        return;
    }

    // Write header
    csv_file << "LOD,ObjectCount,MinRangeMin,MinRangeMax,MaxRangeMin,MaxRangeMax,AvgRangeMin,AvgRangeMax\n";

    // Write data
    csv_file << fixed << setprecision(6);
    for (const auto &stat : stats_list) {
        csv_file << stat.lod << ","
                 << stat.object_count << ","
                 << stat.min_range_min << ","
                 << stat.min_range_max << ","
                 << stat.max_range_min << ","
                 << stat.max_range_max << ","
                 << stat.avg_range_min << ","
                 << stat.avg_range_max << "\n";
    }

    csv_file.close();

    cout << "Range analysis completed. Results saved to: " << output_file << endl;

    delete tile;
}

// Handle hausdorff-based range analysis
void analyze_hausdorff_ranges(const char *data_file, const char *output_file) {
    Tile *tile = new Tile(data_file);
    int num_objects = tile->num_objects();

    vector<RangeStats> stats_list;

    // Decode from LOD 10 to 100 in steps of 10
    for (int lod = 10; lod <= 100; lod += 10) {
        cout << "Processing LOD: " << lod << "%" << endl;

        tile->decode_all(lod);

        vector<float> hausdorff_values;
        vector<float> proxy_hausdorff_values;

        // For each object, collect hausdorff distances
        for (int i = 0; i < num_objects; i++) {
            HiMesh *mesh = tile->get_mesh(i);
            if (!mesh) continue;

            float hd = mesh->getHausdorffDistance();
            float phd = mesh->getProxyHausdorffDistance();

            if (hd >= 0) hausdorff_values.push_back(hd);
            if (phd >= 0) proxy_hausdorff_values.push_back(phd);
        }

        if (hausdorff_values.empty()) continue;

        // Calculate statistics
        RangeStats stat;
        stat.lod = lod;
        stat.object_count = num_objects;

        // Hausdorff distance ranges
        stat.min_range_min = *min_element(hausdorff_values.begin(), hausdorff_values.end());
        stat.min_range_max = *max_element(hausdorff_values.begin(), hausdorff_values.end());

        if (!proxy_hausdorff_values.empty()) {
            stat.max_range_min = *min_element(proxy_hausdorff_values.begin(), proxy_hausdorff_values.end());
            stat.max_range_max = *max_element(proxy_hausdorff_values.begin(), proxy_hausdorff_values.end());
        }

        // Average distances
        stat.avg_range_min = accumulate(hausdorff_values.begin(), hausdorff_values.end(), 0.0f) / hausdorff_values.size();
        if (!proxy_hausdorff_values.empty()) {
            stat.avg_range_max = accumulate(proxy_hausdorff_values.begin(), proxy_hausdorff_values.end(), 0.0f) / proxy_hausdorff_values.size();
        }

        stats_list.push_back(stat);
    }

    // Write to CSV
    ofstream csv_file(output_file);
    if (!csv_file.is_open()) {
        cerr << "Failed to open output file: " << output_file << endl;
        delete tile;
        return;
    }

    // Write header
    csv_file << "LOD,ObjectCount,HausdorffMin,HausdorffMax,ProxyHausdorffMin,ProxyHausdorffMax,AvgHausdorff,AvgProxyHausdorff\n";

    // Write data
    csv_file << fixed << setprecision(6);
    for (const auto &stat : stats_list) {
        csv_file << stat.lod << ","
                 << stat.object_count << ","
                 << stat.min_range_min << ","
                 << stat.min_range_max << ","
                 << stat.max_range_min << ","
                 << stat.max_range_max << ","
                 << stat.avg_range_min << ","
                 << stat.avg_range_max << "\n";
    }

    csv_file.close();

    cout << "Hausdorff-based range analysis completed. Results saved to: " << output_file << endl;

    delete tile;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: range_analysis <data_file> [output_file] [mode]\n";
        cerr << "  mode: 0 = spatial ranges (default), 1 = hausdorff distances\n";
        return 1;
    }

    const char *data_file = argv[1];
    const char *output_file = (argc > 2) ? argv[2] : "range_analysis_result.csv";
    int mode = (argc > 3) ? atoi(argv[3]) : 0;

    cout << "Analyzing LOD Range changes from: " << data_file << endl;
    cout << "Output file: " << output_file << endl;

    if (mode == 1) {
        analyze_hausdorff_ranges(data_file, output_file);
    } else {
        analyze_ranges(data_file, output_file);
    }

    return 0;
}
