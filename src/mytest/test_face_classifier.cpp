#include "face_classifier.h"
#include "tile.h"
#include <getopt.h>
#include <unordered_set>

using namespace tdbase;

namespace tdbase {
Configuration config;
}

int main(int argc, char **argv)
{
	const char *dt_path = nullptr;
	const char *off_path = nullptr;
	int start_lod = 20;
	int end_lod = 100;
	int lod_step = 20;
	int max_objects = -1; // -1 means all objects
	int inspect_obj = 0;  // Object index to inspect in detail

	int opt;
	while ((opt = getopt(argc, argv, "d:o:s:e:p:n:i:")) != -1) {
		switch (opt) {
		case 'd':
			dt_path = optarg;
			break;
		case 'o':
			off_path = optarg;
			break;
		case 's':
			start_lod = atoi(optarg);
			break;
		case 'e':
			end_lod = atoi(optarg);
			break;
		case 'p':
			lod_step = atoi(optarg);
			break;
		case 'n':
			max_objects = atoi(optarg);
			break;
		case 'i':
			inspect_obj = atoi(optarg);
			break;
		default:
			fprintf(stderr,
			        "Usage: %s -d <dt_file> | -o <off_file> "
			        "[-s start_lod] [-e end_lod] [-p lod_step] [-n max_objects] [-i inspect_obj]\n",
			        argv[0]);
			return 1;
		}
	}

	if (!dt_path && !off_path) {
		fprintf(stderr, "Error: must provide -d <dt_file> or -o <off_file>\n");
		fprintf(stderr,
		        "Usage: %s -d <dt_file> | -o <off_file> "
		        "[-s start_lod] [-e end_lod] [-p lod_step] [-n max_objects] [-i inspect_obj]\n",
		        argv[0]);
		return 1;
	}

	printf("=== Face Classifier Test (CPU) ===\n");
	printf("  Start LOD:    %d\n", start_lod);
	printf("  End LOD:      %d\n", end_lod);
	printf("  LOD Step:     %d\n", lod_step);
	printf("  Max Objs:     %d\n", max_objects);
	printf("  Inspect Obj:  %d\n", inspect_obj);
	printf("============================\n\n");

	Tile *tile = nullptr;

	if (dt_path) {
		printf("Loading .dt file: %s\n", dt_path);
		struct timeval tm_start = get_cur_time();

		tile = new Tile(std::string(dt_path));
		size_t total_objects = tile->num_objects();
		size_t process_count = (max_objects < 0) ? total_objects
		                       : std::min((size_t)max_objects, total_objects);

		printf("  Loaded %ld objects from tile [%.2fms]\n",
		       total_objects, get_time_elapsed(tm_start, false));
		printf("  Will process %ld objects\n\n", process_count);

		// ======= CPU Path =======
		// Aggregate stats across all objects and LODs
		std::map<int, ClassificationStats> lod_aggregate;
		std::map<int, std::vector<FaceClassResult>> first_obj_results;

		for (size_t obj_idx = 0; obj_idx < process_count; obj_idx++) {
			HiMesh_Wrapper *wrapper = tile->get_mesh_wrapper(obj_idx);
			wrapper->decode_to(end_lod);
			HiMesh *original = wrapper->get_mesh();

			if (obj_idx == (size_t)inspect_obj) {
				printf("  Object %d: %ld vertices, %ld triangles\n",
				       inspect_obj, original->size_of_vertices(), original->size_of_triangles());
			}

			FaceClassifier classifier(original);

			for (int lod = start_lod; lod <= end_lod; lod += lod_step) {
				if (lod >= end_lod)
					continue;

				HiMesh *low_lod = new HiMesh(original);
				low_lod->decode(lod);

				auto results = classifier.classify_faces(low_lod, 7);
				auto stats = classifier.compute_stats(results);

				if (obj_idx == (size_t)inspect_obj) {
					first_obj_results[lod] = results;
				}

				delete low_lod;

				auto &agg = lod_aggregate[lod];
				agg.total_faces += stats.total_faces;
				agg.inside_faces += stats.inside_faces;
				agg.outside_faces += stats.outside_faces;
				agg.intersect_faces += stats.intersect_faces;
				agg.unknown_faces += stats.unknown_faces;
			}

			if ((obj_idx + 1) % 100 == 0 || obj_idx + 1 == process_count) {
				printf("  Progress: %ld/%ld objects processed\n",
				       obj_idx + 1, process_count);
			}
		}

		printf("\n=== Aggregate Classification Across %ld Objects ===\n\n", process_count);
		for (int lod = start_lod; lod <= end_lod; lod += lod_step) {
			if (lod >= end_lod)
				continue;

			auto &agg = lod_aggregate[lod];
			if (agg.total_faces > 0) {
				agg.inside_ratio = (float)agg.inside_faces / (float)agg.total_faces;
				agg.outside_ratio = (float)agg.outside_faces / (float)agg.total_faces;
				agg.intersect_ratio = (float)agg.intersect_faces / (float)agg.total_faces;
			}
			printf("LOD %d:\n", lod);
			tdbase::FaceClassifier::print_stats(agg);
		}

		// Per-face details for inspected object
		printf("\n=== Object %d Per-Face Details ===\n", inspect_obj);
		for (int lod = start_lod; lod <= end_lod; lod += lod_step) {
			if (lod >= end_lod)
				continue;
			auto it = first_obj_results.find(lod);
			if (it != first_obj_results.end() && !it->second.empty()) {
				printf("\n--- LOD %d ---\n", lod);
				FaceClassifier::print_results(it->second);
			}
		}

	} else if (off_path) {
		printf("Loading .off file: %s\n", off_path);
		struct timeval tm_start = get_cur_time();

		HiMesh *original = read_mesh(off_path, false);

		printf("  Original mesh: %ld vertices, %ld triangles [%.2fms]\n\n",
		       original->size_of_vertices(), original->size_of_triangles(),
		       get_time_elapsed(tm_start, true));

		FaceClassifier classifier(original);

		for (int lod = start_lod; lod <= end_lod; lod += lod_step) {
			if (lod >= end_lod) {
				printf("LOD %d: skip (same as original)\n", lod);
				continue;
			}

			struct timeval lod_start = get_cur_time();

			HiMesh *low_lod = new HiMesh(original);
			low_lod->decode(lod);

			printf("LOD %d: %ld triangles, %ld vertices\n",
			       lod,
			       low_lod->size_of_triangles(),
			       low_lod->size_of_vertices());

			auto results = classifier.classify_faces(low_lod, 7);
			auto stats = classifier.compute_stats(results);
			classifier.print_stats(stats);
			classifier.print_results(results);

			printf("  Time: %.2fms\n\n", get_time_elapsed(lod_start, true));

			delete low_lod;
		}
	}

	delete tile;
	return 0;
}
