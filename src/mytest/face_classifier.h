#ifndef MYTEST_FACE_CLASSIFIER_H
#define MYTEST_FACE_CLASSIFIER_H

#include "himesh.h"
#include <CGAL/Side_of_triangle_mesh.h>
#include <vector>
#include <unordered_map>

namespace tdbase {

enum FaceLocation {
	FACE_INSIDE = 0,
	FACE_OUTSIDE = 1,
	FACE_INTERSECT = 2,
	FACE_UNKNOWN = 3
};

struct FaceClassResult {
	int face_id;
	FaceLocation location;
	float centroid[3];
	int inside_samples;
	int outside_samples;
	int total_samples;
	float inside_ratio;
};

struct ClassificationStats {
	size_t total_faces;
	size_t inside_faces;
	size_t outside_faces;
	size_t intersect_faces;
	size_t unknown_faces;
	float inside_ratio;
	float outside_ratio;
	float intersect_ratio;
};

class FaceClassifier {
public:
	FaceClassifier(HiMesh *original_mesh);
	~FaceClassifier();

	std::vector<FaceClassResult> classify_faces(HiMesh *low_lod_mesh,
	                                            int num_samples_per_face = 5);

	FaceLocation classify_triangle(const float tri[9],
	                               int num_samples = 5);

	bool is_point_inside(const Point &p);

	static ClassificationStats compute_stats(const std::vector<FaceClassResult> &results);
	static void print_stats(const ClassificationStats &stats);
	static void print_results(const std::vector<FaceClassResult> &results);

private:
	HiMesh *original_mesh;
	TriangleTree *tree;
	aab bbox;
	CGAL::Side_of_triangle_mesh<CGAL::Polyhedron_3<MyKernel, MyItems>, MyKernel> *side_of_mesh;

	Point triangle_centroid(const float tri[9]);
	Point triangle_centroid(const Point &p1, const Point &p2, const Point &p3);
	void sample_triangle_points(const float tri[9], int num_samples,
	                            std::vector<Point> &samples);
};

} // namespace tdbase

#endif // MYTEST_FACE_CLASSIFIER_H
