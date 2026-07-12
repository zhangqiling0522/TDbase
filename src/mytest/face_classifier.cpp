#include "face_classifier.h"
#include <algorithm>
#include <cmath>

namespace tdbase {

FaceClassifier::FaceClassifier(HiMesh *original)
	: original_mesh(original)
{
	tree = original_mesh->get_aabb_tree_triangle();
	bbox = original_mesh->get_mbb();
	side_of_mesh = new CGAL::Side_of_triangle_mesh<CGAL::Polyhedron_3<MyKernel, MyItems>, MyKernel>(*original_mesh);
}

FaceClassifier::~FaceClassifier()
{
	delete side_of_mesh;
}

Point FaceClassifier::triangle_centroid(const float tri[9])
{
	return Point(
		(tri[0] + tri[3] + tri[6]) / 3.0f,
		(tri[1] + tri[4] + tri[7]) / 3.0f,
		(tri[2] + tri[5] + tri[8]) / 3.0f
	);
}

Point FaceClassifier::triangle_centroid(const Point &p1, const Point &p2, const Point &p3)
{
	return Point(
		(p1.x() + p2.x() + p3.x()) / 3.0f,
		(p1.y() + p2.y() + p3.y()) / 3.0f,
		(p1.z() + p2.z() + p3.z()) / 3.0f
	);
}

bool FaceClassifier::is_point_inside(const Point &p)
{
	CGAL::Bounded_side side = (*side_of_mesh)(p);
	return side == CGAL::ON_BOUNDED_SIDE;
}

void FaceClassifier::sample_triangle_points(const float tri[9], int num_samples,
                                            std::vector<Point> &samples)
{
	samples.clear();

	Point v0(tri[0], tri[1], tri[2]);
	Point v1(tri[3], tri[4], tri[5]);
	Point v2(tri[6], tri[7], tri[8]);

	samples.push_back(triangle_centroid(tri));

	if (num_samples >= 4) {
		Point m01((v0.x() + v1.x()) / 2.0f,
		          (v0.y() + v1.y()) / 2.0f,
		          (v0.z() + v1.z()) / 2.0f);
		Point m12((v1.x() + v2.x()) / 2.0f,
		          (v1.y() + v2.y()) / 2.0f,
		          (v1.z() + v2.z()) / 2.0f);
		Point m20((v2.x() + v0.x()) / 2.0f,
		          (v2.y() + v0.y()) / 2.0f,
		          (v2.z() + v0.z()) / 2.0f);
		samples.push_back(m01);
		samples.push_back(m12);
		samples.push_back(m20);
	}

	if (num_samples >= 7) {
		Point q0((v0.x() * 2.0f + v1.x() + v2.x()) / 4.0f,
		         (v0.y() * 2.0f + v1.y() + v2.y()) / 4.0f,
		         (v0.z() * 2.0f + v1.z() + v2.z()) / 4.0f);
		Point q1((v1.x() * 2.0f + v0.x() + v2.x()) / 4.0f,
		         (v1.y() * 2.0f + v0.y() + v2.y()) / 4.0f,
		         (v1.z() * 2.0f + v0.z() + v2.z()) / 4.0f);
		Point q2((v2.x() * 2.0f + v0.x() + v1.x()) / 4.0f,
		         (v2.y() * 2.0f + v0.y() + v1.y()) / 4.0f,
		         (v2.z() * 2.0f + v0.z() + v1.z()) / 4.0f);
		samples.push_back(q0);
		samples.push_back(q1);
		samples.push_back(q2);
	}
}

FaceLocation FaceClassifier::classify_triangle(const float tri[9],
                                               int num_samples)
{
	std::vector<Point> sample_points;
	sample_triangle_points(tri, num_samples, sample_points);

	int inside_count = 0;
	int outside_count = 0;

	for (const Point &sp : sample_points) {
		if (is_point_inside(sp)) {
			inside_count++;
		} else {
			outside_count++;
		}
	}

	int total = inside_count + outside_count;
	if (total == 0)
		return FACE_UNKNOWN;

	float inside_ratio = (float)inside_count / (float)total;

	if (inside_ratio >= 0.75f)
		return FACE_INSIDE;
	else if (inside_ratio <= 0.25f)
		return FACE_OUTSIDE;
	else
		return FACE_INTERSECT;
}

std::vector<FaceClassResult> FaceClassifier::classify_faces(HiMesh *low_lod_mesh,
                                                            int num_samples_per_face)
{
	std::vector<FaceClassResult> results;

	float *triangles = nullptr;
	size_t num_tris = low_lod_mesh->fill_triangles(triangles);
	results.reserve(num_tris);

	for (size_t i = 0; i < num_tris; i++) {
		const float *tri = triangles + i * 9;

		FaceClassResult res;
		res.face_id = (int)i;
		res.centroid[0] = (tri[0] + tri[3] + tri[6]) / 3.0f;
		res.centroid[1] = (tri[1] + tri[4] + tri[7]) / 3.0f;
		res.centroid[2] = (tri[2] + tri[5] + tri[8]) / 3.0f;
		res.inside_samples = 0;
		res.outside_samples = 0;
		res.total_samples = 0;

		std::vector<Point> sample_points;
		sample_triangle_points(tri, num_samples_per_face, sample_points);

		for (const Point &sp : sample_points) {
			if (is_point_inside(sp)) {
				res.inside_samples++;
			} else {
				res.outside_samples++;
			}
		}

		int total = res.inside_samples + res.outside_samples;
		res.total_samples = total;

		if (total > 0) {
			res.inside_ratio = (float)res.inside_samples / (float)total;

			if (res.inside_ratio >= 0.75f)
				res.location = FACE_INSIDE;
			else if (res.inside_ratio <= 0.25f)
				res.location = FACE_OUTSIDE;
			else
				res.location = FACE_INTERSECT;
		} else {
			res.inside_ratio = 0.0f;
			res.location = FACE_UNKNOWN;
		}

		results.push_back(res);
	}

	delete[] triangles;
	return results;
}

ClassificationStats FaceClassifier::compute_stats(const std::vector<FaceClassResult> &results)
{
	ClassificationStats stats;
	stats.total_faces = results.size();
	stats.inside_faces = 0;
	stats.outside_faces = 0;
	stats.intersect_faces = 0;
	stats.unknown_faces = 0;

	for (const auto &r : results) {
		switch (r.location) {
		case FACE_INSIDE:
			stats.inside_faces++;
			break;
		case FACE_OUTSIDE:
			stats.outside_faces++;
			break;
		case FACE_INTERSECT:
			stats.intersect_faces++;
			break;
		default:
			stats.unknown_faces++;
			break;
		}
	}

	if (stats.total_faces > 0) {
		stats.inside_ratio = (float)stats.inside_faces / (float)stats.total_faces;
		stats.outside_ratio = (float)stats.outside_faces / (float)stats.total_faces;
		stats.intersect_ratio = (float)stats.intersect_faces / (float)stats.total_faces;
	} else {
		stats.inside_ratio = 0.0f;
		stats.outside_ratio = 0.0f;
		stats.intersect_ratio = 0.0f;
	}

	return stats;
}

void FaceClassifier::print_stats(const ClassificationStats &stats)
{
	printf("=== Face Classification Statistics ===\n");
	printf("  Total faces:    %zu\n", stats.total_faces);
	printf("  Inside faces:   %zu (%.2f%%)\n",
	       stats.inside_faces, stats.inside_ratio * 100.0f);
	printf("  Outside faces:  %zu (%.2f%%)\n",
	       stats.outside_faces, stats.outside_ratio * 100.0f);
	printf("  Intersect faces:%zu (%.2f%%)\n",
	       stats.intersect_faces, stats.intersect_ratio * 100.0f);
	printf("  Unknown faces:  %zu\n", stats.unknown_faces);
	printf("======================================\n");
}

void FaceClassifier::print_results(const std::vector<FaceClassResult> &results)
{
	printf("=== Per-Face Classification Results ===\n");
	printf("%-8s %-12s %-10s %-10s %-10s %-10s\n",
	       "FaceID", "Location", "Inside%", "Inside", "Outside", "Total");
	printf("--------------------------------------------------------------\n");

	const char *loc_str[] = {"INSIDE", "OUTSIDE", "INTERSECT", "UNKNOWN"};

	for (const auto &r : results) {
		printf("%-8d %-12s %-10.1f %-10d %-10d %-10d\n",
		       r.face_id,
		       loc_str[r.location],
		       r.total_samples > 0 ? r.inside_ratio * 100.0f : 0.0f,
		       r.inside_samples,
		       r.outside_samples,
		       r.total_samples);
	}
	printf("--------------------------------------------------------------\n");
	printf("Total: %zu faces\n\n", results.size());
}

} // namespace tdbase
