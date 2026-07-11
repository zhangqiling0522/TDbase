/*
 * himesh_wrapper.cpp
 *
 *  Created on: Jun 24, 2022
 *      Author: teng
 */


#include "himesh.h"

namespace tdbase{

/*
 * himesh wrapper functions
 * */

HiMesh_Wrapper::HiMesh_Wrapper(char *dt, size_t i, Decoding_Type t){
	type = t;
	id = i;
	box.id = i;
	data_size = *(size_t *)(dt);

	data_buffer = dt + sizeof(size_t);
	meta_buffer = data_buffer + data_size;

	size_t vnum = *(size_t *)meta_buffer;
	meta_size = sizeof(vnum);
	if(type == COMPRESSED){
		// the mesh reuses the memory space stored in the Tile class
		mesh = new HiMesh(data_buffer, data_size, false);

		for(int i=0;i<vnum;i++){
			Voxel *v = new Voxel();
			memcpy(v->low, meta_buffer+meta_size, 3*sizeof(float));
			meta_size += 3*sizeof(float);
			memcpy(v->high, meta_buffer+meta_size, 3*sizeof(float));
			meta_size += 3*sizeof(float);
			memcpy(v->core, meta_buffer+meta_size, 3*sizeof(float));
			meta_size += 3*sizeof(float);
			voxels.push_back(v);
			box.update(*v);
			v->id = i;
		}
	}else{
		for(int lod: lod_levels()){
			this->hausdorffs[lod] = *(float *)(meta_buffer+meta_size);
			meta_size += sizeof(float);
			this->proxyhausdorffs[lod] = *(float *)(meta_buffer+meta_size);
			meta_size += sizeof(float);
		}

		// record the decoded data for different voxels in different LODs
		for(int i=0;i<vnum;i++){
			Voxel *v = new Voxel();
			// load the space information
			memcpy(v->low, meta_buffer+meta_size, 3*sizeof(float));
			meta_size += 3*sizeof(float);
			memcpy(v->high, meta_buffer+meta_size, 3*sizeof(float));
			meta_size += 3*sizeof(float);
			memcpy(v->core, meta_buffer+meta_size, 3*sizeof(float));
			meta_size += 3*sizeof(float);
			v->id = i;

			// load the offset and volume information for varying LODs
			for(int lod: lod_levels()){
				size_t of = *(size_t *)(meta_buffer+meta_size);
				meta_size += sizeof(size_t);
				size_t vl = *(size_t *)(meta_buffer+meta_size);
				meta_size += sizeof(size_t);
				v->offset_lod[lod] = of;
				v->volume_lod[lod] = vl;
				//printf("%ld\t%ld\t%ld\t%ld\t%ld\n", id, i, lod, v->offset_lod[lod], v->volumn_lod[lod]);
			}

			voxels.push_back(v);
			box.update(*v);
		}
	}
	pthread_mutex_init(&lock, NULL);
}

// manually
HiMesh_Wrapper::HiMesh_Wrapper(map<int, HiMesh *> &ms, int voxel_size){
	type = MULTIMESH;
	for(auto a:ms){
		meshes[a.first] = a.second;
	}
	assert(ms.find(100)!=ms.end());

	voxels = ms[100]->generate_voxels_skeleton(voxel_size);
	for(Voxel *v:voxels){
		box.update(*v);
	}

}

HiMesh_Wrapper::HiMesh_Wrapper(HiMesh *m, int voxel_size){
	type = COMPRESSED;
	mesh = m;
	voxels = m->generate_voxels_skeleton(voxel_size);
	m->encode();
}

HiMesh_Wrapper::~HiMesh_Wrapper(){
	for(Voxel *v:voxels){
		delete v;
	}
	voxels.clear();
	if(mesh){
		if(mesh->original_mesh){
			delete mesh->original_mesh;
		}
		delete mesh;
	}
	for(auto a:meshes){
		delete a.second;
	}
	meshes.clear();
}

void HiMesh_Wrapper::decode_to(int lod){
	if(lod <= cur_lod){
		return;
	}
	cur_lod = lod;
	for(Voxel *v:voxels){
		v->clear();
	}

	if(type == COMPRESSED){
			assert(mesh);
			mesh->decode(lod);
			mesh->fill_voxels(voxels);
	}else if(type == MULTIMESH){
		assert(meshes.find(cur_lod)!=meshes.end());
		get_mesh()->fill_voxels(voxels);
	}else{
		// for RAW data mode, simply link pointers instead of do the decoding job
		// as data already been stored separately for each voxel
		for(int i=0;i<voxels.size();i++){
			size_t offset = get_voxel_offset(i, lod);
			size_t size = get_voxel_size(i, lod);
			voxels[i]->external_load((float *)(data_buffer+offset), (float *)(data_buffer+offset+9*size*sizeof(float)), size);
		}
	}
	if(config.verbose>=3){
		for(int i=0;i<voxels.size();i++){
			printf("decode_to: id: %ld\t voxel_id: %d\t lod: %d\t offset: %ld\t volume: %ld\n", id, i, lod, voxels[i]->offset_lod[lod], voxels[i]->volume_lod[lod]);
			voxels[i]->print();
		}
	}
}

float HiMesh_Wrapper::getHausdorffDistance(){
	if(type == COMPRESSED){
		return mesh->getHausdorffDistance();
	}else{
		assert(hausdorffs.find(cur_lod)!=hausdorffs.end());
		return hausdorffs[cur_lod];
	}
}
float HiMesh_Wrapper::getProxyHausdorffDistance(){
	if(type == COMPRESSED){
		return mesh->getProxyHausdorffDistance();
	}else{
		assert(hausdorffs.find(cur_lod)!=hausdorffs.end());
		return proxyhausdorffs[cur_lod];
	}
}

size_t HiMesh_Wrapper::get_voxel_offset(int id, int lod){
	return voxels[id]->offset_lod[lod];
}
size_t HiMesh_Wrapper::get_voxel_size(int id, int lod){
	return voxels[id]->volume_lod[lod];
}


// Compute Hausdorff distances for all LOD levels against LOD100.
// Only valid for MULTIMESH type. Call before dump_raw.
void HiMesh_Wrapper::computeMultiLodHausdorff(){
	if(type != MULTIMESH) return;
	assert(meshes.find(100) != meshes.end());

	HiMesh *ref = meshes[100];
	ref->updateAABB();
	ref->area_unit = ref->sampling_gap();
	ref->sample_points(ref->area_unit);

	struct timeval start = get_cur_time();

	for(int lod: lod_levels_desc()){
		if(lod == 100){
			continue;
		}
		assert(meshes.find(lod) != meshes.end());
		HiMesh *cur = meshes[lod];

		// Mark all faces as Splittable so the existing
		// computeHausdorfDistance() processes them
		for(HiMesh::Face_iterator fit = cur->facets_begin();
		    fit != cur->facets_end(); ++fit){
			fit->setSplittable();
		}

		auto hd = cur->computeHausdorfDistance(ref);
		hausdorffs[lod] = hd.second;
		proxyhausdorffs[lod] = hd.first;
		auto avg = cur->collectGlobalHausdorff(AVG);
		log("LOD%d hausdorff MAX(ph=%.3f,h=%.3f) AVG(ph=%.3f,h=%.3f) vertices=%zu",
		    lod, hd.first, hd.second, avg.first, avg.second, cur->size_of_vertices());
	}

	logt("compute hausdorff for multi-lod mesh %ld done", start, id);
}

}
