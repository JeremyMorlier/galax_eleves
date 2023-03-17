#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include <iostream>
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// init acceleration


	float diffx, diffy, diffz;
	float dij, temp;

	if (i < n_particles)
	{
		accelerationsGPU[i].x = 0.0f;
		accelerationsGPU[i].y = 0.0f;
		accelerationsGPU[i].z = 0.0f;
		for(int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				diffx = positionsGPU[j].x - positionsGPU[i].x;
				diffy = positionsGPU[j].y - positionsGPU[i].y;
				diffz = positionsGPU[j].z - positionsGPU[i].z;

				dij = diffx * diffx + diffy * diffy + diffz * diffz;

				// utiliser un masque
				temp = __frsqrt_rn(dij);
				dij = (dij < 1.0)*10.0 + (dij >= 1.0)*10.0*temp*temp*temp;
				

				accelerationsGPU[i].x += diffx * dij * positionsGPU[j].w;
				accelerationsGPU[i].y += diffy * dij * positionsGPU[j].w;
				accelerationsGPU[i].z += diffz * dij * positionsGPU[j].w;
			}
		}
	}

}

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n_particles)
	{
		velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
		velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
		velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
		positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
		positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
		positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
	}

}

void update_position_cu(float4 * positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;
	//std::cout << nblocks << " " << nthreads << "   " << std::endl;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
	maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU