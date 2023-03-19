#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include <iostream>
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// local shared memory (allows concurrent read to the same data)
	__shared__ float4 localParticles[n_particles];

	if (i < n_particles)
	{
		float diffx, diffy, diffz;
		float dij, temp;
		// each thread load its point to the shared memory
		localParticles[i] = positionsGPU[i]
		// wait for each thread to update the shared position of each particle
		__syncthreads();

		// reset acceleration
		accelerationsGPU[i].x = 0.0f;
		accelerationsGPU[i].y = 0.0f;
		accelerationsGPU[i].z = 0.0f;
		
		for(int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				diffx = localParticles[j].x - localParticles[i].x;
				diffy = localParticles[j].y - localParticles[i].y;
				diffz = localParticles[j].z - localParticles[i].z;

				dij = diffx * diffx + diffy * diffy + diffz * diffz;

				// utiliser un masque
				temp = __frsqrt_rn(dij);
				dij = (dij < 1.0)*10.0 + (dij >= 1.0)*10.0*temp*temp*temp;
				

<<<<<<< HEAD
				accelerationsGPU[i].x += diffx * dij * positionsGPU[j].w;
				accelerationsGPU[i].y += diffy * dij * positionsGPU[j].w;
				accelerationsGPU[i].z += diffz * dij * positionsGPU[j].w;
=======
				accelerationsGPU[i].x += diffx * dij * localParticles[j].w;
				accelerationsGPU[i].y += diffy * dij * localParticles[j].w;
				accelerationsGPU[i].z += diffz * dij * localParticles[j].w;
>>>>>>> cd602b6840b2eb9e3de49dc96f997b71da4e6907
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