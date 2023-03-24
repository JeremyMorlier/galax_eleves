#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include <iostream>
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define ARRAYSIZE (64)

__global__ void compute_acc(float4 * positionsGPU, float3 * velocitiesGPU, float4 * accelerationsGPU, int n_particles, int nthreads, int nblocks)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = threadIdx.x;

	// local shared memory (allows concurrent read to the same data)
	__shared__ float4 localParticles[ARRAYSIZE];

	if (i < n_particles)
	{


		float diffx, diffy, diffz;
		float dij;
		float4 myPos;
		float4 myAcc;
		int k = 0;

		// load the block own points
		myPos = positionsGPU[i];
		myAcc = accelerationsGPU[i];

		unsigned int tempp = (n_particles + (ARRAYSIZE -1)) / ARRAYSIZE;

		for(k = 0 ; k < tempp; k++)
		{
			for(int q = 0; q < ARRAYSIZE/nthreads; q++)
			{
				localParticles[ARRAYSIZE/nthreads*j + q] = positionsGPU[ARRAYSIZE/nthreads*j + q + k*ARRAYSIZE];
			}

			__syncthreads();
			for(int p = 0; p < ARRAYSIZE; p++)
			{
				if(i != ARRAYSIZE*k + p)
				{
					diffx = __fsub_rn(localParticles[p].x, myPos.x);
					diffy = __fsub_rn(localParticles[p].y, myPos.y);
					diffz = __fsub_rn(localParticles[p].z, myPos.z);

					dij = fmaf(diffx, diffx, fmaf(diffy, diffy, diffz*diffz ));
					dij = __fmul_rn(10.0, __powf(fmax(dij, 1.0f) , -1.5f));
					//dij = __fmul_rn(10.0, __powf(dij , -1.5f));
					
					myAcc.x += diffx * dij * localParticles[p].w;
					myAcc.y += diffy * dij * localParticles[p].w;
					myAcc.z += diffz * dij * localParticles[p].w;
				}
			}
			__syncthreads();
		}

		myAcc.w = 0.0f;
		__syncthreads();
		accelerationsGPU[i] = myAcc;
		
	}

}

__global__ void maj_pos(float4 * positionsGPU, float3 * velocitiesGPU, float4 * accelerationsGPU, int n_particles)
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

		// reset acceleration
		accelerationsGPU[i].x = 0.0f;
		accelerationsGPU[i].y = 0.0f;
		accelerationsGPU[i].z = 0.0f;
	}

}

void update_position_cu(float4 * positionsGPU, float3* velocitiesGPU, float4* accelerationsGPU, int n_particles)
{
	int nthreads = 64;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;
	//std::cout << nblocks << " " << nthreads << "   " << std::endl;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles, nthreads, nblocks);
	maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU