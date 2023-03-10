#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

// // OMP  version
//  #pragma omp parallel for collapse(2)
//     for (int i = 0; i < n_particles; i ++)
//     {
// 		for (int j = 0; j < n_particles; j++)
// 		{
// 			if(i != j)
// 			{
// 				const float diffx = particles.x[j] - particles.x[i];
// 				const float diffy = particles.y[j] - particles.y[i];
// 				const float diffz = particles.z[j] - particles.z[i];

// 				float dij = diffx * diffx + diffy * diffy + diffz * diffz;

// 				if (dij < 1.0)b_type::size

// 				accelerationsx[i] += diffx * dij * initstate.masses[j];
// 				accelerationsy[i] += diffy * dij * initstate.masses[j];
// 				accelerationsz[i] += diffz * dij * initstate.masses[j];
// 			}
// 		}
//     }
// #pragma omp parallel for 
// 	for (int i = 0; i < n_particles; i++)
// 	{
// 		velocitiesx[i] += accelerationsx[i] * 2.0f;
// 		velocitiesy[i] += accelerationsy[i] * 2.0f;
// 		velocitiesz[i] += accelerationsz[i] * 2.0f;
// 		particles.x[i] += velocitiesx   [i] * 0.1f;
// 		particles.y[i] += velocitiesy   [i] * 0.1f;
// 		particles.z[i] += velocitiesz   [i] * 0.1f;
// 	}


// OMP + xsimd version
// std::size_t all_size = accelerationsx.size();
// std::size_t vec_size = all_size - all_size % b_type::size;	

#pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        // load registers body i
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
              b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
              b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
              b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);
		#pragma omp parallel for
			for (int j = 0; j < n_particles; j += b_type::size)
			{
				//load registers body j
				const b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
				const b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
				const b_type rposz_j = b_type::load_unaligned(&particles.z[j]);

				const b_type rmasses = b_type::load_unaligned(&initstate.masses[j]);
				
				const auto diffx = rposx_j - rposx_i;
				const auto diffy = rposy_j - rposy_i;
				const auto diffz = rposz_j - rposz_i;

				auto dij = diffx * diffx + diffy * diffy + diffz * diffz;

				if (dij < 1.0)
				{
					dji = 10.0;
				}
				else
				{i += b_type::size
					dij = std::sqrt(dij);
					dij = 10.0/ (dij * dij * dij);
				}

				raccx_i += diffx * dij * rmasses;
				raccy_i += diffy * dij * rmasses;
				raccz_i += diffz * dij * rmasses;

				raccx_i.store_unaligned(&accelerationsx[i]);
				raccy_i.store_unaligned(&accelerationsy[i]);
				raccz_i.store_unaligned(&accelerationsz[i]);

			}

    }

#pragma omp parallel for 
	for (int i = 0; i < n_particles; i += b_type::size)
	{
		b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
        b_type rvely_i = b_type::load_unaligned(&celocitiesy[i]);
        b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
        const b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        const b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        const b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

		rvelx_i += raccx_i * 2.0f;
		rvely_i += raccy_i * 2.0f;
		rvelz_i += raccz_i * 2.0f;
		auto rposx_i += rvelx_i * 0.1f;
		auto rposy_i += rvely_i * 0.1f;
		auto rposz_i += rvelz_i * 0.1f;

		rposx_i.store_unaligned(&particles.x[i]);
		rposy_i.store_unaligned(&particles.y[i]);
		rposz_i.store_unaligned(&particles.z[i]);		
	}

}

#endif // GALAX_MODEL_CPU_FAST
