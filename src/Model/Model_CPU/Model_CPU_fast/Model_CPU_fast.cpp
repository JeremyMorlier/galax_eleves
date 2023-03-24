#ifdef GALAX_MODEL_CPU_FAST
#include <iostream>
#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast ::Model_CPU_fast(const Initstate &initstate, Particles &particles)
    : Model_CPU(initstate, particles)
{
}

struct Rot
{
        static constexpr unsigned get(unsigned i, unsigned n)
        {
                return (i + n - 1) % n;
        }
};

constexpr auto mask = xsimd::make_batch_constant<xsimd::batch<uint32_t, xsimd::avx2>, Rot>();

void Model_CPU_fast ::step()
{
        //     std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
        //     std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
        //     std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

        //OMP + xsimd version

        const b_type zeros_batch = 0.0;
        const b_type tens_batch = 10.0;

        // #pragma omp parallel for
        //         for (int i = 0; i < n_particles; i += b_type::size)
        //         {
        //                 // load registers body i
        //                 const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        //                 const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        //                 const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        //                 //       b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
        //                 //       b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
        //                 //       b_type racc        // b_type test = b_type::load_unaligned(&particles.x[0]);
        // for(int i = 0; i <b_type::size; i++ )
        // {
        //       std::cout<<test.get(i) << " ";
        // }Capture d’écran du 2023-03-24 11-57-00
        // std::cout<< std::endl;
        // xsimd::swizzle(test, mask);
        // for(int i = 0; i <b_type::size; i++ )
        // {
        //       std::cout<<test.get(i) << " ";
        // }
        // std::cout<< std::endl;ype::load_unaligned(&velocitiesy[i]);
        //                 b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
        //                 //#pragma omp parallel forrmasses_j
        //                 for (int j = 0; j < test = ters body j
        //                         const b_type rposx_j = particles.x[j];
        //                         const b_type rposy_j = particles.y[j];
        //                         const b_type rposz_j = particles.z[j];
        //                         const b_type rmasses = initstate.masses[j];
        // b_type test = b_type::load_unaligned(&particles.x[0]);
        // for(int i = 0; i <b_type::size; i++ )
        // {
        //       std::cout<<test.get(i) << " ";
        // }
        // std::cout<< std::endl;
        // xsimd::swizzle(test, mask);
        // for(int i = 0; i <b_type::size; i++ )rmasses_j
        // std::cout<< std::endl;/ (dij_s * dij_s * dij_s);
        //                         dij = 10 / (xs::sqrt(dij) * dij);
        //                         dij = xs::min(dij, tens_batch);

        //                         //dij = xs::select(cond, zeros_batch, dij);
        //                         dij = dij * rmasses * 2.0f;

        //                         rvelx_i += diffx * dij;
        //                         rvely_i += diffy * dij;
        //                         rvelz_i += diffz * dij;
        //                 }
        //                 // raccx_i.store_unaligned(&accelerationsx[i]);
        //                 // raccy_i.store_unaligned(&accelerationsy[i]);
        //                 // raccz_i.store_unaligned(&accelerationsz[i]);

        //                 rvelx_i.store_unaligned(&velocitiesx[i]);
        //                 rvely_i.store_unaligned(&velocitiesy[i]);
        //                 rvelz_i.store_unaligned(&velocitiesz[i]);
        //         }
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i += b_type::size)
        {
                // load registers body i
                const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
                const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
                const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
                b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
                b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
                b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);
                const b_type rmasses_i = b_type::load_unaligned(&initstate.masses[i]);

                #pragma omp parallel for
                for (int j = i; j < n_particles; j += b_type::size)
                {
                        b_type rvelx_j = b_type::load_unaligned(&velocitiesx[j]);
                        b_type rvely_j = b_type::load_unaligned(&velocitiesy[j]);
                        b_type rvelz_j = b_type::load_unaligned(&velocitiesz[j]);
                        b_type rposx_j = b_type::load_unaligned(&particles.x[j]);
                        b_type rposy_j = b_type::load_unaligned(&particles.y[j]);
                        b_type rposz_j = b_type::load_unaligned(&particles.z[j]);

                        b_type rmasses_j = b_type::load_unaligned(&initstate.masses[j]);

                        for (int k = 0; k < b_type::size; k++)
                        {

                                const b_type diffx = rposx_j - rposx_i;
                                const b_type diffy = rposy_j - rposy_i;
                                const b_type diffz = rposz_j - rposz_i;
                                b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;

                                auto cond = xs::eq(zeros_batch, dij);

                                dij = 10 / (xs::sqrt(dij) * dij);
                                dij = xs::min(dij, tens_batch);

                                dij = xs::select(cond, zeros_batch, dij);
                                auto dij_a = dij * rmasses_j * 2.0f;
                                rvelx_i += diffx * dij_a;
                                rvely_i += diffy * dij_a;
                                rvelz_i += diffz * dij_a;

                                auto dij_b = - dij * rmasses_i * 2.0f;
                                rvelx_j += diffx * dij_b;
                                rvely_j += diffy * dij_b;
                                rvelz_j += diffz * dij_b;

                                rposx_j = xsimd::swizzle(rposx_j, mask);
                                rposy_j = xsimd::swizzle(rposy_j, mask);
                                rposz_j = xsimd::swizzle(rposz_j, mask);
                                rmasses_j = xsimd::swizzle(rmasses_j, mask);
                                rvelx_j = xsimd::swizzle(rvelx_j, mask);
                                rvely_j = xsimd::swizzle(rvely_j, mask);
                                rvelz_j = xsimd::swizzle(rvelz_j, mask);
                        }
                        rvelx_j.store_unaligned(&velocitiesx[j]);
                        rvely_j.store_unaligned(&velocitiesy[j]);
                        rvelz_j.store_unaligned(&velocitiesz[j]);
                }

                rvelx_i.store_unaligned(&velocitiesx[i]);
                rvely_i.store_unaligned(&velocitiesy[i]);
                rvelz_i.store_unaligned(&velocitiesz[i]);
        }

#pragma omp parallel for
        for (int i = 0; i < n_particles; i += b_type::size)
        {
                b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
                b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
                b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

                const b_type rvelx_i = b_type::load_unaligned(&velocitiesx[i]);
                const b_type rvely_i = b_type::load_unaligned(&velocitiesy[i]);
                const b_type rvelz_i = b_type::load_unaligned(&velocitiesz[i]);


                rposx_i += rvelx_i * 0.1f;
                rposy_i += rvely_i * 0.1f;
                rposz_i += rvelz_i * 0.1f;

                rposx_i.store_unaligned(&particles.x[i]);
                rposy_i.store_unaligned(&particles.y[i]);
                rposz_i.store_unaligned(&particles.z[i]);
        }
}

#endif // GALAX_MODEL_CPU_FAST