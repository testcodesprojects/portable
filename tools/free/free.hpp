/**
 * @file free.hpp
 * @brief Resource-cleanup routines for sTiles tiled-matrix schemes and groups.
 *
 * Implements the destruction path used by sTiles_freeGroup / sTiles_quit:
 *   - destroy_tiled_matrix             : tear down a single TiledMatrix scheme
 *   - destroy_all_schemes_for_group    : destroy every scheme owned by one group
 *   - destroy_all_schemes              : destroy every scheme owned by an sTiles_object
 *
 * All three were previously inline in tools/process/process.cpp; they are
 * factored out here so the cleanup logic lives in one place.
 */

#ifndef _STILES_FREE_HPP_
#define _STILES_FREE_HPP_

#include "../common/stiles_structs.hpp"

namespace sTiles {

void destroy_tiled_matrix(TiledMatrix* tm);
void destroy_all_schemes_for_group(sTiles_object* s, int group_index);
void destroy_all_schemes(sTiles_object* s);

} // namespace sTiles

#endif // _STILES_FREE_HPP_
