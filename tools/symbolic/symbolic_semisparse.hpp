/**
 * @file symbolic_semisparse.hpp
 * @brief Interface for semisparse and sparse symbolic factorization.
 *
 * Declares functions for symbolic factorization phases including semisparse
 * symbolic analysis, correction passes, and full sparse symbolic factorization
 * with parallel execution support.
 *
 * @author Esmail Abdul Fattah
 * @affiliation King Abdullah University of Science and Technology (KAUST)
 * @version 3.0.0
 *
 * @copyright Copyright (c) 2026, KAUST. All rights reserved.
 *
 * PROPRIETARY AND CONFIDENTIAL
 * This software is the exclusive property of KAUST. Unauthorized copying, modification,
 * distribution, or use of this software, in whole or in part, is strictly prohibited.
 * This code may not be reproduced, reverse-engineered, or used to create derivative works
 * without explicit written permission from the copyright holder.
 */

#ifndef STILES_SYMBOLIC_SEMISPARSE_HPP
#define STILES_SYMBOLIC_SEMISPARSE_HPP

#include "../common/stiles_types.hpp"
#include "../common/stiles_structs.hpp"

namespace sTiles {
namespace preprocess {

sTiles::StatusCode symbolic_semisparse(int group_index,
                                       int call_index,
                                       TiledMatrix *scheme);


sTiles::StatusCode symbolic_sparse(int group_index,
                                   int call_index,
                                   TiledMatrix *scheme,
                                   int num_cores);

} // namespace preprocess
} // namespace sTiles

#endif // STILES_SYMBOLIC_SEMISPARSE_HPP
