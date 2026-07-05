/**
 * @file stiles_types.hpp
 * @brief Core type definitions and status codes for sTiles library.
 *
 * Defines fundamental type aliases (i32, f64, etc.), status codes, and
 * enumeration types used throughout the sTiles sparse matrix library.
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

#ifndef STILES_TYPES_HPP
#define STILES_TYPES_HPP

#include <cstdint>
#include <string>       
#include <string_view>  
#include <cstddef>      
// Use explicit path to shared include directory
#include "stiles_config.hpp"

namespace sTiles {

//_________________________________________________________________________________________________

/**
 * @name Status and Error Codes
 * @{
 */
enum class StatusCode : int {
    Success           =   0, ///< Operation completed successfully.
    Ok                =   Success, ///< Alias for Success (legacy name).
    Failure           =  -1, ///< General failure.
    NotInitialized    = -101, ///< Library has not been initialized.
    Reinitialized     = -102, ///< Library was reinitialized incorrectly.
    NotSupported      = -103, ///< The requested operation is not supported.
    IllegalValue      = -104, ///< An illegal value was passed to a function.
    NotFound          = -105, ///< The requested item could not be found.
    OutOfResources    = -106, ///< Insufficient resources (e.g., memory) to proceed.
    InternalLimit     = -107, ///< An internal library limit was reached.
    Unallocated       = -108, ///< A resource was not allocated.
    FilesystemError   = -109, ///< A filesystem-related error occurred.
    Unexpected        = -110, ///< An unexpected internal error occurred.
    SequenceFlushed   = -111, ///< A sequence of operations was flushed prematurely.
    ExecutionFailed   = -112, ///< A computation failed due to a numerical issue (e.g., matrix is singular or an algorithm did not converge).
    InvalidArgument   = IllegalValue ///< Alias for IllegalValue (legacy name).
};
/** @} */

//_________________________________________________________________________________________________

/**
 * @name Core Type Aliases
 * @brief Modern C++ type aliases for fundamental data types.
 *
 * This section defines short, professional aliases for fixed-width types.
 * The naming convention is:
 * - `i` for signed integers (e.g., i32)
 * - `u` for unsigned integers (e.g., u64)
 * - `f` for floating-point numbers (e.g., f64)
 * @{
 */

// --- Fundamental Types ---
using Bool = bool;          ///< Standard boolean type.

using i8  = std::int8_t;    ///< 8-bit signed integer.
using i16 = std::int16_t;   ///< 16-bit signed integer.
using i32 = std::int32_t;   ///< 32-bit signed integer.
using i64 = std::int64_t;   ///< 64-bit signed integer.
using u8  = std::uint8_t;   ///< 8-bit unsigned integer.
using u16 = std::uint16_t;  ///< 16-bit unsigned integer.
using u32 = std::uint32_t;  ///< 32-bit unsigned integer.
using u64 = std::uint64_t;  ///< 64-bit unsigned integer.
using f32 = float;          ///< Single-precision (32-bit) floating-point number.
using f64 = double;         ///< Double-precision (64-bit) floating-point number.

// --- Semantic Types ---
/**
 * @brief Type for indexing into matrices and vectors.
 * @note Based on a 64-bit integer for large-scale problem support.
 */
using Index = sTiles::i64;

/**
 * @brief Type for representing dimension sizes and counts.
 * @note Based on a 64-bit integer to match the Index type.
 */
using Size = sTiles::i64;
using Int = sTiles::i32;


/** @} */

//_________________________________________________________________________________________________

/**
 * @name Enumerated Constants
 * @{
 */

enum class Parameter : int {
    TreeReductionAcc              = 0,
    TreeReductionMaxCores         = 1,
    TreeReductionStrategy         = 2,
    AndMinCores                   = 3,
    AndMinMatrixSize              = 4,
    AndMultipleNestedDissection   = 5,
    AndTileFillInRef              = 6,
    AndAllowedBan                 = 7,
    CoresSplitStrategy            = 8, // Note: Corrected typo from "STARTEGY"
    BoostedETrick                 = 9
};

enum class Precision {
    Byte,
    Integer,
    RealFloat,
    RealDouble,
    ComplexFloat,
    ComplexDouble
};

enum class Op {
    NoTrans = 111,
    Trans   = 112,
    ConjTrans = 113 
};

enum class Uplo {
    Upper   = 121,
    Lower   = 122,
    General = 123
};

enum class Diag {
    NonUnit = 131,
    Unit    = 132
};

enum class Side {
    Left  = 141,
    Right = 142
};

enum class SymbolicMethod {
    Auto,
    Sparse,
    Dense,
    SemiSparse,
    SemiSparsePro,
    SemiSparseVec,
    SemiSparseInplace
};

enum class Function : int {
    DPOSV = 5
};


enum class Norm {
    One       = 171, ///< One norm (max column sum).
    Two       = 173, ///< Two norm (Euclidean, not for all routines).
    Frobenius = 174, ///< Frobenius norm.
    Infinity  = 175, ///< Infinity norm (max row sum).
    Max       = 177  ///< Max norm (max absolute value).
};

enum class Direction        {Forward  = 391, Backward = 392};
enum class Layout           {Columnwise = 401, Rowwise   = 402};
enum class SchedulingPolicy { STATIC, DYNAMIC };
enum class WorkerCommand    { STAND_BY, RUN_TASK, FINALIZE };

enum class OperatingSystem {
    Linux,
    FreeBSD,
    Windows,
    MacOS,
    AIX,
    Unsupported
};

enum class TileStorage {
    Dense,  
    Sparse, 
    Hybrid,
    Smart  
};

enum class SolveType {
    ForwardSubstitution = 0,
    BackwardSubstitution = 1,
    FullSolve = 2 // L and L^T
};

enum class Action : int {
    StandBy  = 0, ///< The thread is idle, waiting for a command.
    Parallel = 1, ///< The thread is executing the main parallel function.
    Dynamic  = 2, ///< Reserved for dynamic task scheduling.
    Finalize = 3  ///< The thread should terminate and exit.
};

enum class LookupMethod {
    Boolean   = 0, ///< Simple boolean array or flag-based lookup.
    HashTable = 1, ///< Hash table for fast key-value lookup.
    CSR       = 2  ///< Compressed Sparse Row format for sparse lookups.
};


} // namespace sTiles

#endif // STILES_TYPES_HPP
