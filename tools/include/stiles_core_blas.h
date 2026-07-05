/**
 * @file stiles_core_blas.h
 *
 * sTiles core BLAS constants and types.
 * This header provides the constants used throughout the sTiles library.
 */
#ifndef STILES_CORE_BLAS_H
#define STILES_CORE_BLAS_H

#ifdef __cplusplus
extern "C" {
#endif

/* sTiles operation constants */
#define sTilesNoTrans       111
#define sTilesTrans         112
#define sTilesUpper         121
#define sTilesLower         122
#define sTilesUpperLower    123

#define sTilesNonUnit       131
#define sTilesUnit          132
#define sTilesLeft          141
#define sTilesRight         142

/* sTiles error codes */
#define STILES_SUCCESS                 0
#define STILES_FAILURE                -1
#define STILES_ERR_NOT_INITIALIZED  -101
#define STILES_ERR_REINITIALIZED    -102
#define STILES_ERR_NOT_SUPPORTED    -103
#define STILES_ERR_ILLEGAL_VALUE    -104
#define STILES_ERR_NOT_FOUND        -105
#define STILES_ERR_OUT_OF_RESOURCES -106
#define STILES_ERR_INTERNAL_LIMIT   -107
#define STILES_ERR_UNALLOCATED      -108
#define STILES_ERR_FILESYSTEM       -109
#define STILES_ERR_UNEXPECTED       -110
#define STILES_ERR_SEQUENCE_FLUSHED -111

#ifdef __cplusplus
}
#endif

#endif /* STILES_CORE_BLAS_H */
