/**
 * @file tile.h
 *
 * Implementation of auxiliary routines for the sTiles framework.
 * These routines provide utilities for accessing and manipulating tiled matrices
 * in high-performance linear algebra computations.
 *
 * Redesigned by:
 * - Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST).
 *
 * Originally developed as part of the PLASMA project by:
 * - Jakub Kurzak
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * This file includes utility functions for:
 * - Computing element addresses within tiled matrices.
 * - Handling block leading dimensions for non-square tiles.
 *
 * @version 1.0.0
 * @author Esmail Abdul Fattah
 * @original_author Jakub Kurzak
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @date 2025-01-30
 *
 * @license Proprietary
 *
 * Copyright (c) 2025, Esmail Abdul Fattah, KAUST.
 * All rights reserved.
 *
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification 
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software 
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission 
 * notice are included in all copies or substantial portions of the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE 
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

 #ifndef _STILES_TILE_H_
 #define _STILES_TILE_H_
 
 #if defined( _WIN32 ) || defined( _WIN64 )
 typedef __int64 int64_t;
 #else
 #include <inttypes.h>
 #endif
  
#define sTilesByte          0
#define sTilesInteger       1
#define sTilesRealFloat     2
#define sTilesRealDouble    3
#define sTilesComplexFloat  4
#define sTilesComplexDouble 5

 static inline int stiles_element_size(int type)
{
    switch(type) {
    case sTilesByte:          return          1;
    case sTilesInteger:       return   sizeof(int);
    case sTilesRealFloat:     return   sizeof(float);
    case sTilesRealDouble:    return   sizeof(double);
    case sTilesComplexFloat:  return 2*sizeof(float);
    case sTilesComplexDouble: return 2*sizeof(double);
    default:
        fprintf(stderr, "stiles_element_size: invalide type parameter\n");
        return -1;
    }
}

 /**
  * Computes the address of an element in a tiled matrix.
  *
  * @param[in] A      Descriptor of the tiled matrix.
  * @param[in] m      Row index within the tile.
  * @param[in] n      Column index within the tile.
  *
  * @return Pointer to the computed address within the tiled matrix.
  */
//  inline static void *stile_getaddr(TilesDescriptor A, int m, int n)
//  {
//      size_t mm = m + A.i / A.mb;
//      size_t nn = n + A.j / A.nb;
//      size_t eltsize = stiles_element_size(A.dtyp);
//      size_t offset = 0;
 
//      if (mm < (size_t)(A.lm1)) {
//          if (nn < (size_t)(A.ln1))
//              offset = A.bsiz * (mm + (size_t)A.lm1 * nn);
//          else
//              offset = A.A12 + ((size_t)A.mb * (A.ln % A.nb) * mm);
//      }
//      else {
//          if (nn < (size_t)(A.ln1))
//              offset = A.A21 + ((size_t)A.nb * (A.lm % A.mb) * nn);
//          else
//              offset = A.A22;
//      }
 
//      return (void*)((char*)A.B + (offset * eltsize));
//  }
 

 #define BLKLDD(A, k) ( ((k) + (A).i / (A).mb) < (A).lm1 ? (A).mb : (A).lm % (A).mb )
 #define BLKLDD_A(A, k) ( ((k) + (A).i / (A).mb) < (A).lm1 ? (A).mb : (A).lm % (A).mb )
 #define BLKLDD_B(B, k) ( ((k) + (B).i / (B).mb) < (B).lm1 ? (B).mb : (B).lm % (B).mb )

 #define A(m,n) BLKADDR(A, double, m, n)
 #define B(m,n) BLKADDR(B, double, m, n)
 //#define BLKADDRS(B, type, m, n)  (type *)stile_getaddr(B, m, n)
 
 
//  inline static void *stiles_getaddr_solve(TilesDescriptor A, int m, int n)
//  {
//     size_t eltsize = stiles_element_size(A.dtyp);
//     size_t offset = (size_t)(m * A.nb) + (size_t)(n * A.nb * A.m);
//     return (void*)((char*)A.B + (offset*eltsize) );

//  }
//  #define BLKADDR_SOLVE(A, type, m, n)  (type *)stiles_getaddr_solve(A, m, n)
//  #define GET_BLOCK(m,n) BLKADDR_SOLVE(B, double, m, n)
//  #define C(m,n) BLKADDRS(B, double, m, n)
 
#endif

/*

 #ifndef _STILES_TILE_H_
 #define _STILES_TILE_H_
 
 #if defined( _WIN32 ) || defined( _WIN64 )
 typedef __int64 int64_t;
 #else
 #include <inttypes.h>
 #endif
 
 #include "../include/descriptor.h"
 
 inline static void *stile_getaddr(TilesDescriptor A, int m, int n)
 {
     size_t mm = m + A.i / A.mb;
     size_t nn = n + A.j / A.nb;
     size_t eltsize = stiles_element_size(A.dtyp);
     size_t offset = 0;
 
     if (mm < (size_t)(A.lm1)) {
         if (nn < (size_t)(A.ln1))
             offset = A.bsiz * (mm + (size_t)A.lm1 * nn);
         else
             offset = A.A12 + ((size_t)A.mb * (A.ln % A.nb) * mm);
     }
     else {
         if (nn < (size_t)(A.ln1))
             offset = A.A21 + ((size_t)A.nb * (A.lm % A.mb) * nn);
         else
             offset = A.A22;
     }
 
     return (void*)((char*)A.B + (offset * eltsize));
 }
 
 #define BLKLDD(A, k) ( ((k) + (A).i / (A).mb) < (A).lm1 ? (A).mb : (A).lm % (A).mb )
 #define A(m,n) BLKADDR(A, double, m, n)
 #define B(m,n) BLKADDR(B, double, m, n)
 #define BLKADDRS(B, type, m, n)  (type *)stile_getaddr(B, m, n)
 #define C(m,n) BLKADDRS(B, double, m, n)
 
 #endif
 



*/
