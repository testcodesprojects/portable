/**
 * @file allocate.h
 *
 * Header file for memory allocation routines in the sTiles framework.
 * These routines provide interfaces for shared and private memory management,
 * tailored to tiled matrix computations in high-performance linear algebra.
 *
 * Redesigned by:
 * - Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST).
 *
 * Originally developed as part of the PLASMA project by:
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * This file declares the following functions:
 * - stiles_shared_alloc: Allocates shared memory for tiled computations.
 * - stiles_shared_free: Frees shared memory.
 * - stiles_private_alloc: Allocates private memory for tiled computations.
 * - stiles_private_free: Frees private memory.
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

#ifndef _STILES_ALLOCATE_H_
#define _STILES_ALLOCATE_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocates shared memory for tiled computations.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] size   Size of the memory to allocate (in elements).
 * @param[in] type   Data type of the elements to allocate.
 *
 * @return Pointer to the allocated shared memory, or NULL on failure.
 */
void *stiles_shared_alloc(stiles_context_t *stile, size_t size, int type);

/**
 * Frees shared memory previously allocated by stiles_shared_alloc.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] ptr    Pointer to the memory to free.
 */
void stiles_shared_free(stiles_context_t *stile, void *ptr);

/**
 * Allocates private memory for tiled computations.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] size   Size of the memory to allocate (in elements).
 * @param[in] type   Data type of the elements to allocate.
 *
 * @return Pointer to the allocated private memory, or NULL on failure.
 */
void *stiles_private_alloc(stiles_context_t *stile, size_t size, int type);

/**
 * Frees private memory previously allocated by stiles_private_alloc.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] ptr    Pointer to the memory to free.
 */
void stiles_private_free(stiles_context_t *stile, void *ptr);

#ifdef __cplusplus
}
#endif

#endif
