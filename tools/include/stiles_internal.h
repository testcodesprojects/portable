/**
 * @file stiles_internal.h
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



#ifndef _STILES_H_
#define _STILES_H_

#include <stdbool.h>
//#include "../core_blas_dense/stiles_types.hpp"
#include "../include/stiles.h"
#include "../include/stiles_process.h"

typedef struct stiles_request_t {
    int status; /**< 0 or appropriate error code */
} STILES_request;

typedef struct stiles_sequence_t {
    int     status;         /**< 0 or appropriate error code       */
    STILES_request *request;        /**< failed request                                 */
} STILES_sequence;

//for request:
//STILES_REQUEST_INITIALIZER



#define STILES_REQUEST_INITIALIZER {0}

#define STILES_FALSE 0
#define STILES_TRUE  1

#define STILES_WARNINGS   1
#define STILES_ERRORS     2
#define STILES_AUTOTUNING 3
#define STILES_DAG        4

#define STILES_CONCURRENCY      1
#define STILES_TILE_SIZE_MACRO  2
#define STILES_INNER_BLOCK_SIZE 3
#define STILES_SCHEDULING_MODE  4
#define STILES_HOUSEHOLDER_MODE 5
#define STILES_HOUSEHOLDER_SIZE 6
#define STILES_TRANSLATION_MODE 7
#define STILES_TNTPIVOTING_MODE 8
#define STILES_TNTPIVOTING_SIZE 9
#define STILES_EV_WSMODE        10
#define STILES_EV_TASKNB        11
#define STILES_EV_SMLSZE        12
#define STILES_STATIC_SCHEDULING  1

#define STILES_E_DEFAULT  0
#define STILES_E_RCM      1
#define STILES_E_ND       2
#define STILES_E_AND      3
#define STILES_E_AMD      4
#define STILES_E_RCMD     5
#define STILES_T_DEFAULT  6
#define STILES_T_RCM      7
#define STILES_T_ND       8
#define STILES_T_AND      9
#define STILES_T_AMD      10
#define STILES_T_RCMD     11

#define STILES_CHOL_TYPE_P_SPARSE_DENSE 0
#define STILES_CHOL_TYPE_P_DENSE_DENSE  1
#define STILES_CHOL_TYPE_S_DENSE        2

#endif
