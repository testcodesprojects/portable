/**
 * @file stiles_threadsetting.h
 *
 * Redesigned by:
 * - Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST).
 *
 * Originally developed as part of the PLASMA project by:
 * - University of Tennessee
 * - University of California, Berkeley
 * - University of Colorado, Denver
 *
 * @version 1.0.0
 * @author Esmail Abdul Fattah
 * @original_author Azzam Haidar
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

#ifndef _STILES_THREADSETTING_H_
#define _STILES_THREADSETTING_H_

#ifdef __cplusplus
extern "C" {
#endif
/***************************************************************************//**
 *  Internal routines
/***************************************************************************/
void stiles_setlapack_multithreads(int numthreads);
void stiles_setlapack_sequential(stiles_context_t *stile);
void stiles_psetlapack_numthreads(stiles_context_t *stile);
/***************************************************************************/
#ifdef __cplusplus
}
#endif

#endif
