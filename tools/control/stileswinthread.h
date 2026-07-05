/**
 * @file stileswinthread.h
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
 * @original_author Piotr Luszczek
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


#ifndef STILESWINTHREAD_H
#define STILESWINTHREAD_H
#include <windows.h>

typedef struct pthread_s {
  HANDLE hThread;
  unsigned int uThId;
} pthread_t;

typedef HANDLE pthread_mutex_t;
typedef int pthread_mutexattr_t;
typedef int pthread_attr_t;
typedef int pthread_condattr_t;

typedef struct pthread_cond_s {
  HANDLE hSem;
  HANDLE hEvt;
  CRITICAL_SECTION cs;
  int waitCount; /* waiting thread counter */
} pthread_cond_t;

typedef int pthread_attr_t;

#define PTHREAD_MUTEX_INITIALIZER ((pthread_mutex_t) -1)

#define PTHREAD_SCOPE_SYSTEM 1

#define STILES_DLLPORT
#define STILES_CDECL __cdecl

STILES_DLLPORT pthread_t STILES_CDECL pthread_self(void);
STILES_DLLPORT int STILES_CDECL pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t * attr);
STILES_DLLPORT int STILES_CDECL pthread_mutex_destroy(pthread_mutex_t *mutex);
STILES_DLLPORT int STILES_CDECL pthread_mutex_lock(pthread_mutex_t *mutex);
STILES_DLLPORT int STILES_CDECL pthread_mutex_trylock(pthread_mutex_t *mutex);
STILES_DLLPORT int STILES_CDECL pthread_mutex_unlock(pthread_mutex_t *mutex);
STILES_DLLPORT int STILES_CDECL pthread_attr_init(pthread_attr_t *attr);
STILES_DLLPORT int STILES_CDECL pthread_attr_destroy(pthread_attr_t *attr);
STILES_DLLPORT int STILES_CDECL pthread_attr_setscope(pthread_attr_t *attr, int scope);
STILES_DLLPORT int STILES_CDECL pthread_create(pthread_t *tid, const pthread_attr_t *attr, void *(*start) (void *), void *arg);
STILES_DLLPORT int STILES_CDECL pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr);
STILES_DLLPORT int STILES_CDECL pthread_cond_destroy(pthread_cond_t *cond);
STILES_DLLPORT int STILES_CDECL pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
STILES_DLLPORT int STILES_CDECL pthread_cond_broadcast(pthread_cond_t *cond);
STILES_DLLPORT int STILES_CDECL pthread_join(pthread_t thread, void **value_ptr);
STILES_DLLPORT int STILES_CDECL pthread_equal(pthread_t thread1, pthread_t thread2);

STILES_DLLPORT int STILES_CDECL pthread_setconcurrency (int);

STILES_DLLPORT unsigned int STILES_CDECL pthread_self_id(void);

#endif
