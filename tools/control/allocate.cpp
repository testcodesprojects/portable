
#include <stdlib.h>
#include "common.h"

/***************************************************************************//**
 * Allocates shared memory for tiled computations in the sTiles framework.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] size   Size of the memory to allocate (in elements).
 * @param[in] type   Data type of the elements to allocate.
 *
 * @return Pointer to the allocated shared memory, or NULL on failure.
 ******************************************************************************/
void *stiles_shared_alloc(stiles_context_t *stile, size_t size, int type)
{
    void *memptr;

    size *= stiles_element_size(type);
    if (size <= 0)
        return NULL;

    if ((memptr = malloc(size)) == NULL) {
        sTiles::Logger::error("Failed to allocate ", size, " bytes for a shared buffer.");
        return NULL;
    }
    return memptr;
}

/***************************************************************************//**
 * Frees shared memory previously allocated by stiles_shared_alloc.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] ptr    Pointer to the memory to free.
 ******************************************************************************/
void stiles_shared_free(stiles_context_t *stile, void *ptr)
{
    if (ptr == NULL)    // Redundant check; free() handles NULL.
        return;
    free(ptr);
}

/***************************************************************************//**
 * Allocates private memory for tiled computations in the sTiles framework.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] size   Size of the memory to allocate (in elements).
 * @param[in] type   Data type of the elements to allocate.
 *
 * @return Pointer to the allocated private memory, or NULL on failure.
 ******************************************************************************/
void *stiles_private_alloc(stiles_context_t *stile, size_t size, int type)
{
    void *memptr;

    size *= stiles_element_size(type);
    if (size <= 0)
        return NULL;

    if ((memptr = malloc(size)) == NULL) {
        sTiles::Logger::error("Failed to allocate ", size, " bytes for a private buffer.");
        return NULL;
    }
    return memptr;
}

/***************************************************************************//**
 * Frees private memory previously allocated by stiles_private_alloc.
 *
 * @param[in] stile  Pointer to the STILES context structure.
 * @param[in] ptr    Pointer to the memory to free.
 ******************************************************************************/
void stiles_private_free(stiles_context_t *stile, void *ptr)
{
    if (ptr == NULL)    // Redundant check; free() handles NULL.
        return;
    free(ptr);
}
