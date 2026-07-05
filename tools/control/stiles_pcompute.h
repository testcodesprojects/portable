
#include "../include/stiles_process.h"

void stiles_pdpotrf (stiles_context_t *stile);
void stiles_pdtrtri (stiles_context_t *stile);
void stiles_pdtrsm  (stiles_context_t *stile);


// Top-level dispatchers: pick the right leaf kernel based on factorization
// variant, tile_type_mode, nrhs, and whether pre-collected solve task lists
// are available. Renamed from stiles_pdtrsm_forward/_backward to make the
// dispatch role explicit and to break the namespace shadowing with the
// sTiles::Process::stiles_pdtrsm_forward/_backward leaf-kernel fallbacks.
void stiles_pdtrsm_forward_dispatch(stiles_context_t *stile);
void stiles_pdtrsm_backward_dispatch(stiles_context_t *stile);

void stiles_pdtrsm_forward_idx(stiles_context_t *stile);
void stiles_pdtrsm_backward_idx(stiles_context_t *stile);



