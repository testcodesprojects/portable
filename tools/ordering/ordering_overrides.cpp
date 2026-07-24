/**
 * @file ordering_overrides.cpp
 * @brief User-supplied overrides for ordering: explicit permutation
 *        and forced ND partition sizes.
 *
 * Owns the per-group storage for two ordering knobs the user can override
 * before sTiles_create:
 *   - user permutation     (sTiles_set/clear_user_permutation, set/clear_all)
 *   - forced ND partitions (sTiles_set/clear_partition_sizes)
 *
 * The ordering module reads these via the `sTiles::get_*` accessors
 * declared in stiles_ordering.hpp.
 */

#include <map>
#include <vector>

#include "../common/stiles_logger.hpp"

// Defined in tools/process/process.cpp. Toggle via sTiles_expert_user().
extern bool expert_mode_enabled;

namespace {

struct UserPermutation {
    int* perm = nullptr;
    int size = 0;
    bool force = false;
};
std::map<int, UserPermutation> user_permutations;

struct ForcedPartitionSizes {
    int p1_size = -1;
    int p2_size = -1;
    int sep_size = -1;
    bool force = false;
};
std::map<int, ForcedPartitionSizes> forced_partition_sizes;

} // namespace

namespace sTiles {

const int* get_user_permutation(int group_index) {
    auto it = user_permutations.find(group_index);
    return (it != user_permutations.end()) ? it->second.perm : nullptr;
}

int get_user_permutation_size(int group_index) {
    auto it = user_permutations.find(group_index);
    return (it != user_permutations.end()) ? it->second.size : 0;
}

bool get_user_permutation_force(int group_index) {
    auto it = user_permutations.find(group_index);
    return (it != user_permutations.end()) ? it->second.force : false;
}

bool get_forced_partition_sizes(int group_index, int& p1, int& p2, int& sep) {
    auto it = forced_partition_sizes.find(group_index);
    if (it != forced_partition_sizes.end() && it->second.force) {
        p1 = it->second.p1_size;
        p2 = it->second.p2_size;
        sep = it->second.sep_size;
        return true;
    }
    return false;
}

// Internal (non-expert-gated) variant used by the preprocessing pipeline to
// donate an already-validated permutation from an identical earlier graph.
void set_user_permutation_internal(int group_index, const int* perm, int n) {
    if (group_index < 0 || !perm || n <= 0) return;
    auto it = user_permutations.find(group_index);
    if (it != user_permutations.end()) {
        delete[] it->second.perm;
        user_permutations.erase(it);
    }
    UserPermutation& up = user_permutations[group_index];
    up.perm = new int[n];
    std::copy(perm, perm + n, up.perm);
    up.size  = n;
    up.force = true;
}

} // namespace sTiles

// C-linkage entry points (declared in tools/include/stiles.h inside extern "C").
extern "C" {

// Forward declarations so impls below can call each other before their definitions.
void sTiles_clear_user_permutation(int group_index);

/**
 * @brief Set a user-provided permutation for ordering for a specific group.
 *
 * @param group_index The group index (0-based).
 * @param perm Pointer to the permutation array (0-based indexing).
 * @param n Size of the permutation array (must match matrix dimension for this group).
 * @param force If true, forces use of this permutation.
 *              If false, this permutation is tested alongside automatic orderings.
 */
void sTiles_set_user_permutation(int group_index, const int* perm, int n, bool force) {
    if (!expert_mode_enabled) {
        return;
    }
    if (group_index < 0) {
        sTiles::Logger::warning("sTiles_set_user_permutation: invalid group_index=", group_index);
        return;
    }

    if (!perm || n <= 0) {
        sTiles::Logger::warning("sTiles_set_user_permutation: invalid arguments (perm=",
                                static_cast<const void*>(perm), ", n=", n, ")");
        return;
    }

    // Validate that perm is a valid permutation
    std::vector<bool> seen(n, false);
    for (int i = 0; i < n; ++i) {
        if (perm[i] < 0 || perm[i] >= n) {
            sTiles::Logger::error("sTiles_set_user_permutation: invalid permutation - ",
                                 "element ", i, " has value ", perm[i],
                                 " (expected 0 <= value < ", n, ")");
            return;
        }
        if (seen[perm[i]]) {
            sTiles::Logger::error("sTiles_set_user_permutation: invalid permutation - ",
                                 "duplicate value ", perm[i], " at position ", i);
            return;
        }
        seen[perm[i]] = true;
    }

    // Clear any existing user permutation for this group
    sTiles_clear_user_permutation(group_index);

    // Allocate and copy the user permutation
    UserPermutation& up = user_permutations[group_index];
    up.perm = new int[n];
    for (int i = 0; i < n; ++i) {
        up.perm[i] = perm[i];
    }
    up.size = n;
    up.force = force;

    sTiles::Logger::info("│ ✓ User permutation set for group ", group_index,
                         " (size=", n, ", force=", force, ")");
}

/**
 * @brief Clear user-provided permutation for a specific group.
 */
void sTiles_clear_user_permutation(int group_index) {
    auto it = user_permutations.find(group_index);
    if (it != user_permutations.end()) {
        if (it->second.perm) {
            delete[] it->second.perm;
        }
        user_permutations.erase(it);
        sTiles::Logger::debug("User permutation cleared for group ", group_index);
    }
}

/**
 * @brief Clear all user-provided permutations.
 */
void sTiles_clear_all_user_permutations(void) {
    for (auto& pair : user_permutations) {
        if (pair.second.perm) {
            delete[] pair.second.perm;
        }
    }
    user_permutations.clear();
    sTiles::Logger::debug("All user permutations cleared");
}

/**
 * @brief Set forced partition sizes for ND ordering for a specific group.
 */
void sTiles_set_partition_sizes(int group_index, int p1_size, int p2_size, int sep_size, bool force) {
    if (group_index < 0) {
        sTiles::Logger::warning("sTiles_set_partition_sizes: invalid group_index=", group_index);
        return;
    }

    if (p1_size < 0 || p2_size < 0 || sep_size < 0) {
        sTiles::Logger::warning("sTiles_set_partition_sizes: invalid partition sizes (",
                                p1_size, ", ", p2_size, ", ", sep_size, ")");
        return;
    }

    ForcedPartitionSizes& fps = forced_partition_sizes[group_index];
    fps.p1_size = p1_size;
    fps.p2_size = p2_size;
    fps.sep_size = sep_size;
    fps.force = force;

    sTiles::Logger::info("│ ✓ Partition sizes set for group ", group_index,
                         " (P1=", p1_size, ", P2=", p2_size, ", Sep=", sep_size, ", force=", force, ")");
}

/**
 * @brief Clear forced partition sizes for a specific group.
 */
void sTiles_clear_partition_sizes(int group_index) {
    auto it = forced_partition_sizes.find(group_index);
    if (it != forced_partition_sizes.end()) {
        forced_partition_sizes.erase(it);
        sTiles::Logger::debug("Partition sizes cleared for group ", group_index);
    }
}

} // extern "C"
