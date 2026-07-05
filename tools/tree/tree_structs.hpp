#ifndef STILES_TREE_STRUCTS_HPP
#define STILES_TREE_STRUCTS_HPP

/** ****************************************************************************
 * Structure representing a single node within a tree leaf
 ******************************************************************************/
typedef struct NodeLeaf {
    int surviving_level; /**< Level at which this node survives */
    int index;           /**< Node index */
    int level;           /**< Current level of the node */
    int leafheight;      /**< Height of the leaf */
    int leafwidth;       /**< Width of the leaf */
    int dirty;           /**< 1 iff this rank wrote to x[] for the current factorization;
                              cleared by the reducer (case 7/10) after consuming.
                              Used to skip no-op DGEADDs on ranks with empty slices. */
    double *x;           /**< Data buffer for GEMM operations */
} NodeLeaf;

/** ****************************************************************************
 * Structure representing a tree of leaves for symbolic factorization
 ******************************************************************************/
typedef struct TreeLeaf {
    NodeLeaf *nodes;       /**< Array of nodes */
    int *max_nodes;        /**< Maximum number of nodes per level */
    int *gold_number;      /**< Array of "gold" numbers for calculations */
    int *half_gold;        /**< Auxiliary array for intermediate values */
    int silver_number;     /**< "Silver" number for calculations */
    int num_splits;
    int num_tasks;
    int num_nodes;         /**< Total number of nodes */
    int counter_nodes;     /**< Counter of nodes at each level */
    int max_levels;        /**< Maximum number of levels in the tree */
    int *dependency;       /**< Dependency vector */
} TreeLeaf;

#endif  // STILES_TREE_STRUCTS_HPP
