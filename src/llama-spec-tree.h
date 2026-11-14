#pragma once

#include <cstdint>
#include <vector>

// Rows of one speculative batch that form a tree: row 0 continues the sequence, every other row continues its parent.
// Parents come before children, and a child follows its parent directly or after a finished sibling subtree
// (depth-first order). Set with llama_spec_tree_set before the decode, resolved with llama_spec_tree_keep after it.
struct llama_spec_tree {
    std::vector<int32_t>  parent; // parent row, -1 for row 0
    std::vector<int32_t>  depth;  // rows between a row and row 0
    std::vector<uint32_t> cells;  // attention cache cell of each row, known once the batch is applied
    int32_t pos0  = 0;            // position of row 0
    int32_t epoch = 0;            // counts the trees set; tags the recurrent update log of the batch
    bool    armed = false;        // applies to the next decode with parent.size() tokens

    bool active(uint32_t n_tokens) const {
        return armed && parent.size() == n_tokens;
    }

    // is row a an ancestor of row b, or b itself
    bool is_anc(int32_t a, int32_t b) const {
        while (b > a) {
            b = parent[b];
        }
        return a == b;
    }
};
