#pragma once
// SPDX-License-Identifier: MIT
// Part of etree — https://github.com/NickAlger/ellipsoid_tree

// AABB tree over a set of leaf boxes, stored in an implicit complete-binary-
// heap layout: node b has children 2b+1 and 2b+2, and a tree with n leaves
// has exactly 2n-1 nodes. The build splits each node's leaf range so that
// every level is full except the last, which is filled left to right — this
// is what makes the heap indexing valid.
//
// Traversal is visitor-based: visit() takes a node predicate ("does the query
// overlap this node box?") and a leaf callback (return false to stop early).
// Concrete query types are two-line predicates on top of it; the convenience
// point/box/ball queries below are exactly that. visit_pairs() and
// visit_self_pairs() implement dual-tree simultaneous descent for
// tree-vs-tree collision queries.
//
// Node boxes and structure are publicly inspectable (node_lo/node_hi/...),
// both for the dual traversal and for visualization tooling.

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "etree/geometry.hpp"

namespace etree {

namespace detail {

inline unsigned int power_of_two_floor( unsigned int x )
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x ^ (x >> 1);
}

// Number of leaves in the left subtree of a complete binary tree with n
// leaves (every level full except the last, which is filled left to right).
inline int heap_left_size( int n )
{
    const int n_full      = static_cast<int>(power_of_two_floor(static_cast<unsigned int>(n)));
    const int n_full_left = n_full / 2;
    const int n_extra     = n - n_full;
    return std::min(n_full_left + n_extra, n_full);
}

} // end namespace detail

class AABBTree
{
public:
    AABBTree() = default;

    AABBTree( const Eigen::Ref<const Eigen::MatrixXd>& leaf_lo,
              const Eigen::Ref<const Eigen::MatrixXd>& leaf_hi )
    {
        build(leaf_lo, leaf_hi);
    }

    void build( const Eigen::Ref<const Eigen::MatrixXd>& leaf_lo,
                const Eigen::Ref<const Eigen::MatrixXd>& leaf_hi )
    {
        if ( leaf_lo.rows() != leaf_hi.rows() || leaf_lo.cols() != leaf_hi.cols() )
        {
            throw std::invalid_argument("etree::AABBTree::build: leaf_lo and leaf_hi shapes differ");
        }
        dim_        = static_cast<int>(leaf_lo.rows());
        num_leaves_ = static_cast<int>(leaf_lo.cols());

        if ( num_leaves_ == 0 )
        {
            num_nodes_ = 0;
            lo_.resize(dim_, 0);
            hi_.resize(dim_, 0);
            i2e_.resize(0);
            return;
        }

        num_nodes_ = 2 * num_leaves_ - 1;
        lo_.resize(dim_, num_nodes_);
        hi_.resize(dim_, num_nodes_);
        i2e_ = Eigen::VectorXi::Constant(num_nodes_, -1);

        std::vector<int> perm(num_leaves_);
        std::iota(perm.begin(), perm.end(), 0);

        // BFS over leaf ranges; children are appended in order, so the list
        // index of each range is its heap node id.
        struct Range { int start; int stop; };
        std::vector<Range> ranges;
        ranges.reserve(num_nodes_);
        ranges.push_back(Range{0, num_leaves_});

        for ( int node = 0; node < num_nodes_; ++node )
        {
            const Range r = ranges[node];

            Eigen::VectorXd blo = leaf_lo.col(perm[r.start]);
            Eigen::VectorXd bhi = leaf_hi.col(perm[r.start]);
            for ( int ii = r.start + 1; ii < r.stop; ++ii )
            {
                blo = blo.cwiseMin(leaf_lo.col(perm[ii]));
                bhi = bhi.cwiseMax(leaf_hi.col(perm[ii]));
            }
            lo_.col(node) = blo;
            hi_.col(node) = bhi;

            if ( r.stop - r.start == 1 )
            {
                i2e_(node) = perm[r.start];
            }
            else
            {
                int axis = 0;
                (bhi - blo).maxCoeff(&axis);

                std::sort(perm.begin() + r.start, perm.begin() + r.stop,
                          [&](int aa, int bb)
                          {
                              return leaf_lo(axis, aa) + leaf_hi(axis, aa)
                                   < leaf_lo(axis, bb) + leaf_hi(axis, bb);
                          });

                const int mid = r.start + detail::heap_left_size(r.stop - r.start);
                ranges.push_back(Range{r.start, mid});
                ranges.push_back(Range{mid, r.stop});
            }
        }
    }

    int  dim() const        { return dim_; }
    int  num_leaves() const { return num_leaves_; }
    int  num_nodes() const  { return num_nodes_; }
    bool empty() const      { return num_nodes_ == 0; }

    bool is_leaf( int node ) const          { return 2 * node + 1 >= num_nodes_; }
    int  left_child( int node ) const       { return 2 * node + 1; }
    int  right_child( int node ) const      { return 2 * node + 2; }
    int  external_index( int node ) const   { return i2e_(node); } // -1 for internal nodes

    Eigen::Ref<const Eigen::VectorXd> node_lo( int node ) const { return lo_.col(node); }
    Eigen::Ref<const Eigen::VectorXd> node_hi( int node ) const { return hi_.col(node); }
    const Eigen::MatrixXd& node_lo_matrix() const { return lo_; }
    const Eigen::MatrixXd& node_hi_matrix() const { return hi_; }

    // Depth-first traversal. overlaps_node(lo, hi) decides whether to descend
    // into a node; on_leaf(external_index) is called for each overlapping
    // leaf and stops the whole traversal by returning false. Returns false
    // iff the traversal was stopped early.
    template <class NodePredicate, class LeafCallback>
    bool visit( NodePredicate&& overlaps_node, LeafCallback&& on_leaf ) const
    {
        if ( num_nodes_ == 0 )
        {
            return true;
        }
        std::vector<int> stack;
        stack.reserve(64);
        stack.push_back(0);
        while ( !stack.empty() )
        {
            const int node = stack.back();
            stack.pop_back();
            if ( !overlaps_node(lo_.col(node), hi_.col(node)) )
            {
                continue;
            }
            if ( is_leaf(node) )
            {
                if ( !on_leaf(i2e_(node)) )
                {
                    return false;
                }
            }
            else
            {
                stack.push_back(2 * node + 1);
                stack.push_back(2 * node + 2);
            }
        }
        return true;
    }

    // Convenience queries (exact for the leaf boxes themselves).
    std::vector<int> point_collisions( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        std::vector<int> out;
        visit([&]( const auto& lo, const auto& hi )
              { return (lo.array() <= p.array()).all() && (p.array() <= hi.array()).all(); },
              [&]( int e ) { out.push_back(e); return true; });
        return out;
    }

    std::vector<int> box_collisions( const Eigen::Ref<const Eigen::VectorXd>& qlo,
                                     const Eigen::Ref<const Eigen::VectorXd>& qhi ) const
    {
        std::vector<int> out;
        visit([&]( const auto& lo, const auto& hi )
              { return (qlo.array() <= hi.array()).all() && (lo.array() <= qhi.array()).all(); },
              [&]( int e ) { out.push_back(e); return true; });
        return out;
    }

    std::vector<int> ball_collisions( const Eigen::Ref<const Eigen::VectorXd>& center,
                                      double radius ) const
    {
        std::vector<int> out;
        visit([&]( const auto& lo, const auto& hi )
              {
                  Eigen::VectorXd closest = center.cwiseMax(lo.matrix()).cwiseMin(hi.matrix());
                  return (closest - center).squaredNorm() <= radius * radius;
              },
              [&]( int e ) { out.push_back(e); return true; });
        return out;
    }

private:
    int             dim_        = 0;
    int             num_leaves_ = 0;
    int             num_nodes_  = 0;
    Eigen::MatrixXd lo_;  // shape (dim, num_nodes)
    Eigen::MatrixXd hi_;  // shape (dim, num_nodes)
    Eigen::VectorXi i2e_; // leaf node -> external object index; -1 for internal
};


// Dual-tree simultaneous descent over two trees: overlaps_pair(loA, hiA, loB,
// hiB) prunes node pairs; on_leaf_pair(ext_a, ext_b) is called for surviving
// leaf pairs and stops everything by returning false. A far-apart pair of
// clusters costs a single box test instead of one traversal per element.
template <class PairPredicate, class LeafPairCallback>
bool visit_pairs( const AABBTree& A, const AABBTree& B,
                  PairPredicate&& overlaps_pair, LeafPairCallback&& on_leaf_pair )
{
    if ( A.empty() || B.empty() )
    {
        return true;
    }
    std::vector<std::pair<int, int>> stack;
    stack.reserve(128);
    stack.emplace_back(0, 0);
    while ( !stack.empty() )
    {
        const int a = stack.back().first;
        const int b = stack.back().second;
        stack.pop_back();

        if ( !overlaps_pair(A.node_lo(a), A.node_hi(a), B.node_lo(b), B.node_hi(b)) )
        {
            continue;
        }
        const bool a_leaf = A.is_leaf(a);
        const bool b_leaf = B.is_leaf(b);
        if ( a_leaf && b_leaf )
        {
            if ( !on_leaf_pair(A.external_index(a), B.external_index(b)) )
            {
                return false;
            }
        }
        else
        {
            // Descend into the node with the larger box (sum of extents).
            const double size_a = a_leaf ? -1.0 : (A.node_hi(a) - A.node_lo(a)).sum();
            const double size_b = b_leaf ? -1.0 : (B.node_hi(b) - B.node_lo(b)).sum();
            if ( size_a >= size_b )
            {
                stack.emplace_back(A.left_child(a), b);
                stack.emplace_back(A.right_child(a), b);
            }
            else
            {
                stack.emplace_back(a, B.left_child(b));
                stack.emplace_back(a, B.right_child(b));
            }
        }
    }
    return true;
}

// Self-collision variant: visits each unordered pair of distinct leaves at
// most once. Expanding a diagonal pair (n, n) into (L,L), (R,R), (L,R) — and
// never (R,L) — is what guarantees the once-only property.
template <class PairPredicate, class LeafPairCallback>
bool visit_self_pairs( const AABBTree& T,
                       PairPredicate&& overlaps_pair, LeafPairCallback&& on_leaf_pair )
{
    if ( T.empty() )
    {
        return true;
    }
    std::vector<std::pair<int, int>> stack;
    stack.reserve(128);
    stack.emplace_back(0, 0);
    while ( !stack.empty() )
    {
        const int a = stack.back().first;
        const int b = stack.back().second;
        stack.pop_back();

        if ( a == b )
        {
            if ( !T.is_leaf(a) )
            {
                stack.emplace_back(T.left_child(a),  T.left_child(a));
                stack.emplace_back(T.right_child(a), T.right_child(a));
                stack.emplace_back(T.left_child(a),  T.right_child(a));
            }
            continue;
        }

        if ( !overlaps_pair(T.node_lo(a), T.node_hi(a), T.node_lo(b), T.node_hi(b)) )
        {
            continue;
        }
        const bool a_leaf = T.is_leaf(a);
        const bool b_leaf = T.is_leaf(b);
        if ( a_leaf && b_leaf )
        {
            if ( !on_leaf_pair(T.external_index(a), T.external_index(b)) )
            {
                return false;
            }
        }
        else
        {
            const double size_a = a_leaf ? -1.0 : (T.node_hi(a) - T.node_lo(a)).sum();
            const double size_b = b_leaf ? -1.0 : (T.node_hi(b) - T.node_lo(b)).sum();
            if ( size_a >= size_b )
            {
                stack.emplace_back(T.left_child(a), b);
                stack.emplace_back(T.right_child(a), b);
            }
            else
            {
                stack.emplace_back(a, T.left_child(b));
                stack.emplace_back(a, T.right_child(b));
            }
        }
    }
    return true;
}

} // end namespace etree
