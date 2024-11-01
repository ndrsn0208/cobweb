#ifndef COBWEB_CONTINUOUS_NODE_H
#define COBWEB_CONTINUOUS_NODE_H
#define EIGEN_USE_THREADS

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/stl/tuple.h>
#include <vector>
#include <tuple>
#include <cmath>
#include <variant>
#include <iostream>
#include "helper.h"
#include <optional>


class CobwebContinuousTree;

class CobwebContinuousNode {
public:
    CobwebContinuousTree *tree;
    CobwebContinuousNode *parent;
    std::vector<CobwebContinuousNode *> children;

    float count;
    int label_size;
    Eigen::VectorXf mean;
    Eigen::VectorXf sum_sq;
    int node_id;
    // Eigen::VectorXf labels;
    // do a sparse vector for labels
    // Eigen::SparseVector<float> labels;


    CobwebContinuousNode(int size, int label_size);
    CobwebContinuousNode(CobwebContinuousNode *otherNode);
    int depth();
    void increment_counts(const Eigen::VectorXf &instance, int label);
    void update_counts_from_node(CobwebContinuousNode *node);
    bool is_exact_match(const Eigen::VectorXf &instance, int label);
    size_t _hash();
    std::string __str__();

    std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::SparseVector<float>> mean_var();
    std::tuple<Eigen::VectorXf, Eigen::VectorXf> mean_var_new(const Eigen::VectorXf &instance);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::SparseVector<float>> mean_var_insert(const Eigen::VectorXf &instance, int label);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf> mean_var_merge(CobwebContinuousNode *other, const Eigen::VectorXf &instance);

    float pu_for_insert(CobwebContinuousNode *child, const Eigen::VectorXf &instance);
    float pu_for_new(const Eigen::VectorXf &instance);
    float pu_for_merge(CobwebContinuousNode *best1, CobwebContinuousNode *best2, const Eigen::VectorXf &instance);
    float pu_for_split(CobwebContinuousNode *best);

    std::tuple<float, int> get_best_operation(const Eigen::VectorXf &instance, CobwebContinuousNode *best1, CobwebContinuousNode *best2, float best1_pu);
    std::tuple<float, CobwebContinuousNode *, CobwebContinuousNode *> two_best_children(const Eigen::VectorXf &instance, int label);

    float log_prob(const Eigen::VectorXf &instance);
    float log_prob_class_given_instance(const Eigen::VectorXf &instance);

    Eigen::ArrayXf get_linked_var();

    std::vector<float> log_prob_children_given_instance(const Eigen::VectorXf &instance);
    const Eigen::VectorXf& predict_mean(const Eigen::VectorXf &instance);

    // std::string concept_hash();
    // std::string pretty_print(int depth = 0);
    // int depth();
    // bool is_parent(CobwebContinuousNode *otherConcept);
    // int num_concepts();

    // std::string avcounts_to_json();
    // std::string ser_avcounts();
    // std::string a_count_to_json();
    // std::string sum_n_logn_to_json();
    // std::string dump_json();
    std::string output_json();
};

#endif // COBWEB_CONTINUOUS_NODE_H
