#ifndef COBWEB_CONTINUOUS_TREE_H
#define COBWEB_CONTINUOUS_TREE_H
#define EIGEN_USE_THREADS

#include <queue>
#include <nanobind/eigen/dense.h>
#include <Eigen/Dense>
#include <nanobind/stl/tuple.h>
#include <Eigen/Sparse>
#include "cobweb_continuous_node.h"
#include "helper.h"
#include <variant>
#include <optional>


namespace nb = nanobind;

#define BEST 0
#define NEW 1
#define MERGE 2
#define SPLIT 3

class CobwebContinuousTree {
public:
    int size;
    int max_num_instances;
    int covar_type;
    int label_size;
    int covar_from;
    int embed_dim;
    bool learn_embedding;
    bool use_onehot;

    int node_count;

    float learning_rate;

    Eigen::VectorXf prior_var;

    // create a embedding matrix
    Eigen::MatrixXf embedding_matrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor> label_space;
    



    CobwebContinuousNode *root;

    // covar_type: 1=diag
    // covar_from: 1=self, 2=parent
    CobwebContinuousTree(int size, int label_size, int max_num_instances, int embed_size, float learning_rate, int covar_type, int covar_from, bool learn_embedding, bool use_onehot);
    Eigen::VectorXf get_token_embedding(int token_id);
    Eigen::MatrixXf get_embedding_matrix();
    // std::tuple<bool, bool> binary_and_balance(const CobwebContinuousNode *root);
    bool isBinary(CobwebContinuousNode *node);
    int checkHeight(CobwebContinuousNode* node, bool &isBalanced);
    std::tuple<Eigen::MatrixXf, Eigen::VectorXf> get_weights_and_bias(int hidden_dim);
    std::tuple<Eigen::VectorXf, double> get_decision_boundary(CobwebContinuousNode *node1, CobwebContinuousNode *node2);
    bool isBinaryAndBalanced(CobwebContinuousNode *node);
    Eigen::VectorXf cosine_similarity(const Eigen::VectorXf &output_logits);
    Eigen::VectorXf weighted_state(const std::vector<CobwebContinuousNode *> &path, const std::vector<int> &levels);
    CobwebContinuousNode* ifit(const Eigen::VectorXf &instance, int label);
    CobwebContinuousNode* ifit_helper(const Eigen::VectorXf &instance, int label);
    CobwebContinuousNode* cobweb(const Eigen::VectorXf &instance, int label);
    void set_embedding_matrix(const Eigen::MatrixXf &embedding_matrix);

    Eigen::VectorXf score_reduction_path();
    Eigen::MatrixXf get_clusters(int n);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::VectorXf> predict(const Eigen::VectorXf &instance, int max_nodes, bool greedy);
    Eigen::VectorXf predict_helper(const Eigen::VectorXf &instance, int max_nodes, bool greedy);
    std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::VectorXf> predict_label(const Eigen::VectorXf &instance, int max_nodes, bool greedy);
    float log_prob(const Eigen::VectorXf &instance, int max_nodes, bool greedy);
    int num_nodes();

    std::string __str__();
    // std::string dump_json()
    // std::string load_json()
    void clear();

    Eigen::VectorXf compute_var(const Eigen::VectorXf& meanSq, const float count);
    float compute_score(const Eigen::VectorXf& child_mean,
            const Eigen::VectorXf& child_var, 
            const Eigen::SparseVector<float>& child_p_label,
            const Eigen::VectorXf& parent_mean,
            const Eigen::VectorXf& parent_var,
            const Eigen::SparseVector<float>& parent_p_label);

};

#endif // COBWEB_CONTINUOUS_TREE_H
