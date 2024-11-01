#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"
// include chrono for timing
#include <chrono>

CobwebContinuousTree::CobwebContinuousTree(int size, int label_size, int max_num_instances, int embed_size, float learning_rate,
                                            int covar_type, int covar_from, bool learn_embedding, bool use_onehot)
    : root(nullptr),
      size(size),
      max_num_instances(max_num_instances),
      label_size(label_size),
      covar_type(covar_type),
      embed_dim(embed_size),
      learning_rate(learning_rate),
      learn_embedding(learn_embedding),
      use_onehot(use_onehot),
      //   labels(Eigen::VectorXf::Zero(label_size)),
      label_space(Eigen::SparseMatrix<float>(2 * max_num_instances, label_size)),
      covar_from(covar_from)
{
    this->prior_var = Eigen::VectorXf::Constant(size, 0.05854983152);
    
    this->node_count = 0;

    this->clear();
    
    // initialize the embedding matrix with Kaiming initialization
    if (use_onehot)
    {
        this->embedding_matrix = Eigen::MatrixXf::Identity(label_size, label_size);
        // add one to the entire matrix to avoid zero division
        this->embedding_matrix = this->embedding_matrix + Eigen::MatrixXf::Ones(label_size, label_size);
    }
    else
    {
        this->embedding_matrix = Eigen::MatrixXf::Random(label_size, embed_size);
        // this->embedding_matrix = Eigen::MatrixXf::Random(embed_size, label_size);
        // this->embedding_matrix = this->embedding_matrix / sqrt(embed_size);
    }
    // this->embedding_matrix = Eigen::MatrixXf::Random(label_size, embed_size);
    // this->embedding_matrix = this->embedding_matrix / sqrt(embed_size);
}

CobwebContinuousNode *CobwebContinuousTree::ifit(const Eigen::VectorXf &instance, int label)
{
    return this->ifit_helper(instance, label);
}

void CobwebContinuousTree::set_embedding_matrix(const Eigen::MatrixXf &embedding_matrix)
{
    // check dimension
    if (embedding_matrix.rows() != this->label_size || embedding_matrix.cols() != this->embed_dim)
    {
        throw std::invalid_argument("The dimension of the embedding matrix does not match the label size and embed size");
    }
    this->embedding_matrix = embedding_matrix;
}

Eigen::MatrixXf CobwebContinuousTree::get_embedding_matrix()
{
    // return this->embedding_matrix.transpose();
    return this->embedding_matrix;
}

Eigen::MatrixXf CobwebContinuousTree::get_clusters(int n)
{
    // n is the number of clusters


    std::vector<CobwebContinuousNode *> nodes;
    std::vector<CobwebContinuousNode *> queue;

    // get all nodes then truncate to the first n nodes
    queue.push_back(this->root);
    while (!queue.empty())
    {
        CobwebContinuousNode *current = queue.back();
        queue.pop_back();
        nodes.push_back(current);
        for (auto &child : current->children)
        {
            queue.push_back(child);
        }
    }

    // truncate to the first n nodes
    nodes.resize(n);

    // out shape is 2^k x embed_dim
    Eigen::MatrixXf out = Eigen::MatrixXf::Zero(nodes.size(), this->embed_dim);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        out.row(i) = nodes[i]->mean;
    }

    return out;
}

// std::tuple<bool, bool> CobwebContinuousTree::binary_and_balance(const CobwebContinuousNode *root)
// {
//     if (root->children.empty())
//     {
//         return std::make_tuple(true, true);
//     }

//     bool all_balanced = true;
//     bool all_binary = true;

//     for (auto &child : root->children)
//     {
//         auto [balanced, binary] = binary_and_balance(child);
//         all_balanced = all_balanced && balanced;
//         all_binary = all_binary && binary;
//     }

//     if (root->children.size() == 2)
//     {
//         return std::make_tuple(all_balanced, all_binary);
//     }
//     else
//     {
//         return std::make_tuple(false, all_binary);
//     }
// }

bool CobwebContinuousTree::isBinary(CobwebContinuousNode* node) {
    if (node == nullptr) return true;  // An empty node is considered binary
    if (node->children.size() > 2) return false;  // More than 2 children violates binary property

    // Recursively check all children
    for (CobwebContinuousNode* child : node->children) {
        if (!isBinary(child)) return false;
    }
    return true;
}

// Function to check if the tree is balanced and returns the height of the tree
int CobwebContinuousTree::checkHeight(CobwebContinuousNode* node, bool &isBalanced) {
    if (node == nullptr) return 0;

    int leftHeight = node->children.size() > 0 ? checkHeight(node->children[0], isBalanced) : 0;
    int rightHeight = node->children.size() > 1 ? checkHeight(node->children[1], isBalanced) : 0;

    // Check if the current node is balanced
    if (std::abs(leftHeight - rightHeight) > 1) {
        isBalanced = false;
    }

    // Return the height of the current node
    return 1 + std::max(leftHeight, rightHeight);
}

// Function to determine if a tree is binary and balanced
bool CobwebContinuousTree::isBinaryAndBalanced(CobwebContinuousNode* root) {
    if (root == nullptr) return true;  // An empty tree is considered both binary and balanced

    // Check if the tree is binary
    if (!isBinary(root)) return false;

    // Check if the tree is balanced
    bool isBalanced = true;
    checkHeight(root, isBalanced);
    return isBalanced;
}


CobwebContinuousNode *CobwebContinuousTree::ifit_helper(const Eigen::VectorXf &instance, int label)
{
    // timing the extraction of the embeddings
    if (this->learn_embedding)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // convert token ids into average embeddings
        Eigen::MatrixXf selected_rows(instance.size(), this->embed_dim);
        for (int i = 0; i < instance.size(); ++i)
        {
            // get the vector representation of the token from get_token_embedding
            // selected_rows.row(i) = this->get_token_embedding(instance[i]);
            // // if get_token_embedding returns a zero vector, we can use the embedding matrix directly
            // if (selected_rows.row(i).isZero())
            // {
            //     selected_rows.row(i) = this->embedding_matrix.row(instance[i]);
            // }
            selected_rows.row(i) = this->embedding_matrix.row(instance[i]);
        }
        // Compute the mean by averaging across the rows
        Eigen::VectorXf instance_embedding = selected_rows.colwise().mean();

        // Eigen::MatrixXf selected_columns(this->embed_dim, instance.size());
        // for (int i = 0; i < instance.size(); ++i)
        // {
        //     // Access the column directly from the embedding matrix
        //     selected_columns.col(i) = this->embedding_matrix.col(instance[i]);
        // }

        // // Compute the mean by averaging across the columns
        // Eigen::VectorXf instance_embedding = selected_columns.rowwise().mean();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        // std::cout << "Elapsed time for embedding extraction: " << elapsed_seconds.count() << "s\n";

        return this->cobweb(instance_embedding, label);
    }
    else
    {
        return this->cobweb(instance, label);
    }
}

int CobwebContinuousNode::depth()
{
    int level = 1;
    CobwebContinuousNode *current = this;
    while (current->parent != nullptr)
    {
        level += 1;
        current = current->parent;
    }
    return level;
}

int CobwebContinuousTree::num_nodes()
{
    std::vector<CobwebContinuousNode *> queue;
    queue.push_back(this->root);
    int count = 0;
    while (!queue.empty())
    {
        CobwebContinuousNode *current = queue.back();
        queue.pop_back();
        count += 1;
        for (auto &child : current->children)
        {
            queue.push_back(child);
        }
    }
    return count;
}

Eigen::VectorXf CobwebContinuousTree::get_token_embedding(int token_id)
{
    // search through the tree by collecting all pathes that contains the token_id in the labels (label count > 0)
    // then average the embeddings of the nodes in the pathes weighted by levels

    std::vector<CobwebContinuousNode *> path;
    std::vector<int> levels;

    std::vector<CobwebContinuousNode *> queue;
    queue.push_back(this->root);

    while (!queue.empty())
    {
        CobwebContinuousNode *current = queue.back();
        queue.pop_back();

        // if (current->labels[token_id] > 0)
        // if (current->labels.coeff(token_id) > 0)
        if (this->label_space.coeff(current->node_id, token_id) > 0)
        {
            path.push_back(current);
            levels.push_back(current->depth());

            for (auto &child : current->children)
            {
                // if (child->labels[token_id] > 0)
                // if (child->labels.coeff(token_id) > 0)
                if (this->label_space.coeff(child->node_id, token_id) > 0)
                {
                    queue.push_back(child);
                }
            }
        }
    }

    if (path.empty())
    {
        return Eigen::VectorXf::Zero(this->size);
    }
    return this->weighted_state(path, levels);
}

std::tuple<Eigen::VectorXf, double> CobwebContinuousTree::get_decision_boundary(CobwebContinuousNode *node1, CobwebContinuousNode *node2)
{
    // get the decision boundary between two nodes
    // the decision boundary is the line that is perpendicular to the line connecting the two means
    // and passes through the midpoint of the line connecting the two means

    Eigen::VectorXf mean1 = node1->mean;
    Eigen::VectorXf mean2 = node2->mean;

    Eigen::VectorXf var = node1->get_linked_var();

    // Perform element-wise operations using .array()
    Eigen::VectorXf w = (mean1 - mean2).array() / var.array().square();
    // std::cout << "w: " << w << std::endl;
    // std::cout << "mean1: " << mean1 << std::endl;
    // std::cout << "mean2: " << mean2 << std::endl;

    // Calculate b using element-wise operations and summation
    double b = ((mean2.array().square() - mean1.array().square()) / (2 * var.array().square())).sum();

    // b += log(static_cast<double>(node1->count) / (node1->count + node2->count));

    return std::make_tuple(w, b);
}

std::tuple<Eigen::MatrixXf, Eigen::VectorXf> CobwebContinuousTree::get_weights_and_bias(int hidden_dim)
{
    // get the decision boundary between two nodes up to the hidden_dim number of nodes
    // if the tree has less than hidden_dim number of nodes, then do all nodes

    Eigen::MatrixXf weights = Eigen::MatrixXf::Zero(hidden_dim, this->size);
    Eigen::VectorXf bias = Eigen::VectorXf::Zero(hidden_dim);

    std::vector<CobwebContinuousNode *> queue;
    queue.push_back(this->root);

    int count = 0;

    while (!queue.empty())
    {
        CobwebContinuousNode *current = queue.back();
        queue.pop_back();

        if (current->children.empty())
        {
            continue;
        }

        // for each node, if it has two children, then get the decision boundary
        if (current->children.size() == 2)
        {
            auto [w, b] = this->get_decision_boundary(current->children[0], current->children[1]);
            weights.row(count) = w;
            // print out w and weights row
            // std::cout << "w: " << w << std::endl;
            // std::cout << "weights row: " << weights.row(count) << std::endl;
            bias[count] = b;
            count += 1;
        }

        for (auto &child : current->children)
        {
            queue.push_back(child);
        }

        if (count == hidden_dim)
        {
            break;
        }
    }

    return std::make_tuple(weights, bias);
}

Eigen::VectorXf CobwebContinuousTree::weighted_state(const std::vector<CobwebContinuousNode *> &path, const std::vector<int> &levels)
{
    // softmax the levels
    // find the max level
    float max_level = *std::max_element(levels.begin(), levels.end());

    Eigen::VectorXf level_softmax = Eigen::VectorXf::Zero(path.size());
    for (size_t i = 0; i < path.size(); ++i)
    {
        level_softmax[i] = exp(levels[i] / max_level);
    }
    level_softmax = level_softmax / level_softmax.sum();

    Eigen::VectorXf out = Eigen::VectorXf::Zero(this->size);
    for (size_t i = 0; i < path.size(); ++i)
    {
        out += level_softmax[i] * path[i]->mean;
    }
    return out;
}

CobwebContinuousNode *CobwebContinuousTree::cobweb(const Eigen::VectorXf &instance, int label)
{
    int cobweb_label = label;

    CobwebContinuousNode *current = this->root;

    // keep track of the categorization path (nodes visited)
    // also keep track of the level of each node

    std::vector<CobwebContinuousNode *> path;
    std::vector<int> levels;

    // timing the cobweb algorithm
    auto start = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (current->children.empty() &&
            (current->count == 0 || current->is_exact_match(instance, cobweb_label)))
        {
            // std::cout << "empty / exact match" << std::endl;
            current->increment_counts(instance, cobweb_label);

            // update the path and levels
            path.push_back(current);
            levels.push_back(current->depth());

            break;
        }
        else if (current->children.empty())
        {
            // std::cout << "fringe split" << std::endl;
            CobwebContinuousNode *new_node = new CobwebContinuousNode(current);
            current->parent = new_node;
            new_node->children.push_back(current);

            if (new_node->parent == nullptr)
            {
                root = new_node;
            }
            else
            {
                new_node->parent->children.erase(remove(new_node->parent->children.begin(),
                                                        new_node->parent->children.end(), current),
                                                 new_node->parent->children.end());
                new_node->parent->children.push_back(new_node);
            }
            new_node->increment_counts(instance, cobweb_label);

            current = new CobwebContinuousNode(this->size, this->label_size);
            current->parent = new_node;
            current->tree = this;
            current->node_id = this->node_count;
            this->node_count += 1;
            
            current->increment_counts(instance, cobweb_label);
            new_node->children.push_back(current);

            // update the path and levels
            path.push_back(new_node);
            levels.push_back(new_node->depth());
            path.push_back(current);
            levels.push_back(current->depth());

            break;
        }
        else
        {
            auto [best1_mi, best1, best2] = current->two_best_children(instance, cobweb_label);
            // auto best1 = current->children[0];
            // auto best2 = current->children[0];
            // auto[_, bestAction] = current->get_best_operation(instance, best1, best2, best1_mi);
            size_t bestAction = BEST;

            if (bestAction == BEST)
            {
                // std::cout << "best" << std::endl;
                current->increment_counts(instance, cobweb_label);
                current = best1;

                // update the path and levels
                path.push_back(current);
                levels.push_back(current->depth());
            }
            else if (bestAction == NEW)
            {
                // std::cout << "new" << std::endl;
                current->increment_counts(instance, label);

                // current = current->create_new_child(instance);
                CobwebContinuousNode *new_child = new CobwebContinuousNode(this->size, this->label_size);
                new_child->parent = current;
                new_child->tree = this;
                new_child->increment_counts(instance, label);
                current->children.push_back(new_child);
                current = new_child;
                break;
            }
            else if (bestAction == MERGE)
            {
                // std::cout << "merge" << std::endl;
                current->increment_counts(instance, label);
                // CobwebContinuousNode* new_child = current->merge(best1, best2);

                CobwebContinuousNode *new_child = new CobwebContinuousNode(this->size, this->label_size);
                new_child->parent = current;
                new_child->tree = this;

                new_child->update_counts_from_node(best1);
                new_child->update_counts_from_node(best2);
                best1->parent = new_child;
                best2->parent = new_child;
                new_child->children.push_back(best1);
                new_child->children.push_back(best2);
                current->children.erase(remove(current->children.begin(),
                                               current->children.end(), best1),
                                        current->children.end());
                current->children.erase(remove(current->children.begin(),
                                               current->children.end(), best2),
                                        current->children.end());
                current->children.push_back(new_child);
                current = new_child;
            }
            else if (bestAction == SPLIT)
            {
                // std::cout << "split" << std::endl;
                current->children.erase(remove(current->children.begin(),
                                               current->children.end(), best1),
                                        current->children.end());
                for (auto &c : best1->children)
                {
                    c->parent = current;
                    c->tree = this;
                    current->children.push_back(c);
                }
                delete best1;
            }
            else
            {
                throw "Best action choice \"" + std::to_string(bestAction) +
                    "\" (best=0, new=1, merge=2, split=3) not a recognized option. This should be impossible...";
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "Elapsed time for cobweb algorithm: " << elapsed_seconds.count() << "s\n";

    if (this->learn_embedding)
    {
        // update the embedding matrix
        Eigen::VectorXf hidden_state = weighted_state(path, levels);

        // check if the range of the hidden state is within the range of the embedding matrix
        // if ((hidden_state.minCoeff() < -1.0) || (hidden_state.maxCoeff() > 1.0))
        // {
        //     std::cout << "Hidden state range: " << hidden_state.minCoeff() << " " << hidden_state.maxCoeff() << std::endl;
        //     std::cout << "Embedding matrix range: " << this->embedding_matrix.minCoeff() << " " << this->embedding_matrix.maxCoeff() << std::endl;
        // }
        // update the embedding matrix
        // this->embedding_matrix.row(label) = (1 - this->learning_rate) * this->embedding_matrix.row(label) + this->learning_rate * hidden_state.transpose();
    }

    return current;
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::VectorXf> CobwebContinuousTree::predict(const Eigen::VectorXf &instance, int max_nodes, bool greedy)
{
    if (this->learn_embedding)
    {
        // convert token ids into average embeddings
        Eigen::MatrixXf selected_rows(instance.size(), this->embed_dim);
        for (int i = 0; i < instance.size(); ++i)
        {
            // get the vector representation of the token from get_token_embedding
            // selected_rows.row(i) = this->get_token_embedding(instance[i]);
            // // if get_token_embedding returns a zero vector, we can use the embedding matrix directly
            // if (selected_rows.row(i).isZero())
            // {
            //     selected_rows.row(i) = this->embedding_matrix.row(instance[i]);
            // }
            selected_rows.row(i) = this->embedding_matrix.row(instance[i]);
        }
        // Compute the mean by averaging across the rows
        Eigen::VectorXf instance_embedding = selected_rows.colwise().mean();

        // Eigen::MatrixXf selected_columns(this->embed_dim, instance.size());
        // for (int i = 0; i < instance.size(); ++i)
        // {
        //     // Access the column directly from the embedding matrix
        //     selected_columns.col(i) = this->embedding_matrix.col(instance[i]);
        // }

        // // Compute the mean by averaging across the columns
        // Eigen::VectorXf instance_embedding = selected_columns.rowwise().mean();

        // std::cout << "Elapsed time for embedding extraction: " << elapsed_seconds.count() << "s\n";
        // std::cout << "Instance embedding: " << std::endl << instance_embedding << std::endl;
        return this->predict_label(instance_embedding, max_nodes, greedy);
    }

    // this wrapper is useful if we want to do anything before calling predict.
    return this->predict_label(instance, max_nodes, greedy);
}

std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::VectorXf> CobwebContinuousTree::predict_label(const Eigen::VectorXf &instance, int max_nodes, bool greedy)
{

    float total_weight = 0.0;
    Eigen::VectorXf out = Eigen::VectorXf::Zero(this->label_size);
    Eigen::VectorXf hidden_state = Eigen::VectorXf::Zero(this->size);

    int nodes_expanded = 0;

    float root_ll_inst = this->root->log_prob(instance);

    auto queue = std::priority_queue<std::tuple<float, float, CobwebContinuousNode *>>();

    // std::cout << "root score: " << std::to_string(root_ll_inst) << std::endl;
    queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

    while (queue.size() > 0)
    {
        auto node = queue.top();
        queue.pop();
        nodes_expanded += 1;

        if (greedy)
        {
            queue = std::priority_queue<
                std::tuple<float, float, CobwebContinuousNode *>>();
        }

        float curr_score = std::get<0>(node);
        float curr_ll = std::get<1>(node);
        CobwebContinuousNode *curr = std::get<2>(node);

        // total_weight += curr_score;
        // std::cout << "weight = logsumexp(" << std::to_string(total_weight) << ", " << std::to_string(curr_score) << ")" << std::endl;
        // std::cout << "weight += " << std::to_string(curr_score) << " (" << std::to_string(exp(curr_score)) << ")" << std::endl;

        if (total_weight == 0)
        {
            total_weight = curr_score;
        }
        else
        {
            total_weight = logsumexp_f(total_weight, curr_score);
        }

        // auto curr_preds = curr->predict_probs();

        auto curr_predicted_mean = curr->predict_mean(instance);
        // float label_sum = 0.0f;
        // for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(this->label_space, curr->node_id); it; ++it)
        // {
        //     label_sum += it.value();
        // }

        // if (label_sum != 0)
        // {
        //     // auto curr_predicted_label = curr->labels / curr->labels.sum();
        //     Eigen::VectorXf curr_predicted_label = Eigen::VectorXf::Zero(this->label_size);
        //     // float label_sum = curr->labels.sum();

        //     // for (Eigen::SparseVector<float>::InnerIterator it(curr->labels); it; ++it)
        //     for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(this->label_space, curr->node_id); it; ++it)
        //     {
        //         curr_predicted_label[it.index()] = it.value() / label_sum;
        //     }
        //     out += exp(curr_score - total_weight) * (curr_predicted_label - out);
        // }

        // std::cout << "curr_score: " << std::to_string(curr_score) << std::endl;
        // std::cout << "total weight: " << std::to_string(total_weight) << std::endl;
        // std::cout << "weight ratio: " << std::to_string(exp(curr_score - total_weight)) << std::endl;
        // std::cout << "predicted mean: " << std::endl << curr_predicted_mean << std::endl << std::endl;

        // Eigen::VectorXf delta = curr_predicted_mean - out;
        // out += exp(curr_score - total_weight) * delta;
        hidden_state += exp(curr_score - total_weight) * (curr_predicted_mean - hidden_state);
        // std::cout << "out: " << std::endl << out << std::endl << std::endl;
        // std::cout << std::endl;

        if (nodes_expanded >= max_nodes)
            break;

        // TODO look at missing in computing prob children given instance
        // std::vector<double> children_probs = curr->prob_children_given_instance(instance);
        std::vector<float> log_children_probs = curr->log_prob_children_given_instance(instance);

        for (size_t i = 0; i < curr->children.size(); ++i)
        {
            auto child = curr->children[i];
            float child_ll_inst = child->log_prob(instance);
            float child_ll_given_parent = log_children_probs[i];
            float child_ll = child_ll_given_parent + curr_ll;

            // std::cout << "ll_node: " << child_ll << ", ll_inst: " << child_ll_inst << std::endl;
            queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
        }
    }

    return std::make_tuple(out, hidden_state, this->cosine_similarity(hidden_state.transpose()));
}

Eigen::VectorXf CobwebContinuousTree::cosine_similarity(const Eigen::VectorXf &output_logits)
{
    // compute the cosine similarity between the output logits and each row of the embedding matrix
    Eigen::VectorXf out = Eigen::VectorXf::Zero(this->label_size);
    // std::cout << "anything wrong here?" << std::endl;
    for (int i = 0; i < this->label_size; ++i)
    {
        out[i] = output_logits.dot(this->embedding_matrix.row(i)) / (output_logits.norm() * this->embedding_matrix.row(i).norm());
    }
    return out;
}

Eigen::VectorXf CobwebContinuousTree::predict_helper(const Eigen::VectorXf &instance, int max_nodes, bool greedy)
{

    float total_weight = 0.0;
    Eigen::VectorXf out = Eigen::VectorXf::Zero(this->size);

    int nodes_expanded = 0;

    float root_ll_inst = this->root->log_prob(instance);

    auto queue = std::priority_queue<std::tuple<float, float, CobwebContinuousNode *>>();

    // std::cout << "root score: " << std::to_string(root_ll_inst) << std::endl;
    queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

    while (queue.size() > 0)
    {
        auto node = queue.top();
        queue.pop();
        nodes_expanded += 1;

        if (greedy)
        {
            queue = std::priority_queue<
                std::tuple<float, float, CobwebContinuousNode *>>();
        }

        float curr_score = std::get<0>(node);
        float curr_ll = std::get<1>(node);
        CobwebContinuousNode *curr = std::get<2>(node);

        // total_weight += curr_score;
        // std::cout << "weight = logsumexp(" << std::to_string(total_weight) << ", " << std::to_string(curr_score) << ")" << std::endl;
        // std::cout << "weight += " << std::to_string(curr_score) << " (" << std::to_string(exp(curr_score)) << ")" << std::endl;

        if (total_weight == 0)
        {
            total_weight = curr_score;
        }
        else
        {
            total_weight = logsumexp_f(total_weight, curr_score);
        }

        // auto curr_preds = curr->predict_probs();
        auto curr_predicted_mean = curr->predict_mean(instance);

        // std::cout << "curr_score: " << std::to_string(curr_score) << std::endl;
        // std::cout << "total weight: " << std::to_string(total_weight) << std::endl;
        // std::cout << "weight ratio: " << std::to_string(exp(curr_score - total_weight)) << std::endl;
        // std::cout << "predicted mean: " << std::endl << curr_predicted_mean << std::endl << std::endl;

        // Eigen::VectorXf delta = curr_predicted_mean - out;
        // out += exp(curr_score - total_weight) * delta;
        out += exp(curr_score - total_weight) * (curr_predicted_mean - out);
        // std::cout << "out: " << std::endl << out << std::endl << std::endl;
        // std::cout << std::endl;

        if (nodes_expanded >= max_nodes)
            break;

        // TODO look at missing in computing prob children given instance
        // std::vector<double> children_probs = curr->prob_children_given_instance(instance);
        std::vector<float> log_children_probs = curr->log_prob_children_given_instance(instance);

        for (size_t i = 0; i < curr->children.size(); ++i)
        {
            auto child = curr->children[i];
            float child_ll_inst = child->log_prob(instance);
            float child_ll_given_parent = log_children_probs[i];
            float child_ll = child_ll_given_parent + curr_ll;

            // std::cout << "ll_node: " << child_ll << ", ll_inst: " << child_ll_inst << std::endl;
            queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
        }
    }

    return out;
}

float CobwebContinuousTree::log_prob(const Eigen::VectorXf &instance, int max_nodes, bool greedy)
{

    float total_weight = 0.0;
    float out = 0.0;
    int nodes_expanded = 0;

    float root_ll_inst = this->root->log_prob(instance);
    auto queue = std::priority_queue<std::tuple<float, float, CobwebContinuousNode *>>();

    // std::cout << "root score: " << std::to_string(root_ll_inst) << std::endl;
    queue.push(std::make_tuple(root_ll_inst, 0.0, this->root));

    while (queue.size() > 0)
    {
        auto node = queue.top();
        queue.pop();
        nodes_expanded += 1;

        if (greedy)
        {
            queue = std::priority_queue<
                std::tuple<float, float, CobwebContinuousNode *>>();
        }

        float curr_score = std::get<0>(node);
        float curr_ll = std::get<1>(node);
        CobwebContinuousNode *curr = std::get<2>(node);

        if (total_weight == 0)
        {
            total_weight = curr_score;
        }
        else
        {
            total_weight = logsumexp_f(total_weight, curr_score);
        }

        // auto curr_preds = curr->predict_probs();
        auto curr_predicted_log_prob = curr->log_prob(instance);

        // std::cout << "curr_score: " << std::to_string(curr_score) << std::endl;
        // std::cout << "total weight: " << std::to_string(total_weight) << std::endl;
        // std::cout << "weight ratio: " << std::to_string(exp(curr_score - total_weight)) << std::endl;
        // std::cout << "predicted mean: " << std::endl << curr_predicted_mean << std::endl << std::endl;

        // Eigen::VectorXf delta = curr_predicted_mean - out;
        // out += exp(curr_score - total_weight) * delta;
        out += exp(curr_score - total_weight) * (curr_predicted_log_prob - out);
        // std::cout << "out: " << std::endl << out << std::endl << std::endl;
        // std::cout << std::endl;

        if (nodes_expanded >= max_nodes)
            break;

        // TODO look at missing in computing prob children given instance
        // std::vector<double> children_probs = curr->prob_children_given_instance(instance);
        std::vector<float> log_children_probs = curr->log_prob_children_given_instance(instance);

        for (size_t i = 0; i < curr->children.size(); ++i)
        {
            auto child = curr->children[i];
            float child_ll_inst = child->log_prob(instance);
            float child_ll_given_parent = log_children_probs[i];
            float child_ll = child_ll_given_parent + curr_ll;

            // std::cout << "ll_node: " << child_ll << ", ll_inst: " << child_ll_inst << std::endl;
            queue.push(std::make_tuple(child_ll_inst + child_ll, child_ll, child));
            // queue.push(std::make_tuple(child_ll_inst, child_ll, child));
        }
    }

    return out;
}

std::string CobwebContinuousTree::__str__()
{
    return this->root->__str__();
}

void CobwebContinuousTree::clear()
{
    delete this->root;
    this->root = new CobwebContinuousNode(this->size, this->label_size);
    this->root->tree = this;
    this->root->node_id = this->node_count;
    this->node_count += 1;
}

Eigen::VectorXf CobwebContinuousTree::compute_var(const Eigen::VectorXf &sum_sq, const float count)
{
    return sum_sq / count + this->prior_var;
}

Eigen::VectorXf CobwebContinuousTree::score_reduction_path()
{
    Eigen::VectorXf out = Eigen::VectorXf::Zero(this->max_num_instances * 2);
    std::vector<CobwebContinuousNode *> queue;
    queue.push_back(this->root);
    while (!queue.empty())
    {
        CobwebContinuousNode *current = queue.back();
        queue.pop_back();
        // out[current->node_id] = current->score_reduction();
        for (auto &child : current->children)
        {
            queue.push_back(child);

            std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::SparseVector<float>> root_mean_var;
            std::tuple<Eigen::VectorXf, Eigen::VectorXf, Eigen::SparseVector<float>> child_mean_var;

            root_mean_var = current->mean_var();
            child_mean_var = child->mean_var();

            int score = this->compute_score(std::get<0>(child_mean_var), std::get<1>(child_mean_var), std::get<2>(child_mean_var),
                                            std::get<0>(root_mean_var), std::get<1>(root_mean_var), std::get<2>(root_mean_var));
            out[child->node_id] = score;
        }
    }
    return out;
}

float CobwebContinuousTree::compute_score(const Eigen::VectorXf &child_mean, const Eigen::VectorXf &child_var, const Eigen::SparseVector<float> &child_p_label,
                                          const Eigen::VectorXf &parent_mean, const Eigen::VectorXf &parent_var, const Eigen::SparseVector<float> &parent_p_label)
{

    // something like cosine angle?
    // float score = 0.5 * (1 - (child_mean).cwiseProduct(parent_mean).array().sum() / (child_mean.norm() * parent_mean.norm()));

    float score;

    // use own covar
    if (this->covar_from == 1)
    {
        // Typical info CU (using own diag covar)
        score = 0.5 * (parent_var.array().log() - child_var.array().log()).sum();
    }
    // use parent covar
    else if (this->covar_from == 2)
    {
        score = 0.5 * (child_mean - parent_mean).cwiseProduct(child_mean - parent_mean).cwiseQuotient(parent_var).array().sum();
    }

    // account for label
    // print the length of the label vector
    // std::cout << "parent label size: " << parent_p_label.size() << std::endl;
    // std::cout << "child label size: " << child_p_label.size() << std::endl;

    // find the runtime
    // auto start = std::chrono::high_resolution_clock::now();
    // std::cout << "score before: " << score << std::endl;
    // if (child_p_label.sum() != 0 || parent_p_label.sum() != 0)
    // {
    //     float smooth = 1e-8f;
    //     score += (-1.0 * (child_p_label.array() + smooth) * (child_p_label.array() + smooth).log()).sum();
    //     // std::cout << "score after: " << score << std::endl;
    //     score -= (-1.0 * (child_p_label.array() + smooth) * (parent_p_label.array() + smooth).log()).sum();
    // }

    float child_sum = child_p_label.sum();
    float parent_sum = parent_p_label.sum();
    // float score = 0.0f;

    if (child_sum != 0 || parent_sum != 0)
    {
        float smooth = 1e-8f;
        float temp_score = 0.0f;

        // Calculate the first term: (-1.0 * (child_p_label + smooth) * log(child_p_label + smooth)).sum()
        for (Eigen::SparseVector<float>::InnerIterator it(child_p_label); it; ++it) {
            float value = it.value() + smooth;
            temp_score += value * std::log(value);
        }
        score += -1.0f * temp_score;

        // Calculate the second term: (-1.0 * (child_p_label + smooth) * log(parent_p_label + smooth)).sum()
        temp_score = 0.0f;
        for (Eigen::SparseVector<float>::InnerIterator it(child_p_label); it; ++it) {
            float child_value = it.value() + smooth;
            float parent_value = parent_p_label.coeff(it.index()) + smooth; // Access the corresponding value in parent_p_label
            temp_score += child_value * std::log(parent_value);
        }
        score -= -1.0f * temp_score;
    }

    // std::cout << "score after: " << score << std::endl;


    // std::cout << "score after: " << score << std::endl;
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    return score;
}
