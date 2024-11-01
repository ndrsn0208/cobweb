#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include "cobweb_continuous_node.h"
#include "cobweb_continuous_tree.h"
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

NB_MODULE(cobweb_continuous, m)
{
        m.doc() = "cobweb continuous"; // optional module docstring

        nb::class_<CobwebContinuousNode>(m, "CobwebContinuousNode")
            .def(nb::init<int, int>())
            // .def("increment_counts", &CobwebContinuousNode::increment_counts)
            .def("log_prob", &CobwebContinuousNode::log_prob)
            .def("log_prob_class_given_instance", &CobwebContinuousNode::log_prob_class_given_instance)
            .def("__str__", &CobwebContinuousNode::__str__)
            .def("output_json", &CobwebContinuousNode::output_json)
            .def("get_linked_var", &CobwebContinuousNode::get_linked_var)
            .def_ro("count", &CobwebContinuousNode::count)
            .def_ro("children", &CobwebContinuousNode::children,
                    nb::rv_policy::reference)
            .def_ro("parent", &CobwebContinuousNode::parent,
                    nb::rv_policy::reference)
            .def_ro("tree", &CobwebContinuousNode::tree,
                    nb::rv_policy::reference)
            .def_ro("sum_sq", &CobwebContinuousNode::sum_sq,
                    nb::rv_policy::reference)
            .def_ro("mean", &CobwebContinuousNode::mean,
                    nb::rv_policy::reference)
            .def_ro("node_id", &CobwebContinuousNode::node_id);

        nb::class_<CobwebContinuousTree>(m, "CobwebContinuousTree")
            .def(nb::init<int, int, int, int, float, int, int, bool, bool>(),
                 nb::arg("size"),
                 nb::arg("label_size"),
                nb::arg("max_num_instances"),
                 nb::arg("embed_size"),
                 nb::arg("learning_rate"),
                 nb::arg("covar_type") = 1,
                 nb::arg("covar_from") = 2,
                 nb::arg("learn_embedding") = false,
                 nb::arg("use_onehot") = false)
            .def("ifit", &CobwebContinuousTree::ifit, nb::rv_policy::reference)
            .def("predict", &CobwebContinuousTree::predict,
                 nb::arg("instance"),
                 nb::arg("max_nodes") = 1000,
                 nb::arg("greedy") = false)
            .def("log_prob", &CobwebContinuousTree::log_prob,
                 nb::arg("instance"),
                 nb::arg("max_nodes") = 1000,
                 nb::arg("greedy") = false)
            .def("get_token_embedding", &CobwebContinuousTree::get_token_embedding,
                 nb::arg("token_id"))
            .def("num_nodes", &CobwebContinuousTree::num_nodes)
        //     .def("binary_and_balance", &CobwebContinuousTree::binary_and_balance)
                .def("isBinaryAndBalanced", &CobwebContinuousTree::isBinaryAndBalanced)
                .def("isBinary", &CobwebContinuousTree::isBinary)
            .def("score_reduction_path", &CobwebContinuousTree::score_reduction_path)
            .def("set_embedding_matrix", &CobwebContinuousTree::set_embedding_matrix,
                 nb::arg("embedding_matrix"))
            .def("get_decision_boundary", &CobwebContinuousTree::get_decision_boundary,
                        nb::arg("node1"),
                        nb::arg("node2"))
            .def("get_weights_and_bias", &CobwebContinuousTree::get_weights_and_bias,
                        nb::arg("hidden_dim"))
            .def("get_clusters", &CobwebContinuousTree::get_clusters,
                 nb::arg("n"))
            // .def("fit", &CobwebTree::fit,
            //      nb::arg("instances") = std::vector<AV_COUNT_TYPE>(),
            //      nb::arg("mode"),
            //      nb::arg("iterations") = 1,
            //      nb::arg("randomizeFirst") = true)
            // .def("categorize", &CobwebTree::categorize,
            //      nb::arg("instance") = std::vector<AV_COUNT_TYPE>(),
            //      // nb::arg("get_best_concept") = false,
            //      nb::rv_policy::reference)
            // .def("predict_probs", &CobwebTree::predict_probs_mixture)
            // .def("predict_probs_parallel", &CobwebTree::predict_probs_mixture_parallel)
            .def("clear", &CobwebContinuousTree::clear)
            .def("__str__", &CobwebContinuousTree::__str__)
            .def("get_embedding_matrix", &CobwebContinuousTree::get_embedding_matrix)
            // .def("dump_json", &CobwebTree::dump_json)
            // .def("load_json", &CobwebTree::load_json)
            // .def("load_json_stream", &CobwebTree::load_json_stream)
            .def_ro("root", &CobwebContinuousTree::root, nb::rv_policy::reference)
            .def_ro("embedding_matrix", &CobwebContinuousTree::embedding_matrix,
                    nb::rv_policy::reference);

}
