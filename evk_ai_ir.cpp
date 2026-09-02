#include "evk_ai_ir.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace evk::ai {
namespace {

[[noreturn]] void invalid(std::string_view message) {
    throw std::runtime_error("evk::ai IR: " + std::string(message));
}

bool is_source(const Operation& operation) {
    return std::holds_alternative<Input>(operation) ||
           std::holds_alternative<Parameter>(operation) ||
           std::holds_alternative<Constant>(operation);
}

void validate_desc(const TensorDesc& desc) {
    if (desc.shape.rank() == 0) invalid("tensor rank must be non-zero");
    uint64_t elements = 1;
    for (uint32_t i = 0; i < desc.shape.rank(); ++i) {
        if (desc.shape[i] == 0) invalid("tensor dimensions must be non-zero");
        elements *= desc.shape[i];
        if (elements > std::numeric_limits<uint32_t>::max()) {
            invalid("tensor contains too many elements");
        }
    }
}

bool supports_image_convolution(const Graph& graph, NodeId id) {
    const Node& node = graph.node(id);
    const auto* conv = std::get_if<Conv2D>(&node.operation);
    if (!conv || conv->stride_y != 1u || conv->stride_x != 1u ||
        conv->pad_y != 1u || conv->pad_x != 1u || conv->groups != 1u) {
        return false;
    }
    const TensorDesc& input = graph.desc(node.inputs[0]);
    const TensorDesc& weight = graph.desc(node.inputs[1]);
    return input.type == DataType::Float16 && input.shape.rank() == 4u &&
           input.shape[0] == 1u && weight.shape[2] == 3u && weight.shape[3] == 3u;
}

bool supports_pool_fusion(const Graph& graph, NodeId id) {
    const auto* pool = std::get_if<MaxPool2D>(&graph.node(id).operation);
    return pool && pool->kernel_y == 2u && pool->kernel_x == 2u &&
           pool->stride_y == 2u && pool->stride_x == 2u;
}

bool supports_upsample_fusion(const Graph& graph, NodeId id) {
    const auto* upsample = std::get_if<Upsample2D>(&graph.node(id).operation);
    return upsample && upsample->scale_y == 2u && upsample->scale_x == 2u;
}

uint32_t normalized_axis(int32_t axis, uint32_t rank) {
    int64_t normalized = axis;
    if (normalized < 0) normalized += rank;
    if (normalized < 0 || normalized >= rank) invalid("concat axis is out of range");
    return uint32_t(normalized);
}

} // namespace

DataId DataStore::add(TensorData data) {
    validate_desc(data.desc);
    if (data.values.size() != data.desc.shape.count()) {
        invalid("tensor data size does not match its descriptor");
    }
    if (values.size() >= invalid_data) invalid("too many data values");
    DataId id = DataId(values.size());
    values.push_back(std::move(data));
    return id;
}

const TensorData& DataStore::get(DataId id) const {
    if (id >= values.size()) invalid("data id is out of range");
    return values[id];
}

ValueId Graph::add_node(Operation operation, std::vector<ValueId> inputs,
                        TensorDesc result, std::string_view name) {
    if (nodes_.size() >= invalid_node) invalid("too many nodes");
    NodeId id = NodeId(nodes_.size());
    for (ValueId input : inputs) {
        if (input.node >= id) invalid("nodes must be added in topological order");
    }
    validate_desc(result);
    nodes_.push_back(Node{
        .operation = std::move(operation),
        .inputs = std::move(inputs),
        .result = std::move(result),
        .debug_name = std::string(name),
    });
    return ValueId{id};
}

ValueId Graph::input(TensorDesc desc, std::string_view name) {
    ValueId value = add_node(Input{}, {}, std::move(desc), name);
    inputs_.push_back(value);
    return value;
}

ValueId Graph::parameter(DataId data, TensorDesc desc, std::string_view name) {
    return add_node(Parameter{data}, {}, std::move(desc), name);
}

ValueId Graph::constant(DataId data, TensorDesc desc, std::string_view name) {
    return add_node(Constant{data}, {}, std::move(desc), name);
}

ValueId Graph::conv2d(ValueId input, ValueId weight, ValueId bias,
                      Conv2D attributes, std::string_view name) {
    const TensorDesc& x = desc(input);
    const TensorDesc& w = desc(weight);
    const TensorDesc& b = desc(bias);
    if (x.shape.rank() != 4 || w.shape.rank() != 4 || b.shape.rank() != 1) {
        invalid("conv2d expects NCHW input, OIHW weight, and O bias");
    }
    if (attributes.stride_y == 0 || attributes.stride_x == 0 ||
        attributes.groups == 0) {
        invalid("conv2d stride and groups must be non-zero");
    }
    if (x.shape[1] % attributes.groups != 0 ||
        w.shape[0] % attributes.groups != 0 ||
        w.shape[1] * attributes.groups != x.shape[1] ||
        b.shape[0] != w.shape[0]) {
        invalid("conv2d channel dimensions are incompatible");
    }
    uint64_t padded_h = uint64_t(x.shape[2]) + 2ull * attributes.pad_y;
    uint64_t padded_w = uint64_t(x.shape[3]) + 2ull * attributes.pad_x;
    if (padded_h < w.shape[2] || padded_w < w.shape[3]) {
        invalid("conv2d kernel exceeds the padded input");
    }
    uint64_t output_h64 = (padded_h - w.shape[2]) / attributes.stride_y + 1u;
    uint64_t output_w64 = (padded_w - w.shape[3]) / attributes.stride_x + 1u;
    if (output_h64 > std::numeric_limits<uint32_t>::max() ||
        output_w64 > std::numeric_limits<uint32_t>::max()) {
        invalid("conv2d output dimensions overflow");
    }
    uint32_t output_h = uint32_t(output_h64);
    uint32_t output_w = uint32_t(output_w64);
    return add_node(attributes, {input, weight, bias},
                    TensorDesc{Shape({x.shape[0], w.shape[0], output_h, output_w}), x.type},
                    name);
}

ValueId Graph::relu(ValueId input, std::string_view name) {
    return add_node(Relu{}, {input}, desc(input), name);
}

ValueId Graph::max_pool2d(ValueId input, MaxPool2D attributes,
                          std::string_view name) {
    const TensorDesc& x = desc(input);
    if (x.shape.rank() != 4 || attributes.kernel_y == 0 ||
        attributes.kernel_x == 0 || attributes.stride_y == 0 ||
        attributes.stride_x == 0 || x.shape[2] < attributes.kernel_y ||
        x.shape[3] < attributes.kernel_x) {
        invalid("invalid max_pool2d dimensions");
    }
    uint32_t output_h = (x.shape[2] - attributes.kernel_y) / attributes.stride_y + 1u;
    uint32_t output_w = (x.shape[3] - attributes.kernel_x) / attributes.stride_x + 1u;
    return add_node(attributes, {input},
                    TensorDesc{Shape({x.shape[0], x.shape[1], output_h, output_w}), x.type},
                    name);
}

ValueId Graph::upsample2d(ValueId input, Upsample2D attributes,
                          std::string_view name) {
    const TensorDesc& x = desc(input);
    if (x.shape.rank() != 4 || attributes.scale_y == 0 || attributes.scale_x == 0) {
        invalid("invalid upsample2d dimensions");
    }
    uint64_t output_h = uint64_t(x.shape[2]) * attributes.scale_y;
    uint64_t output_w = uint64_t(x.shape[3]) * attributes.scale_x;
    if (output_h > std::numeric_limits<uint32_t>::max() ||
        output_w > std::numeric_limits<uint32_t>::max()) {
        invalid("upsample2d output dimensions overflow");
    }
    return add_node(attributes, {input},
                    TensorDesc{Shape({x.shape[0], x.shape[1],
                                      uint32_t(output_h), uint32_t(output_w)}), x.type},
                    name);
}

ValueId Graph::concat(ValueId a, ValueId b, int32_t axis, std::string_view name) {
    const TensorDesc& a_desc = desc(a);
    const TensorDesc& b_desc = desc(b);
    if (a_desc.type != b_desc.type || a_desc.shape.rank() != b_desc.shape.rank()) {
        invalid("concat inputs must have the same rank and type");
    }
    uint32_t normalized = normalized_axis(axis, a_desc.shape.rank());
    Shape output = a_desc.shape;
    for (uint32_t i = 0; i < output.rank(); ++i) {
        if (i != normalized && a_desc.shape[i] != b_desc.shape[i]) {
            invalid("concat non-axis dimensions must match");
        }
    }
    uint64_t axis_size = uint64_t(output.values[normalized]) +
                         b_desc.shape[normalized];
    if (axis_size > std::numeric_limits<uint32_t>::max()) {
        invalid("concat axis size overflows");
    }
    output.values[normalized] = uint32_t(axis_size);
    return add_node(Concat{int32_t(normalized)}, {a, b},
                    TensorDesc{output, a_desc.type}, name);
}

void Graph::add_output(ValueId output) {
    desc(output);
    outputs_.push_back(output);
}

const TensorDesc& Graph::desc(ValueId value) const {
    if (value.node >= nodes_.size()) invalid("value node is out of range");
    return nodes_[value.node].result;
}

const Node& Graph::node(NodeId id) const {
    if (id >= nodes_.size()) invalid("node id is out of range");
    return nodes_[id];
}

void Graph::validate(const DataStore& data) const {
    if (outputs_.empty()) invalid("graph has no outputs");
    for (const Node& current : nodes_) {
        const DataId* data_id = nullptr;
        if (auto* parameter = std::get_if<Parameter>(&current.operation)) {
            data_id = &parameter->data;
        } else if (auto* constant = std::get_if<Constant>(&current.operation)) {
            data_id = &constant->data;
        }
        if (data_id && data.get(*data_id).desc != current.result) {
            invalid("source descriptor does not match its data");
        }
    }
}

Plan build_fusion_plan(const Graph& graph) {
    Plan plan;
    std::vector<uint32_t> use_counts(graph.nodes().size(), 0u);
    for (const Node& node : graph.nodes()) {
        for (ValueId input : node.inputs) ++use_counts[input.node];
    }
    for (ValueId output : graph.outputs()) ++use_counts[output.node];
    std::vector<uint8_t> covered(graph.nodes().size(), 0u);

    auto single_use = [&](NodeId id) { return use_counts[id] == 1u; };
    auto cover = [&](Kernel kernel, std::vector<NodeId> nodes) {
        for (NodeId id : nodes) covered[id] = true;
        plan.push_back(PlanStep{kernel, std::move(nodes)});
    };

    for (NodeId root = NodeId(graph.nodes().size()); root-- > 0;) {
        if (covered[root]) continue;
        const Node& node = graph.node(root);
        if (is_source(node.operation)) continue;

        if (supports_pool_fusion(graph, root)) {
            NodeId relu = node.inputs[0].node;
            if (single_use(relu) &&
                std::holds_alternative<Relu>(graph.node(relu).operation)) {
                NodeId conv = graph.node(relu).inputs[0].node;
                if (single_use(conv) && supports_image_convolution(graph, conv)) {
                    cover(Kernel::Conv2DReluMaxPool2D, {conv, relu, root});
                    continue;
                }
            }
        }

        if (std::holds_alternative<Relu>(node.operation)) {
            NodeId conv = node.inputs[0].node;
            if (single_use(conv) && supports_image_convolution(graph, conv)) {
                NodeId conv_input = graph.node(conv).inputs[0].node;
                if (single_use(conv_input) &&
                    std::holds_alternative<Concat>(graph.node(conv_input).operation)) {
                    const Node& concat = graph.node(conv_input);
                    NodeId upsample = invalid_node;
                    uint32_t upsample_count = 0u;
                    for (ValueId input : concat.inputs) {
                        if (single_use(input.node) && supports_upsample_fusion(graph, input.node)) {
                            upsample = input.node;
                            ++upsample_count;
                        }
                    }
                    const auto& concat_attributes = std::get<Concat>(concat.operation);
                    if (upsample_count == 1u && concat_attributes.axis == 1) {
                        cover(Kernel::Upsample2DConcatConv2DRelu,
                              {upsample, conv_input, conv, root});
                        continue;
                    }
                }
                cover(Kernel::Conv2DRelu, {conv, root});
                continue;
            }
        }

        Kernel kernel = std::visit([](const auto& operation) -> Kernel {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, Conv2D>) return Kernel::Conv2D;
            if constexpr (std::is_same_v<T, Relu>) return Kernel::Relu;
            if constexpr (std::is_same_v<T, MaxPool2D>) return Kernel::MaxPool2D;
            if constexpr (std::is_same_v<T, Upsample2D>) return Kernel::Upsample2D;
            if constexpr (std::is_same_v<T, Concat>) return Kernel::Concat;
            invalid("source operation cannot be planned");
        }, node.operation);
        cover(kernel, {root});
    }

    std::reverse(plan.begin(), plan.end());
    return plan;
}

} // namespace evk::ai
