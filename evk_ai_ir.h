#pragma once

#include "evk_ai.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace evk::ai {

using NodeId = uint32_t;
using DataId = uint32_t;

inline constexpr NodeId invalid_node = std::numeric_limits<NodeId>::max();
inline constexpr DataId invalid_data = std::numeric_limits<DataId>::max();

struct ValueId {
    NodeId node = invalid_node;

    friend bool operator==(ValueId, ValueId) = default;
};

enum class DataType : uint8_t {
    Float16,
};

struct TensorDesc {
    Shape shape;
    DataType type = DataType::Float16;

    friend bool operator==(const TensorDesc&, const TensorDesc&) = default;
};

struct Input {};
struct Parameter {
    DataId data = invalid_data;
};
struct Constant {
    DataId data = invalid_data;
};

struct Conv2D {
    uint32_t stride_y = 1;
    uint32_t stride_x = 1;
    uint32_t pad_y = 0;
    uint32_t pad_x = 0;
    uint32_t groups = 1;
};

struct Relu {};

struct MaxPool2D {
    uint32_t kernel_y = 2;
    uint32_t kernel_x = 2;
    uint32_t stride_y = 2;
    uint32_t stride_x = 2;
};

struct Upsample2D {
    uint32_t scale_y = 2;
    uint32_t scale_x = 2;
};

struct Concat {
    int32_t axis = 1;
};

using Operation = std::variant<
    Input,
    Parameter,
    Constant,
    Conv2D,
    Relu,
    MaxPool2D,
    Upsample2D,
    Concat>;

struct Node {
    Operation operation;
    std::vector<ValueId> inputs;
    TensorDesc result;
    std::string debug_name;
};

struct TensorData {
    TensorDesc desc;
    std::vector<float16_t> values;
};

struct DataStore {
    std::vector<TensorData> values;

    DataId add(TensorData data);
    const TensorData& get(DataId id) const;
};

class Graph {
public:
    ValueId input(TensorDesc desc, std::string_view name = {});
    ValueId parameter(DataId data, TensorDesc desc, std::string_view name = {});
    ValueId constant(DataId data, TensorDesc desc, std::string_view name = {});

    ValueId conv2d(ValueId input, ValueId weight, ValueId bias,
                   Conv2D attributes = {}, std::string_view name = {});
    ValueId relu(ValueId input, std::string_view name = {});
    ValueId max_pool2d(ValueId input, MaxPool2D attributes = {},
                       std::string_view name = {});
    ValueId upsample2d(ValueId input, Upsample2D attributes = {},
                       std::string_view name = {});
    ValueId concat(ValueId a, ValueId b, int32_t axis,
                   std::string_view name = {});

    void add_output(ValueId output);
    void validate(const DataStore& data) const;

    const TensorDesc& desc(ValueId value) const;
    const Node& node(NodeId id) const;
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<ValueId>& inputs() const { return inputs_; }
    const std::vector<ValueId>& outputs() const { return outputs_; }

private:
    ValueId add_node(Operation operation, std::vector<ValueId> inputs,
                     TensorDesc result, std::string_view name);

    std::vector<Node> nodes_;
    std::vector<ValueId> inputs_;
    std::vector<ValueId> outputs_;
};

enum class Layout : uint8_t {
    NCHW,
    NHWC,
};

enum class Kernel : uint8_t {
    Conv2D,
    Conv2DRelu,
    Conv2DReluMaxPool2D,
    Upsample2DConcatConv2DRelu,
    Relu,
    MaxPool2D,
    Upsample2D,
    Concat,
};

struct PlanStep {
    Kernel kernel;
    std::vector<NodeId> nodes;
};

using Plan = std::vector<PlanStep>;

// Produces a deterministic semantic cover. Physical kernel support and target
// costs are layered onto this cover by the Vulkan compiler.
Plan build_fusion_plan(const Graph& graph);

class Executable {
public:
    Executable();
    ~Executable();
    Executable(Executable&&) noexcept;
    Executable& operator=(Executable&&) noexcept;
    Executable(const Executable&) = delete;
    Executable& operator=(const Executable&) = delete;

    Tensor& input(uint32_t index = 0);
    Tensor& output(uint32_t index = 0);
    Layout input_layout(uint32_t index = 0) const;
    Layout output_layout(uint32_t index = 0) const;
    const Plan& plan() const;
    void eval(bool submit = true, bool wait = true, bool profile = false);

    // Vulkan image bindings are external storage adapters, not semantic graph
    // operations. They preserve the compiled input/output storage layouts.
    void from_rgba(evk::Image& color, uint32_t width, uint32_t height,
                   uint32_t padded_width, uint32_t padded_height);
    void from_rgba(evk::Image& color, evk::Image& albedo, evk::Image& normal,
                   uint32_t width, uint32_t height,
                   uint32_t padded_width, uint32_t padded_height);
    void to_rgb(evk::Buffer& output, uint32_t width, uint32_t height,
                uint32_t padded_width);
    void to_rgba(evk::Image& output, uint32_t width, uint32_t height,
                 uint32_t padded_width);

private:
    struct Impl;
    explicit Executable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend Executable compile(const Graph&, const DataStore&);
};

// Compiles the semantic graph with the built-in Vulkan kernel catalog. The
// current image catalog requires cooperative-matrix support for its fused
// FP16 3x3 convolution implementations.
Executable compile(const Graph& graph, const DataStore& data);

} // namespace evk::ai
