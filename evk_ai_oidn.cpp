#include "evk_ai_oidn.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace evk::ai::oidn {
namespace {

struct ArchiveTensor {
    std::string name;
    Shape shape;
    char data_type = 0;
    uint64_t offset = 0;
};

uint32_t round_up_image_dimension(uint32_t value) {
    if (value == 0 || value > std::numeric_limits<uint32_t>::max() - 15u) {
        throw std::runtime_error("OIDN image dimensions are invalid");
    }
    return (value + 15u) & ~15u;
}

template<typename T>
T read_value(const std::vector<uint8_t>& bytes, size_t& offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        throw std::runtime_error("truncated TZA archive");
    }
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

std::string read_string(const std::vector<uint8_t>& bytes, size_t& offset, size_t size) {
    if (offset > bytes.size() || size > bytes.size() - offset) {
        throw std::runtime_error("truncated TZA archive");
    }
    std::string value(reinterpret_cast<const char*>(bytes.data() + offset), size);
    offset += size;
    return value;
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("could not open OIDN weights: " + path);
    }

    std::streamsize size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("could not read OIDN weights size: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("could not read OIDN weights: " + path);
    }
    return bytes;
}

std::vector<ArchiveTensor> parse_table(const std::vector<uint8_t>& bytes) {
    size_t header_offset = 0;
    uint16_t magic = read_value<uint16_t>(bytes, header_offset);
    uint8_t major = read_value<uint8_t>(bytes, header_offset);
    read_value<uint8_t>(bytes, header_offset); // minor version
    uint64_t table_offset64 = read_value<uint64_t>(bytes, header_offset);

    if (magic != 0x41D7u || major != 2u) {
        throw std::runtime_error("unsupported OIDN TZA archive");
    }
    if (table_offset64 > bytes.size()) {
        throw std::runtime_error("invalid OIDN TZA table offset");
    }

    size_t offset = static_cast<size_t>(table_offset64);
    uint32_t count = read_value<uint32_t>(bytes, offset);
    std::vector<ArchiveTensor> tensors;
    tensors.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        ArchiveTensor tensor;
        uint16_t name_size = read_value<uint16_t>(bytes, offset);
        tensor.name = read_string(bytes, offset, name_size);

        uint8_t rank = read_value<uint8_t>(bytes, offset);
        if (rank == 0 || rank > Shape::MAX_DIMENSIONS) {
            throw std::runtime_error("unsupported tensor rank in OIDN weights");
        }
        tensor.shape.size = rank;
        uint64_t element_count = 1;
        for (uint8_t dim = 0; dim < rank; ++dim) {
            uint32_t value = read_value<uint32_t>(bytes, offset);
            if (value == 0 || element_count > std::numeric_limits<uint32_t>::max() / value) {
                throw std::runtime_error("invalid tensor shape in OIDN weights");
            }
            tensor.shape.values[dim] = value;
            element_count *= value;
        }

        std::string layout = read_string(bytes, offset, rank);
        if (!((rank == 1 && layout == "x") || (rank == 4 && layout == "oihw"))) {
            throw std::runtime_error("unsupported tensor layout in OIDN weights: " + layout);
        }

        tensor.data_type = read_value<char>(bytes, offset);
        if (tensor.data_type != 'h' && tensor.data_type != 'f') {
            throw std::runtime_error("unsupported tensor type in OIDN weights");
        }
        tensor.offset = read_value<uint64_t>(bytes, offset);

        uint64_t byte_count = element_count * (tensor.data_type == 'h' ? 2u : 4u);
        if (tensor.offset > bytes.size() || byte_count > bytes.size() - tensor.offset) {
            throw std::runtime_error("tensor data is outside the OIDN weights archive");
        }
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

} // namespace

struct Model::Kernels {
    evk::Pipeline conv;
    evk::Pipeline concat_conv;
    evk::Pipeline narrow_conv;
    evk::Pipeline narrow_concat_conv;
    evk::Pipeline phase_concat_conv;
    evk::Pipeline narrow_phase_concat_conv;
    evk::Pipeline prepacked_phase_concat_conv;
    evk::Pipeline conv_final;
    evk::Pipeline max_pool;
    evk::Pipeline to_rgb;

    Kernels() {
        if (!evk::GetFeatures().coopmat) {
            throw std::runtime_error("OIDN GPU inference requires cooperative matrix support");
        }
        conv = create("oidn_conv");
        concat_conv = evk::CreatePipeline({
            .name = "oidn_concat_conv",
            .CS = evk::loadSpirvFile("shaders/bin/oidn_conv.comp.spv"),
            .constants = evk::Constant{uint32_t(1u)},
        });
        narrow_conv = evk::CreatePipeline({
            .name = "oidn_narrow_conv",
            .CS = evk::loadSpirvFile("shaders/bin/oidn_conv_narrow.comp.spv"),
        });
        narrow_concat_conv = evk::CreatePipeline({
            .name = "oidn_narrow_concat_conv",
            .CS = evk::loadSpirvFile("shaders/bin/oidn_conv_narrow.comp.spv"),
            .constants = evk::Constant{uint32_t(1u)},
        });
        phase_concat_conv = create("oidn_concat_phase");
        narrow_phase_concat_conv = create("oidn_concat_phase_narrow");
        prepacked_phase_concat_conv = create("oidn_concat_phase_prepacked");
        conv_final = create("oidn_conv_output");
        max_pool = create("oidn_max_pool");
        to_rgb = create("oidn_to_rgb");
    }

    static evk::Pipeline create(const char* name) {
        return evk::CreatePipeline({
            .name = name,
            .CS = evk::loadSpirvFile(std::string("shaders/bin/") + name + ".comp.spv"),
        });
    }

    void convolution(Tensor& input, Tensor& weight, Tensor& bias, Tensor& output,
                     bool activation) const {
        uint32_t height = input.shape[1];
        uint32_t width = input.shape[2];
        uint32_t input_channels = input.shape[3];
        uint32_t output_channels = output.shape[3];
        uint32_t padded_k = weight.shape[0];
        auto& cmd = evk::ai::GetCmd();

        if (!activation) {
            cmd.bind(conv_final);
            cmd.push(evk::Constant{
                input.buffer.GetReference(),
                input.buffer.GetReference(),
                weight.buffer.GetReference(),
                bias.buffer.GetReference(),
                output.buffer.GetReference(),
                width,
                height,
                input_channels,
                output_channels,
                padded_k,
                width,
                input_channels,
                0u,
            });
            cmd.dispatch((width + 63u) / 64u, 1u, height);
            cmd.barrier();
            return;
        }

        cooperative_convolution(input.buffer, input.buffer, weight, bias, output,
                                width, height, input_channels,
                                width, input_channels, 0u, false);
    }

    void concat_convolution(Tensor& low_resolution, Tensor& skip, Tensor& weight,
                            Tensor& bias, Tensor& output) const {
        if ((skip.shape[2] % 32u) == 0u) {
            auto& cmd = evk::ai::GetCmd();
            uint32_t output_channels = output.shape[3];
            if (skip.shape[3] == 3u) {
                cmd.bind(prepacked_phase_concat_conv);
            } else {
                cmd.bind(output_channels <= 64u ? narrow_phase_concat_conv : phase_concat_conv);
            }
            cmd.push(evk::Constant{
                low_resolution.buffer.GetReference(),
                skip.buffer.GetReference(),
                weight.buffer.GetReference(),
                bias.buffer.GetReference(),
                output.buffer.GetReference(),
                skip.shape[2],
                skip.shape[1],
                low_resolution.shape[3] + skip.shape[3],
                output_channels,
                ((low_resolution.shape[3] + skip.shape[3] + 15u) / 16u) * 9u * 16u,
                low_resolution.shape[2],
                low_resolution.shape[3],
                skip.shape[3],
            });
            cmd.dispatch((skip.shape[2] / 2u + 63u) / 64u, skip.shape[1], 2u);
            cmd.barrier();
            return;
        }
        cooperative_convolution(low_resolution.buffer, skip.buffer, weight, bias, output,
                                skip.shape[2], skip.shape[1],
                                low_resolution.shape[3] + skip.shape[3],
                                low_resolution.shape[2], low_resolution.shape[3],
                                skip.shape[3], true);
    }

    void cooperative_convolution(evk::Buffer& input, evk::Buffer& skip,
                                 Tensor& weight, Tensor& bias, Tensor& output,
                                 uint32_t width, uint32_t height,
                                 uint32_t input_channels, uint32_t low_width,
                                 uint32_t low_channels,
                                 uint32_t skip_channels, bool fused) const {
        uint32_t output_channels = output.shape[3];
        uint32_t padded_k = ((input_channels + 15u) / 16u) * 9u * 16u;
        auto& cmd = evk::ai::GetCmd();
        bool narrow = output_channels <= 64u;
        if (narrow) {
            cmd.bind(fused ? narrow_concat_conv : narrow_conv);
        } else {
            cmd.bind(fused ? concat_conv : conv);
        }
        cmd.push(evk::Constant{
            input.GetReference(),
            skip.GetReference(),
            weight.buffer.GetReference(),
            bias.buffer.GetReference(),
            output.buffer.GetReference(),
            width,
            height,
            input_channels,
            output_channels,
            padded_k,
            low_width,
            low_channels,
            skip_channels,
        });
        uint32_t channels_per_workgroup = narrow ? 64u : 128u;
        cmd.dispatch((width + 63u) / 64u,
                     (output_channels + channels_per_workgroup - 1u) /
                         channels_per_workgroup,
                     height);
        cmd.barrier();
    }

    void pooling(Tensor& input, Tensor& output) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(max_pool);
        cmd.push(evk::Constant{
            input.buffer.GetReference(),
            output.buffer.GetReference(),
            output.shape[2],
            output.shape[1],
            output.shape[3],
        });
        cmd.dispatch((output.shape.count() + 255u) / 256u, 1u, 1u);
        cmd.barrier();
    }

    void convert_to_rgb(Tensor& input, evk::Buffer& output, uint32_t width,
                        uint32_t height, uint32_t padded_width) const {
        auto& cmd = evk::ai::GetCmd();
        cmd.bind(to_rgb);
        cmd.push(evk::Constant{
            input.buffer.GetReference(),
            output.GetReference(),
            width,
            height,
            padded_width,
            input.shape[3],
        });
        uint32_t elements = width * height * 3u;
        cmd.dispatch((elements + 255u) / 256u, 1u, 1u);
        cmd.barrier();
    }
};

Model::Model(const std::string& weights_path) {
    kernels_ = std::make_unique<Kernels>();
    load(weights_path);
}

Model::~Model() = default;

void Model::load(const std::string& weights_path) {
    std::vector<uint8_t> bytes = read_file(weights_path);
    std::vector<ArchiveTensor> archive_tensors = parse_table(bytes);

    std::unordered_map<std::string, std::unique_ptr<Tensor>> parameters;
    parameters.reserve(archive_tensors.size());

    for (const ArchiveTensor& source : archive_tensors) {
        auto tensor = std::make_unique<Tensor>(source.shape);
        float16_t* destination = tensor->cpu();
        uint32_t element_count = source.shape.count();

        if (source.data_type == 'h') {
            std::memcpy(destination, bytes.data() + source.offset,
                        size_t(element_count) * sizeof(float16_t));
        } else {
            const uint8_t* source_data = bytes.data() + source.offset;
            for (uint32_t i = 0; i < element_count; ++i) {
                float value;
                std::memcpy(&value, source_data + size_t(i) * sizeof(float), sizeof(float));
                destination[i] = float16_t(value);
            }
        }
        tensor->cpu_upload(false);

        auto [it, inserted] = parameters.emplace(source.name, std::move(tensor));
        if (!inserted) {
            throw std::runtime_error("duplicate tensor in OIDN weights: " + source.name);
        }
    }

    if (parameters.empty()) {
        throw std::runtime_error("OIDN weights archive is empty");
    }
    SubmitCmd(true);
    parameters_ = std::move(parameters);

    std::unordered_map<std::string, std::unique_ptr<Tensor>> packed_parameters;
    for (const auto& [name, source] : parameters_) {
        if (name.ends_with(".bias")) {
            uint32_t channels = source->shape[0];
            uint32_t packed_channels = (channels + 15u) & ~15u;
            auto packed = std::make_unique<Tensor>(Shape({16u, packed_channels}));
            float16_t* destination = packed->cpu();
            float16_t* source_data = source->cpu();
            std::fill(destination, destination + packed->shape.count(), float16_t(0.0f));
            for (uint32_t row = 0; row < 16u; ++row) {
                std::copy(source_data, source_data + channels,
                          destination + row * packed_channels);
            }
            packed->cpu_upload(false);
            packed_parameters.emplace(name, std::move(packed));
            continue;
        }
        if (!name.ends_with(".weight")) continue;
        uint32_t output_channels = source->shape[0];
        uint32_t packed_output_channels = (output_channels + 15u) & ~15u;
        uint32_t input_channels = source->shape[1];
        uint32_t kernel_height = source->shape[2];
        uint32_t kernel_width = source->shape[3];
        uint32_t channel_blocks = (input_channels + 15u) / 16u;
        uint32_t padded_k = channel_blocks * kernel_height * kernel_width * 16u;
        bool prepack_phase = name == "dec_conv1a.weight";
        uint32_t prepack_low_channels = 0u;
        if (prepack_phase) {
            if (input_channels <= 3u) {
                throw std::runtime_error("unsupported dec_conv1a channel count");
            }
            prepack_low_channels = input_channels - 3u;
            if ((prepack_low_channels % 16u) != 0u) {
                throw std::runtime_error("unsupported dec_conv1a channel count");
            }
        }
        uint32_t phase_k = prepack_low_channels / 16u * 4u * 16u;
        auto packed = std::make_unique<Tensor>(
            Shape({padded_k + 4u * phase_k, packed_output_channels}));
        float16_t* destination = packed->cpu();
        std::fill(destination, destination + packed->shape.count(), float16_t(0.0f));
        float16_t* source_data = source->cpu();

        for (uint32_t output_channel = 0; output_channel < output_channels; ++output_channel) {
            for (uint32_t input_channel = 0; input_channel < input_channels; ++input_channel) {
                for (uint32_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                    for (uint32_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                        uint32_t kernel_offset = kernel_y * kernel_width + kernel_x;
                        uint32_t k = ((input_channel / 16u) * kernel_height * kernel_width +
                                      kernel_offset) * 16u + input_channel % 16u;
                        uint32_t source_index =
                            ((output_channel * input_channels + input_channel) * kernel_height +
                             kernel_y) * kernel_width + kernel_x;
                        destination[k * packed_output_channels + output_channel] = source_data[source_index];
                    }
                }
            }
        }
        if (prepack_phase) {
            for (uint32_t phase_y = 0u; phase_y < 2u; ++phase_y) {
                for (uint32_t phase_x = 0u; phase_x < 2u; ++phase_x) {
                    uint32_t phase = phase_y * 2u + phase_x;
                    for (uint32_t output_channel = 0u; output_channel < output_channels;
                         ++output_channel) {
                        for (uint32_t input_channel = 0u;
                             input_channel < prepack_low_channels;
                             ++input_channel) {
                            for (uint32_t tap_y = 0u; tap_y < 2u; ++tap_y) {
                                for (uint32_t tap_x = 0u; tap_x < 2u; ++tap_x) {
                                    float sum = 0.0f;
                                    for (uint32_t kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
                                        uint32_t mapped_y = phase_y == 0u
                                            ? (kernel_y == 0u ? 0u : 1u)
                                            : (kernel_y == 2u ? 1u : 0u);
                                        if (mapped_y != tap_y) continue;
                                        for (uint32_t kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
                                            uint32_t mapped_x = phase_x == 0u
                                                ? (kernel_x == 0u ? 0u : 1u)
                                                : (kernel_x == 2u ? 1u : 0u);
                                            if (mapped_x != tap_x) continue;
                                            uint32_t source_index =
                                                ((output_channel * input_channels + input_channel) *
                                                 kernel_height + kernel_y) * kernel_width + kernel_x;
                                            sum += float(source_data[source_index]);
                                        }
                                    }
                                    uint32_t tap = tap_y * 2u + tap_x;
                                    uint32_t k = (input_channel / 16u * 4u + tap) * 16u +
                                                 input_channel % 16u;
                                    destination[(padded_k + phase * phase_k + k) *
                                                    packed_output_channels + output_channel] =
                                        float16_t(sum);
                                }
                            }
                        }
                    }
                }
            }
        }
        packed->cpu_upload(false);
        packed_parameters.emplace(name, std::move(packed));

    }
    SubmitCmd(true);
    packed_parameters_ = std::move(packed_parameters);

    // The balanced and fast color-only models share this topology and differ
    // only in their internal channel counts.
    bool balanced_model = parameters_.contains("dec_conv4a.weight") &&
                          parameter("dec_conv4a.weight").shape[0] == 112u &&
                          parameters_.contains("dec_conv1a.weight") &&
                          parameter("dec_conv1a.weight").shape[1] == 67u;
    bool fast_model = parameters_.contains("dec_conv4a.weight") &&
                      parameter("dec_conv4a.weight").shape[0] == 64u &&
                      parameters_.contains("dec_conv1a.weight") &&
                      parameter("dec_conv1a.weight").shape[1] == 35u;
    if (!parameters_.contains("enc_conv0.weight") ||
        parameter("enc_conv0.weight").shape[0] != 32u ||
        parameter("enc_conv0.weight").shape[1] != 3u ||
        (!balanced_model && !fast_model)) {
        parameters_.clear();
        throw std::runtime_error(
            "weights are not a supported color-only RT LDR OIDN model");
    }
}

Tensor& Model::parameter(const std::string& name) const {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::runtime_error("missing tensor in OIDN weights: " + name);
    }
    return *it->second;
}

Tensor& Model::packed_parameter(const std::string& name) const {
    auto it = packed_parameters_.find(name);
    if (it == packed_parameters_.end()) {
        throw std::runtime_error("missing packed tensor in OIDN weights: " + name);
    }
    return *it->second;
}

void Model::convert_to_rgb(Tensor& input, evk::Buffer& output, uint32_t width,
                           uint32_t height, uint32_t padded_width) const {
    kernels_->convert_to_rgb(input, output, width, height, padded_width);
}

Tensor& Model::build(Graph& graph, Tensor& input) const {
    if (!loaded()) {
        throw std::runtime_error("OIDN model is not loaded");
    }
    if (input.shape.rank() != 4 || input.shape[0] != 1u || input.shape[3] != 3u) {
        throw std::runtime_error("OIDN input must have shape [1, height, width, 3]");
    }
    if ((input.shape[1] % 16u) != 0u || (input.shape[2] % 16u) != 0u) {
        throw std::runtime_error("OIDN input height and width must be multiples of 16");
    }

    auto conv = [this, &graph](const char* name, Tensor& value, bool activation = true) -> Tensor& {
        std::string weight_name = std::string(name) + ".weight";
        std::string bias_name = std::string(name) + ".bias";
        Tensor& weight = packed_parameter(weight_name);
        Tensor& bias = packed_parameter(bias_name);
        uint32_t output_channels = parameter(bias_name).shape[0];
        uint32_t storage_channels = activation ? output_channels : (output_channels + 15u) & ~15u;
        Tensor& output = graph.tensor(Shape({
            value.shape[0], value.shape[1], value.shape[2], storage_channels
        }));
        output.name = std::string(name) + ".conv2d";
        output.forward_fn = [this, &value, &weight,
                             &bias, &output, activation]() {
            kernels_->convolution(value, weight, bias, output, activation);
        };
        return output;
    };

    auto concat_conv = [this, &graph](const char* name, Tensor& low_resolution,
                                      Tensor& skip) -> Tensor& {
        if (low_resolution.shape[1] * 2u != skip.shape[1] ||
            low_resolution.shape[2] * 2u != skip.shape[2]) {
            throw std::runtime_error("OIDN concat-convolution spatial dimensions do not match");
        }
        std::string weight_name = std::string(name) + ".weight";
        std::string bias_name = std::string(name) + ".bias";
        Tensor& weight = packed_parameter(weight_name);
        Tensor& bias = packed_parameter(bias_name);
        uint32_t output_channels = parameter(bias_name).shape[0];
        if (parameter(weight_name).shape[1] !=
            low_resolution.shape[3] + skip.shape[3]) {
            throw std::runtime_error("OIDN concat-convolution channel count does not match");
        }
        Tensor& output = graph.tensor(Shape({
            skip.shape[0], skip.shape[1], skip.shape[2], output_channels
        }));
        output.name = std::string(name) + ".conv2d";
        output.forward_fn = [this, &low_resolution, &skip, &weight,
                             &bias, &output]() {
            kernels_->concat_convolution(low_resolution, skip, weight, bias, output);
        };
        return output;
    };

    auto pool = [this, &graph](const char* name, Tensor& value) -> Tensor& {
        Tensor& output = graph.tensor(Shape({
            value.shape[0], value.shape[1] / 2u, value.shape[2] / 2u, value.shape[3]
        }));
        output.name = name;
        output.forward_fn = [this, &value, &output]() {
            kernels_->pooling(value, output);
        };
        return output;
    };

    Tensor* x = &conv("enc_conv0", input);
    x = &conv("enc_conv1", *x);
    Tensor& pool1 = pool("pool1", *x);

    x = &conv("enc_conv2", pool1);
    Tensor& pool2 = pool("pool2", *x);

    x = &conv("enc_conv3", pool2);
    Tensor& pool3 = pool("pool3", *x);

    x = &conv("enc_conv4", pool3);
    Tensor& pool4 = pool("pool4", *x);

    x = &conv("enc_conv5a", pool4);
    x = &conv("enc_conv5b", *x);

    x = &concat_conv("dec_conv4a", *x, pool3);
    x = &conv("dec_conv4b", *x);

    x = &concat_conv("dec_conv3a", *x, pool2);
    x = &conv("dec_conv3b", *x);

    x = &concat_conv("dec_conv2a", *x, pool1);
    x = &conv("dec_conv2b", *x);

    x = &concat_conv("dec_conv1a", *x, input);
    x = &conv("dec_conv1b", *x);
    return conv("dec_conv0", *x, false);
}

Denoiser::Denoiser(const std::string& weights_path, uint32_t width, uint32_t height)
    : width_(width),
      height_(height),
      padded_width_(round_up_image_dimension(width)),
      padded_height_(round_up_image_dimension(height)),
      model_(weights_path) {
    uint64_t padded_values = uint64_t(padded_width_) * padded_height_ * 3u;
    if (padded_values > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("OIDN image dimensions are too large");
    }

    input_ = &graph_.tensor(Shape({1u, padded_height_, padded_width_, 3u}));
    output_ = &model_.build(graph_, *input_);
    uint64_t output_size = uint64_t(width_) * height_ * 3u * sizeof(float);
    output_gpu_ = evk::CreateBuffer({
        .size = output_size,
        .usage = evk::BufferUsage::Storage | evk::BufferUsage::TransferSrc,
    });
    output_cpu_ = evk::CreateBuffer({
        .size = output_size,
        .usage = evk::BufferUsage::TransferDst,
        .memoryType = evk::MemoryType::GPU_TO_CPU,
    });
}

void Denoiser::denoise(std::span<const float> input_rgb, std::span<float> output_rgb,
                       bool profile) {
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    auto input_pack_begin = Clock::now();
    size_t value_count = size_t(width_) * height_ * 3u;
    if (input_rgb.size() != value_count || output_rgb.size() != value_count) {
        throw std::runtime_error("OIDN input and output must contain width * height * 3 values");
    }

    float16_t* input_data = input_->cpu();
    uint32_t padded_spatial = padded_width_ * padded_height_;
    std::fill(input_data, input_data + size_t(padded_spatial) * 3u, float16_t(0.0f));
    for (uint32_t y = 0; y < height_; ++y) {
        for (uint32_t x = 0; x < width_; ++x) {
            uint32_t source_pixel = y * width_ + x;
            uint32_t padded_pixel = y * padded_width_ + x;
            for (uint32_t channel = 0; channel < 3u; ++channel) {
                float value = input_rgb[size_t(source_pixel) * 3u + channel];
                if (!(value >= 0.0f)) value = 0.0f;
                input_data[padded_pixel * 3u + channel] =
                    float16_t((std::min)(value, 1.0f));
            }
        }
    }

    auto input_pack_end = Clock::now();
    input_->cpu_upload(false);
    auto graph_begin = Clock::now();
    graph_.eval(false, true, true, profile);
    auto graph_end = Clock::now();
    if (profile) {
        timings_ = evk::CmdTimestamps();
    } else {
        timings_.clear();
    }
    auto download_begin = Clock::now();
    model_.convert_to_rgb(*output_, output_gpu_, width_, height_, padded_width_);
    auto& download_cmd = evk::ai::GetCmd();
    download_cmd.copy(output_gpu_, output_cpu_, value_count * sizeof(float));
    evk::ai::SubmitCmd(true);
    auto download_end = Clock::now();

    auto output_unpack_begin = Clock::now();
    std::memcpy(output_rgb.data(), output_cpu_.GetPtr(), value_count * sizeof(float));
    auto output_unpack_end = Clock::now();
    cpu_timings_ = {
        .input_pack_ms = elapsed_ms(input_pack_begin, input_pack_end),
        .graph_ms = elapsed_ms(graph_begin, graph_end),
        .download_ms = elapsed_ms(download_begin, download_end),
        .output_unpack_ms = elapsed_ms(output_unpack_begin, output_unpack_end),
    };
}

} // namespace evk::ai::oidn
