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

bool has_storage(evk::ImageUsage usage) {
    return (uint32_t(usage) & uint32_t(evk::ImageUsage::Storage)) != 0u;
}

const evk::ImageDesc& validate_image(evk::Image& image, uint32_t width,
                                     uint32_t height, const char* role) {
    const evk::ImageDesc& desc = evk::GetDesc(image);
    if (desc.extent.width != width || desc.extent.height != height ||
        desc.extent.depth != 1u) {
        throw std::runtime_error(std::string("OIDN ") + role +
                                 " image dimensions do not match the denoiser");
    }
    if (!has_storage(desc.usage)) {
        throw std::runtime_error(std::string("OIDN ") + role +
                                 " image requires Storage usage");
    }
    return desc;
}

void validate_rgba_image(evk::Image& image, uint32_t width, uint32_t height,
                         const char* role) {
    evk::Format format = validate_image(image, width, height, role).format;
    if (format != evk::Format::RGBA8Unorm &&
        format != evk::Format::RGBA16Sfloat) {
        throw std::runtime_error(std::string("OIDN ") + role +
            " image must use RGBA8Unorm or RGBA16Sfloat");
    }
}

void validate_normal_image(evk::Image& image, uint32_t width, uint32_t height) {
    evk::Format format = validate_image(image, width, height, "normal").format;
    if (format != evk::Format::RGBA8Snorm &&
        format != evk::Format::RGBA16Snorm &&
        format != evk::Format::RGBA16Sfloat &&
        format != evk::Format::RGBA32Sfloat) {
        throw std::runtime_error(
            "OIDN normal image must use RGBA8Snorm, RGBA16Snorm, "
            "RGBA16Sfloat, or RGBA32Sfloat");
    }
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


Model::Model(const std::string& weights_path) {
    std::vector<uint8_t> bytes = read_file(weights_path);
    std::vector<ArchiveTensor> archive_tensors = parse_table(bytes);

    data_ids_.reserve(archive_tensors.size());
    data_.values.reserve(archive_tensors.size());

    for (const ArchiveTensor& source : archive_tensors) {
        uint32_t element_count = source.shape.count();
        std::vector<float16_t> values(element_count);

        if (source.data_type == 'h') {
            std::memcpy(values.data(), bytes.data() + source.offset,
                        size_t(element_count) * sizeof(float16_t));
        } else {
            const uint8_t* source_data = bytes.data() + source.offset;
            for (uint32_t i = 0; i < element_count; ++i) {
                float value;
                std::memcpy(&value, source_data + size_t(i) * sizeof(float), sizeof(float));
                values[i] = float16_t(value);
            }
        }

        evk::ai::DataId data_id = data_.add({
            .desc = evk::ai::TensorDesc{source.shape},
            .values = std::move(values),
        });
        if (!data_ids_.emplace(source.name, data_id).second) {
            throw std::runtime_error("duplicate tensor in OIDN weights: " + source.name);
        }
    }

    if (data_ids_.empty()) {
        throw std::runtime_error("OIDN weights archive is empty");
    }

    if (!data_ids_.contains("enc_conv0.weight") ||
        !data_ids_.contains("dec_conv4a.weight") ||
        !data_ids_.contains("dec_conv1a.weight")) {
        throw std::runtime_error("weights are not a supported RT LDR OIDN model");
    }
    const Shape& input = data_.get(data_ids_.at("enc_conv0.weight")).desc.shape;
    const Shape& decoder4 = data_.get(data_ids_.at("dec_conv4a.weight")).desc.shape;
    const Shape& decoder1 = data_.get(data_ids_.at("dec_conv1a.weight")).desc.shape;
    input_channels_ = input[1];

    bool decoder_channels_supported =
        (decoder4[0] == 64u && decoder1[1] == 32u + input_channels_) ||
        (decoder4[0] == 112u && decoder1[1] == 64u + input_channels_);
    if (input[0] != 32u || (input_channels_ != 3u && input_channels_ != 9u) ||
        !decoder_channels_supported) {
        throw std::runtime_error("weights are not a supported RT LDR OIDN model");
    }
}

evk::ai::Graph Model::build_ir(uint32_t width, uint32_t height) const {
    namespace ai = evk::ai;
    if (width == 0u || height == 0u || (width % 16u) != 0u ||
        (height % 16u) != 0u) {
        throw std::runtime_error("OIDN IR dimensions must be positive multiples of 16");
    }

    ai::Graph graph;
    ai::ValueId input = graph.input(
        {Shape({1u, input_channels_, height, width})}, "input");

    auto constant = [this, &graph](const std::string& name) {
        auto id = data_ids_.find(name);
        if (id == data_ids_.end()) {
            throw std::runtime_error("missing tensor data in OIDN weights: " + name);
        }
        return graph.constant(id->second, data_.get(id->second).desc, name);
    };

    auto conv = [&graph, &constant](const char* name, ai::ValueId value,
                                    bool activation = true) {
        std::string prefix(name);
        ai::ValueId weight = constant(prefix + ".weight");
        ai::ValueId bias = constant(prefix + ".bias");
        ai::ValueId output = graph.conv2d(
            value, weight, bias, ai::Conv2D{.pad_y = 1u, .pad_x = 1u},
            prefix + ".conv2d");
        return activation ? graph.relu(output, prefix + ".relu") : output;
    };

    auto conv_pool = [&graph, &conv](const char* name, ai::ValueId value) {
        ai::ValueId convolved = conv(name, value);
        return graph.max_pool2d(convolved, {}, std::string(name) + ".pool");
    };

    auto concat_conv = [&graph, &conv](const char* name, ai::ValueId low_resolution,
                                       ai::ValueId skip) {
        ai::ValueId upsampled = graph.upsample2d(
            low_resolution, {}, std::string(name) + ".upsample");
        ai::ValueId joined = graph.concat(
            upsampled, skip, 1, std::string(name) + ".concat");
        return conv(name, joined);
    };

    ai::ValueId x = conv("enc_conv0", input);
    ai::ValueId pool1 = conv_pool("enc_conv1", x);
    ai::ValueId pool2 = conv_pool("enc_conv2", pool1);
    ai::ValueId pool3 = conv_pool("enc_conv3", pool2);
    ai::ValueId pool4 = conv_pool("enc_conv4", pool3);

    x = conv("enc_conv5a", pool4);
    x = conv("enc_conv5b", x);
    x = concat_conv("dec_conv4a", x, pool3);
    x = conv("dec_conv4b", x);
    x = concat_conv("dec_conv3a", x, pool2);
    x = conv("dec_conv3b", x);
    x = concat_conv("dec_conv2a", x, pool1);
    x = conv("dec_conv2b", x);
    x = concat_conv("dec_conv1a", x, input);
    x = conv("dec_conv1b", x);
    x = conv("dec_conv0", x, false);
    graph.add_output(x);
    return graph;
}

Denoiser::Denoiser(const std::string& weights_path, uint32_t width, uint32_t height)
    : width_(width),
      height_(height),
      padded_width_(round_up_image_dimension(width)),
      padded_height_(round_up_image_dimension(height)) {
    Model model(weights_path);
    input_channels_ = model.input_channels();
    uint64_t padded_values = uint64_t(padded_width_) * padded_height_ *
                             input_channels_;
    if (padded_values > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("OIDN image dimensions are too large");
    }

    evk::ai::Graph graph = model.build_ir(padded_width_, padded_height_);
    executable_ = evk::ai::compile(graph, model.data());
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

void Denoiser::denoise(std::span<const float> color_rgb, std::span<float> output_rgb,
                       bool profile) {
    if (uses_auxiliary_inputs()) {
        throw std::runtime_error(
            "OIDN weights require color, albedo, and normal inputs");
    }
    denoise_cpu(color_rgb, {}, {}, output_rgb, profile);
}

void Denoiser::denoise(std::span<const float> color_rgb,
                       std::span<const float> albedo_rgb,
                       std::span<const float> normal_xyz,
                       std::span<float> output_rgb, bool profile) {
    if (!uses_auxiliary_inputs()) {
        throw std::runtime_error(
            "OIDN weights do not accept albedo and normal inputs");
    }
    denoise_cpu(color_rgb, albedo_rgb, normal_xyz, output_rgb, profile);
}

void Denoiser::denoise_cpu(std::span<const float> color_rgb,
                           std::span<const float> albedo_rgb,
                           std::span<const float> normal_xyz,
                           std::span<float> output_rgb, bool profile) {
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point begin, Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    auto input_pack_begin = Clock::now();
    size_t value_count = size_t(width_) * height_ * 3u;
    if (color_rgb.size() != value_count || output_rgb.size() != value_count) {
        throw std::runtime_error(
            "OIDN color and output must contain width * height * 3 values");
    }
    if (uses_auxiliary_inputs() &&
        (albedo_rgb.size() != value_count || normal_xyz.size() != value_count)) {
        throw std::runtime_error(
            "OIDN albedo and normal must contain width * height * 3 values");
    }

    Tensor& input = executable_.input();
    float16_t* input_data = input.cpu();
    uint32_t input_channels = input_channels_;
    std::fill(input_data,
              input_data + size_t(padded_width_) * padded_height_ * input_channels,
              float16_t(0.0f));
    for (uint32_t y = 0; y < height_; ++y) {
        for (uint32_t x = 0; x < width_; ++x) {
            size_t source_pixel = size_t(y) * width_ + x;
            size_t padded_pixel = size_t(y) * padded_width_ + x;
            size_t input_index = padded_pixel * input_channels;
            for (uint32_t channel = 0; channel < 3u; ++channel) {
                size_t source_index = source_pixel * 3u + channel;
                float value = color_rgb[source_index];
                if (!(value >= 0.0f)) value = 0.0f;
                input_data[input_index + channel] =
                    float16_t((std::min)(value, 1.0f));

                if (input_channels == 9u) {
                    value = albedo_rgb[source_index];
                    if (!(value >= 0.0f)) value = 0.0f;
                    input_data[input_index + 3u + channel] =
                        float16_t((std::min)(value, 1.0f));

                    value = normal_xyz[source_index];
                    if (value != value) value = 0.0f;
                    input_data[input_index + 6u + channel] =
                        float16_t((std::clamp)(value, -1.0f, 1.0f));
                }
            }
        }
    }

    auto input_pack_end = Clock::now();
    input.cpu_upload(false);
    auto graph_begin = Clock::now();
    executable_.eval(true, true, profile);
    auto graph_end = Clock::now();
    if (profile) {
        timings_ = evk::CmdTimestamps();
    } else {
        timings_.clear();
    }
    auto download_begin = Clock::now();
    executable_.to_rgb(output_gpu_, width_, height_, padded_width_);
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

void Denoiser::denoise(evk::Cmd& cmd, evk::Image& input_rgba,
                       evk::Image& output_rgba) {
    if (uses_auxiliary_inputs()) {
        throw std::runtime_error(
            "OIDN weights require color, albedo, and normal inputs");
    }
    validate_rgba_image(input_rgba, width_, height_, "input");
    validate_rgba_image(output_rgba, width_, height_, "output");

    evk::ai::WithCmd(cmd, [&]() {
        cmd.barrier();
        executable_.from_rgba(input_rgba, width_, height_,
                              padded_width_, padded_height_);
        executable_.eval(false, false, false);
        executable_.to_rgba(output_rgba, width_, height_, padded_width_);
    });
}

void Denoiser::denoise(evk::Cmd& cmd, evk::Image& color_rgba,
                       evk::Image& albedo_rgba, evk::Image& normal_rgba,
                       evk::Image& output_rgba) {
    if (!uses_auxiliary_inputs()) {
        throw std::runtime_error(
            "OIDN weights do not accept albedo and normal inputs");
    }
    validate_rgba_image(color_rgba, width_, height_, "color");
    validate_rgba_image(albedo_rgba, width_, height_, "albedo");
    validate_normal_image(normal_rgba, width_, height_);
    validate_rgba_image(output_rgba, width_, height_, "output");

    evk::ai::WithCmd(cmd, [&]() {
        cmd.barrier();
        executable_.from_rgba(color_rgba, albedo_rgba, normal_rgba,
                              width_, height_, padded_width_, padded_height_);
        executable_.eval(false, false, false);
        executable_.to_rgba(output_rgba, width_, height_, padded_width_);
    });
}

} // namespace evk::ai::oidn
