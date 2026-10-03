#include "evk_metal_shader.h"

#include <spirv_msl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
class MetalCompiler : public spirv_cross::CompilerMSL {
public:
    using CompilerMSL::CompilerMSL;

protected:
    std::vector<uint32_t> samplerAliases;

    void emit_entry_point_declarations() override {
        CompilerMSL::emit_entry_point_declarations();
        for (const auto& image : get_shader_resources().sampled_images) {
            if (!has_extended_decoration(image.id, spirv_cross::SPIRVCrossDecorationOverlappingBinding)) continue;
            if (std::find(samplerAliases.begin(), samplerAliases.end(), image.id) != samplerAliases.end()) continue;
            samplerAliases.push_back(image.id);
            auto base = get_extended_decoration(image.id, spirv_cross::SPIRVCrossDecorationOverlappingBinding);
            if (get_variable_data_type(get<spirv_cross::SPIRVariable>(base)).basetype != spirv_cross::SPIRType::SampledImage) {
                throw std::runtime_error("Overlapping sampler requires a sampled-image base binding");
            }
            get<spirv_cross::SPIRFunction>(ir.default_entry_point).fixup_hooks_in.push_back([this, id = image.id, base] {
                auto* meta = ir.find_meta(base);
                bool override = meta && meta->decoration.qualified_alias_explicit_override;
                if (meta) meta->decoration.qualified_alias_explicit_override = false;
                auto baseName = to_name(base, false);
                if (meta) meta->decoration.qualified_alias_explicit_override = override;
                statement("const device auto& ", to_sampler_expression(id), " = ", baseName, "Smplr;");
            });
        }
    }

    void emit_instruction(const spirv_cross::Instruction& instruction) override {
        if (instruction.op != spv::OpRayQueryGetIntersectionTriangleVertexPositionsKHR) {
            CompilerMSL::emit_instruction(instruction);
            return;
        }
        const auto* operands = stream(instruction);
        flush_variable_declaration(operands[2]);
        auto query = to_expression(operands[2]);
        bool committed = get<spirv_cross::SPIRConstant>(operands[3]).scalar_i32() != 0;
        auto data = query + (committed ? ".get_committed_primitive_data()" : ".get_candidate_primitive_data()");
        auto pointer = "reinterpret_cast<const device packed_float3*>(" + data + ")";
        emit_op(operands[0], operands[1], "spvUnsafeArray<float3, 3>({float3(" + pointer + "[0]), float3(" + pointer
            + "[1]), float3(" + pointer + "[2])})", false);
    }

    void CopyMembers(const spirv_cross::SPIRType& type, const std::string& target, const std::string& source) {
        for (uint32_t index = 0; index < type.member_types.size(); ++index) {
            auto member = "." + to_member_name(type, index);
            const auto& memberType = get<spirv_cross::SPIRType>(type.member_types[index]);
            if (memberType.basetype == spirv_cross::SPIRType::Struct && memberType.array.empty()) {
                CopyMembers(memberType, target + member, source + member);
            } else {
                statement(target, member, " = ", source, member, ";");
            }
        }
    }

    void emit_store_statement(uint32_t target, uint32_t source) override {
        const auto& type = expression_type(source);
        auto* variable = maybe_get_backing_variable(target);
        if (type.basetype != spirv_cross::SPIRType::Struct || !type.array.empty() || !variable
            || !get_buffer_block_flags(variable->self).get(spv::DecorationVolatile)) {
            CompilerMSL::emit_store_statement(target, source);
            return;
        }
        CopyMembers(type, to_expression(target), to_unpacked_expression(source));
        register_write(target);
    }
};

std::vector<uint8_t> Read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    const auto size = file.tellg();
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid SPIR-V byte count");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return bytes;
}

void Translate(const std::filesystem::path& input, const std::filesystem::path& output) {
    const auto bytes = Read(input);
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    MetalCompiler compiler(std::move(words));
    const auto stage = compiler.get_execution_model();
    auto options = compiler.get_msl_options();
    options.msl_version = 30000;
    options.argument_buffers = true;
    options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    options.pad_argument_buffer_resources = true;
    compiler.set_msl_options(options);
    compiler.set_argument_buffer_device_address_space(0, true);
    auto common = compiler.get_common_options();
    common.vertex.flip_vert_y = false;
    compiler.set_common_options(common);

    for (uint32_t binding = 0; binding < 4; ++binding) {
        spirv_cross::MSLResourceBinding resource;
        resource.stage = stage;
        resource.desc_set = 0;
        resource.binding = binding;
        resource.basetype = binding == 0 ? spirv_cross::SPIRType::UInt
            : binding == 1 ? spirv_cross::SPIRType::SampledImage
            : binding == 2 ? spirv_cross::SPIRType::Image
            : spirv_cross::SPIRType::UInt64;
        resource.count = binding == 0 ? evk::metal::BUFFER_COUNT : binding == 3 ? evk::metal::TLAS_COUNT : evk::metal::IMAGE_COUNT;
        resource.msl_buffer = binding == 0 ? evk::metal::STORAGE_ID : evk::metal::TLAS_ID;
        resource.msl_texture = binding == 1 ? evk::metal::TEXTURE_ID : evk::metal::IMAGE_ID;
        resource.msl_sampler = evk::metal::SAMPLER_ID;
        compiler.add_msl_resource_binding(resource);
    }
    spirv_cross::MSLResourceBinding push;
    push.stage = stage;
    push.desc_set = spirv_cross::kPushConstDescSet;
    push.binding = spirv_cross::kPushConstBinding;
    push.basetype = spirv_cross::SPIRType::UInt;
    push.msl_buffer = evk::metal::PUSH_BUFFER;
    compiler.add_msl_resource_binding(push);

    evk::metal::ShaderInfo info;
    switch (stage) {
        case spv::ExecutionModelVertex: info.stage = evk::metal::ShaderStage::Vertex; break;
        case spv::ExecutionModelFragment: info.stage = evk::metal::ShaderStage::Fragment; break;
        case spv::ExecutionModelGLCompute: info.stage = evk::metal::ShaderStage::Compute; break;
        default: throw std::runtime_error("Unsupported Metal shader stage");
    }
    info.hash = evk::metal::ShaderHash(bytes);
    const auto& entry = compiler.get_entry_point(compiler.get_entry_points_and_stages()[0].name, stage);
    if (stage == spv::ExecutionModelGLCompute) {
        if (entry.workgroup_size.id_x || entry.workgroup_size.id_y || entry.workgroup_size.id_z) {
            throw std::runtime_error("Specialized workgroup sizes require explicit metadata support");
        }
        info.groupX = entry.workgroup_size.x;
        info.groupY = entry.workgroup_size.y;
        info.groupZ = entry.workgroup_size.z;
    }
    for (const auto& constant : compiler.get_specialization_constants()) {
        if (constant.constant_id >= 32) throw std::runtime_error("Function constant ID exceeds EVK constant slots");
        const auto& type = compiler.get_type(compiler.get_constant(constant.id).constant_type);
        if (type.vecsize != 1 || type.columns != 1 || (type.basetype != spirv_cross::SPIRType::Boolean && type.width != 32)) {
            throw std::runtime_error("Unsupported function constant type");
        }
        using Type = evk::metal::ConstantType;
        switch (type.basetype) {
            case spirv_cross::SPIRType::Boolean: info.constantTypes[constant.constant_id] = Type::Bool; break;
            case spirv_cross::SPIRType::Int: info.constantTypes[constant.constant_id] = Type::Int; break;
            case spirv_cross::SPIRType::UInt: info.constantTypes[constant.constant_id] = Type::UInt; break;
            case spirv_cross::SPIRType::Float: info.constantTypes[constant.constant_id] = Type::Float; break;
            default: throw std::runtime_error("Unsupported function constant type");
        }
    }
    std::ostringstream name;
    name << std::hex << std::setw(16) << std::setfill('0') << info.hash;
    std::filesystem::create_directories(output);
    const auto stem = output / name.str();
    const std::string source = compiler.compile();
    std::ofstream metal(stem.string() + ".metal");
    metal << source;
    if (!metal) throw std::runtime_error("Cannot write MSL source");
    std::ofstream metadata(stem.string() + ".info", std::ios::binary);
    metadata.write(reinterpret_cast<const char*>(&info), sizeof(info));
    if (!metadata) throw std::runtime_error("Cannot write shader metadata");
    std::cout << input.filename().string() << '\t' << stem.string() << '\n';
}
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: evk_metal_shader <output-directory> <shader.spv>...\n";
        return 1;
    }
    bool success = true;
    for (int i = 2; i < argc; ++i) {
        try {
            Translate(argv[i], argv[1]);
        } catch (const std::exception& error) {
            std::cerr << argv[i] << ": " << error.what() << '\n';
            success = false;
        }
    }
    return success ? 0 : 1;
}
