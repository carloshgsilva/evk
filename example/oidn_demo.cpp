#include "bmp.h"
#include "evk_ai_oidn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vec3 {
    float x = 0;
    float y = 0;
    float z = 0;

    Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator*(Vec3 b) const { return {x * b.x, y * b.y, z * b.z}; }
};

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 normalize(Vec3 value) {
    float inverse_length = 1.0f / std::sqrt(dot(value, value));
    return value * inverse_length;
}

struct Random {
    uint32_t state;

    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return float(state & 0x00FFFFFFu) / float(0x01000000u);
    }
};

Vec3 random_hemisphere(Vec3 normal, Random& random) {
    float z = random.next();
    float angle = 6.28318530718f * random.next();
    float radius = std::sqrt((std::max)(0.0f, 1.0f - z * z));
    Vec3 direction{radius * std::cos(angle), radius * std::sin(angle), z};
    if (dot(direction, normal) < 0.0f) direction = direction * -1.0f;
    return normalize(direction + normal);
}

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 albedo;
};

struct Hit {
    float distance = 1e30f;
    Vec3 position;
    Vec3 normal;
    Vec3 albedo;
    bool found = false;
};

void intersect_sphere(Vec3 origin, Vec3 direction, const Sphere& sphere, Hit& hit) {
    Vec3 relative = origin - sphere.center;
    float half_b = dot(relative, direction);
    float c = dot(relative, relative) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - c;
    if (discriminant < 0.0f) return;

    float distance = -half_b - std::sqrt(discriminant);
    if (distance <= 0.001f || distance >= hit.distance) return;
    hit.distance = distance;
    hit.position = origin + direction * distance;
    hit.normal = normalize(hit.position - sphere.center);
    hit.albedo = sphere.albedo;
    hit.found = true;
}

Hit intersect_scene(Vec3 origin, Vec3 direction) {
    static const Sphere spheres[] = {
        {{-0.85f, 0.05f, -3.2f}, 0.75f, {0.82f, 0.18f, 0.12f}},
        {{ 0.75f, 0.10f, -2.6f}, 0.60f, {0.12f, 0.36f, 0.82f}},
        {{ 0.00f, 1.15f, -3.7f}, 0.55f, {0.72f, 0.68f, 0.14f}},
    };

    Hit hit;
    for (const Sphere& sphere : spheres) {
        intersect_sphere(origin, direction, sphere, hit);
    }

    if (direction.y < -0.0001f) {
        float distance = (-0.72f - origin.y) / direction.y;
        if (distance > 0.001f && distance < hit.distance) {
            Vec3 position = origin + direction * distance;
            int checker = int(std::floor(position.x)) + int(std::floor(position.z));
            hit.distance = distance;
            hit.position = position;
            hit.normal = {0.0f, 1.0f, 0.0f};
            hit.albedo = (checker & 1) ? Vec3{0.22f, 0.22f, 0.22f}
                                        : Vec3{0.72f, 0.72f, 0.72f};
            hit.found = true;
        }
    }
    return hit;
}

Vec3 trace(Vec3 origin, Vec3 direction, Random& random) {
    Vec3 radiance;
    Vec3 throughput{1.0f, 1.0f, 1.0f};

    for (int bounce = 0; bounce < 4; ++bounce) {
        Hit hit = intersect_scene(origin, direction);
        if (!hit.found) {
            float sky = 0.5f * (direction.y + 1.0f);
            Vec3 environment = Vec3{0.70f, 0.80f, 1.0f} * sky +
                               Vec3{0.10f, 0.12f, 0.16f} * (1.0f - sky);
            radiance = radiance + throughput * environment;
            break;
        }

        Vec3 light_direction = normalize(Vec3{-0.7f, 1.0f, 0.3f});
        float direct = (std::max)(0.0f, dot(hit.normal, light_direction));
        radiance = radiance + throughput * hit.albedo * (0.07f + 0.45f * direct);
        throughput = throughput * hit.albedo * 0.55f;
        origin = hit.position + hit.normal * 0.002f;
        direction = random_hemisphere(hit.normal, random);
    }
    return radiance;
}

float linear_to_srgb(float value) {
    value = (std::max)(0.0f, value);
    return value <= 0.0031308f
        ? 12.92f * value
        : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

std::vector<float> render_path_traced_image(uint32_t width, uint32_t height) {
    constexpr uint32_t samples_per_pixel = 4;
    std::vector<float> image(size_t(width) * height * 3u);
    Vec3 camera{0.0f, 0.15f, 0.7f};
    float aspect = float(width) / float(height);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            Random random{0x9E3779B9u ^ (x + y * width + 1u) * 747796405u};
            Vec3 color;
            for (uint32_t sample = 0; sample < samples_per_pixel; ++sample) {
                float u = (float(x) + random.next()) / float(width);
                float v = (float(y) + random.next()) / float(height);
                Vec3 direction = normalize({
                    (2.0f * u - 1.0f) * aspect,
                    1.0f - 2.0f * v,
                    -1.65f,
                });
                color = color + trace(camera, direction, random);
            }
            color = color * (1.0f / float(samples_per_pixel));

            size_t index = (size_t(y) * width + x) * 3u;
            image[index + 0] = (std::min)(linear_to_srgb(color.x), 1.0f);
            image[index + 1] = (std::min)(linear_to_srgb(color.y), 1.0f);
            image[index + 2] = (std::min)(linear_to_srgb(color.z), 1.0f);
        }
    }
    return image;
}

void save_image(const char* path, uint32_t width, uint32_t height,
                const std::vector<float>& values) {
    BMP bitmap{int(width), int(height)};
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t index = (size_t(y) * width + x) * 3u;
            auto to_byte = [&](uint32_t channel) {
                float value = (std::clamp)(values[index + channel], 0.0f, 1.0f);
                return uint8_t(value * 255.0f + 0.5f);
            };
            bitmap.set_pixel(int(x), int(y), to_byte(0), to_byte(1), to_byte(2));
        }
    }
    if (!bitmap.save(path)) {
        throw std::runtime_error(std::string("could not save image: ") + path);
    }
}

} // namespace

void oidn_demo(const char* weights_path) {
    constexpr uint32_t width = 1920;
    constexpr uint32_t height = 1080;

    printf("[oidn] Rendering deterministic %ux%u path-traced input...\n", width, height);
    auto pathtrace_begin = std::chrono::steady_clock::now();
    std::vector<float> noisy = render_path_traced_image(width, height);
    auto pathtrace_end = std::chrono::steady_clock::now();
    save_image("oidn_noisy.bmp", width, height, noisy);

    printf("[oidn] Loading %s\n", weights_path);
    auto load_begin = std::chrono::steady_clock::now();
    evk::ai::oidn::Denoiser denoiser(weights_path, width, height);
    auto load_end = std::chrono::steady_clock::now();
    std::vector<float> denoised(noisy.size());
    auto inference_begin = std::chrono::steady_clock::now();
    denoiser.denoise(noisy, denoised, true);
    auto inference_end = std::chrono::steady_clock::now();

    std::vector<float16_t> noisy_rgba(size_t(width) * height * 4u,
                                      float16_t(1.0f));
    for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel) {
        noisy_rgba[pixel * 4u + 0u] = noisy[pixel * 3u + 0u];
        noisy_rgba[pixel * 4u + 1u] = noisy[pixel * 3u + 1u];
        noisy_rgba[pixel * 4u + 2u] = noisy[pixel * 3u + 2u];
    }
    evk::Image input_image = evk::CreateImage({
        .extent = {width, height},
        .format = evk::Format::RGBA16Sfloat,
        .usage = evk::ImageUsage::Storage | evk::ImageUsage::TransferDst,
    });
    evk::Image output_image = evk::CreateImage({
        .extent = {width, height},
        .format = evk::Format::RGBA16Sfloat,
        .usage = evk::ImageUsage::Storage,
    });
    auto& image_cmd = evk::CmdBegin();
    image_cmd.barrier(input_image, evk::ImageLayout::Undefined,
                      evk::ImageLayout::TransferDst);
    image_cmd.copy(noisy_rgba.data(), input_image,
                   noisy_rgba.size() * sizeof(float16_t));
    image_cmd.barrier(input_image, evk::ImageLayout::TransferDst,
                      evk::ImageLayout::General);
    image_cmd.barrier(output_image, evk::ImageLayout::Undefined,
                      evk::ImageLayout::General);
    int image_timestamp = image_cmd.beginTimestamp("oidn_record_image");
    denoiser.denoise(image_cmd, input_image, output_image);
    image_cmd.endTimestamp(image_timestamp);
    evk::CmdWait(image_cmd.submit());
    double gpu_record_image_ms = 0.0;
    for (const evk::TimestampEntry& timing : evk::CmdTimestamps()) {
        if (std::string(timing.name) == "oidn_record_image") {
            gpu_record_image_ms = timing.end - timing.start;
        }
    }

    save_image("oidn_denoised.bmp", width, height, denoised);

    auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    printf("pathtrace: %.3f ms\n", milliseconds(pathtrace_begin, pathtrace_end));
    printf("oidn_load: %.3f ms\n", milliseconds(load_begin, load_end));
    printf("oidn_inference: %.3f ms\n", milliseconds(inference_begin, inference_end));
    const auto& cpu_timings = denoiser.cpu_timings();
    printf("cpu_input_pack: %.3f ms\n", cpu_timings.input_pack_ms);
    printf("cpu_graph: %.3f ms\n", cpu_timings.graph_ms);
    printf("cpu_download: %.3f ms\n", cpu_timings.download_ms);
    printf("cpu_output_unpack: %.3f ms\n", cpu_timings.output_unpack_ms);
    printf("gpu_record_image: %.3f ms\n", gpu_record_image_ms);

    double gpu_graph = 0.0;
    double gpu_conv2d = 0.0;
    double gpu_relu = 0.0;
    double gpu_pool = 0.0;
    double gpu_upsample = 0.0;
    double gpu_concat = 0.0;
    for (const evk::TimestampEntry& timing : denoiser.timings()) {
        double elapsed = timing.end - timing.start;
        gpu_graph += elapsed;
        std::string name = timing.name;
        if (name.ends_with(".conv2d")) gpu_conv2d += elapsed;
        else if (name.ends_with(".relu")) gpu_relu += elapsed;
        else if (name.starts_with("pool")) gpu_pool += elapsed;
        else if (name.starts_with("upsample")) gpu_upsample += elapsed;
        else if (name.starts_with("concat")) gpu_concat += elapsed;
        printf("gpu_%s: %.3f ms\n", timing.name, elapsed);
    }
    printf("gpu_graph: %.3f ms\n", gpu_graph);
    printf("gpu_conv2d: %.3f ms\n", gpu_conv2d);
    printf("gpu_relu: %.3f ms\n", gpu_relu);
    printf("gpu_pool: %.3f ms\n", gpu_pool);
    printf("gpu_upsample: %.3f ms\n", gpu_upsample);
    printf("gpu_concat: %.3f ms\n", gpu_concat);

    double absolute_difference = 0.0;
    for (size_t i = 0; i < noisy.size(); ++i) {
        absolute_difference += std::abs(double(noisy[i] - denoised[i]));
    }
    absolute_difference /= double(noisy.size());
    printf("[oidn] Mean absolute change: %.6f\n", absolute_difference);
    printf("[oidn] Wrote oidn_noisy.bmp and oidn_denoised.bmp\n");
}
