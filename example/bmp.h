#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

struct BMP {
    int width;
    int height;
    uint8_t* pixels;  // RGB format, 3 bytes per pixel

    BMP(int w, int h) : width(w), height(h) {
        pixels = new uint8_t[w * h * 3];
        clear(0, 0, 0);
    }

    ~BMP() {
        delete[] pixels;
    }

    void clear(uint8_t r, uint8_t g, uint8_t b) {
        for (int i = 0; i < width * height; ++i) {
            pixels[i * 3 + 0] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }
    }

    void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= width || y < 0 || y >= height) return;
        int idx = (y * width + x) * 3;
        pixels[idx + 0] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
    }

    void draw_point(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= radius * radius) {
                    set_pixel(cx + dx, cy + dy, r, g, b);
                }
            }
        }
    }

    void draw_circle(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
        for (int y = cy - radius - thickness; y <= cy + radius + thickness; ++y) {
            for (int x = cx - radius - thickness; x <= cx + radius + thickness; ++x) {
                int dx = x - cx;
                int dy = y - cy;
                float dist = sqrtf(float(dx * dx + dy * dy));
                if (fabsf(dist - float(radius)) <= float(thickness)) {
                    set_pixel(x, y, r, g, b);
                }
            }
        }
    }

    void draw_cross(int cx, int cy, int size, uint8_t r, uint8_t g, uint8_t b) {
        for (int i = -size; i <= size; ++i) {
            set_pixel(cx + i, cy, r, g, b);
            set_pixel(cx, cy + i, r, g, b);
        }
    }

    void draw_line(int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true) {
            set_pixel(x0, y0, r, g, b);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    void draw_grid(int spacing, uint8_t r, uint8_t g, uint8_t b) {
        for (int x = 0; x < width; x += spacing) {
            for (int y = 0; y < height; ++y) {
                set_pixel(x, y, r, g, b);
            }
        }
        for (int y = 0; y < height; y += spacing) {
            for (int x = 0; x < width; ++x) {
                set_pixel(x, y, r, g, b);
            }
        }
    }

    void draw_rotated_square(int cx, int cy, int half_size, float rotation, uint8_t r, uint8_t g, uint8_t b, int thickness = 2) {
        float cos_r = cosf(rotation);
        float sin_r = sinf(rotation);
        
        float corners[4][2] = {
            {-float(half_size), -float(half_size)},
            { float(half_size), -float(half_size)},
            { float(half_size),  float(half_size)},
            {-float(half_size),  float(half_size)}
        };
        
        int screen_corners[4][2];
        for (int i = 0; i < 4; ++i) {
            float rx = corners[i][0] * cos_r - corners[i][1] * sin_r;
            float ry = corners[i][0] * sin_r + corners[i][1] * cos_r;
            screen_corners[i][0] = cx + int(rx);
            screen_corners[i][1] = cy + int(ry);
        }
        
        for (int t = -thickness/2; t <= thickness/2; ++t) {
            for (int i = 0; i < 4; ++i) {
                int next = (i + 1) % 4;
                draw_line(screen_corners[i][0] + t, screen_corners[i][1],
                          screen_corners[next][0] + t, screen_corners[next][1], r, g, b);
                draw_line(screen_corners[i][0], screen_corners[i][1] + t,
                          screen_corners[next][0], screen_corners[next][1] + t, r, g, b);
            }
        }
    }

    bool save(const char* filename) {
        FILE* f = fopen(filename, "wb");
        if (!f) return false;

        int row_padding = (4 - (width * 3) % 4) % 4;
        int row_size = width * 3 + row_padding;
        int pixel_data_size = row_size * height;
        int file_size = 54 + pixel_data_size;

        // BMP Header (14 bytes)
        uint8_t header[54] = {0};
        header[0] = 'B';
        header[1] = 'M';
        *(uint32_t*)&header[2] = file_size;
        *(uint32_t*)&header[10] = 54;  // Pixel data offset

        // DIB Header (40 bytes)
        *(uint32_t*)&header[14] = 40;  // DIB header size
        *(int32_t*)&header[18] = width;
        *(int32_t*)&header[22] = height;
        *(uint16_t*)&header[26] = 1;   // Color planes
        *(uint16_t*)&header[28] = 24;  // Bits per pixel
        *(uint32_t*)&header[34] = pixel_data_size;

        fwrite(header, 1, 54, f);

        // Write pixel data (BMP stores bottom-to-top, BGR order)
        uint8_t padding[3] = {0, 0, 0};
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                uint8_t bgr[3] = {pixels[idx + 2], pixels[idx + 1], pixels[idx + 0]};
                fwrite(bgr, 1, 3, f);
            }
            if (row_padding > 0) {
                fwrite(padding, 1, row_padding, f);
            }
        }

        fclose(f);
        return true;
    }
};

// Helper class for world-to-screen coordinate mapping
struct ImageMapper {
    float world_min_x, world_max_x;
    float world_min_y, world_max_y;
    int img_width, img_height;
    int margin;

    ImageMapper(float min_x, float max_x, float min_y, float max_y, int w, int h, int m = 20)
        : world_min_x(min_x), world_max_x(max_x)
        , world_min_y(min_y), world_max_y(max_y)
        , img_width(w), img_height(h), margin(m) {}

    int to_screen_x(float wx) const {
        float t = (wx - world_min_x) / (world_max_x - world_min_x);
        return margin + int(t * (img_width - 2 * margin));
    }

    int to_screen_y(float wy) const {
        float t = (wy - world_min_y) / (world_max_y - world_min_y);
        return margin + int((1.0f - t) * (img_height - 2 * margin));  // Flip Y
    }

    int to_screen_radius(float wr) const {
        float scale_x = float(img_width - 2 * margin) / (world_max_x - world_min_x);
        float scale_y = float(img_height - 2 * margin) / (world_max_y - world_min_y);
        float scale = (std::min)(scale_x, scale_y);
        return int(wr * scale);
    }
};

