#pragma once

#include <vector>
#include "stb_image_write.h"

inline void save_rgrid_png(const char* filename, double* Rgrid, int width, int height) {

    int scale = 10; // 1 cell = scale x scale pixels

    int out_w = width * scale;
    int out_h = height * scale;

    std::vector<unsigned char> img(out_w * out_h * 3);

    auto idx = [&](int x, int y) {
        return y * width + x;
    };

    double eps = 1e-6; // threshold for "difference"

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // --- compute 4-neighbor average ---
            double center = Rgrid[idx(x, y)];
            double sum = center;
            int count = 1;

            if (x > 0) {
                sum += Rgrid[idx(x - 1, y)];
                count++;
            }
            if (x < width - 1) {
                sum += Rgrid[idx(x + 1, y)];
                count++;
            }
            if (y > 0) {
                sum += Rgrid[idx(x, y - 1)];
                count++;
            }
            if (y < height - 1) {
                sum += Rgrid[idx(x, y + 1)];
                count++;
            }

            double avg = sum / count;

            // --- FIXED PHYSICAL COLOR SCALE ---
            double t = (avg - 1.0) / (1000.0 - 1.0);

            if (t < 0) t = 0;
            if (t > 1) t = 1;

            unsigned char r = (unsigned char)(t * 255.0);
            unsigned char g = 0;
            unsigned char b = (unsigned char)((1.0 - t) * 255.0);

            // --- render block with borders ---
            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {

                    int xx = x * scale + dx;
                    int yy = y * scale + dy;

                    int i = (yy * out_w + xx) * 3;

                    bool border = false;

                    // right boundary
                    if (dx == scale - 1 && x < width - 1) {
                        double a = Rgrid[idx(x, y)];
                        double b2 = Rgrid[idx(x + 1, y)];
                        if (fabs(a - b2) > eps) border = true;
                    }

                    // bottom boundary
                    if (dy == scale - 1 && y < height - 1) {
                        double a = Rgrid[idx(x, y)];
                        double b2 = Rgrid[idx(x, y + 1)];
                        if (fabs(a - b2) > eps) border = true;
                    }

                    if (border) {
                        img[i + 0] = 255;
                        img[i + 1] = 255;
                        img[i + 2] = 255;
                    } else {
                        img[i + 0] = r;
                        img[i + 1] = g;
                        img[i + 2] = b;
                    }
                }
            }
        }
    }

    stbi_write_png(filename, out_w, out_h, 3, img.data(), out_w * 3);
}