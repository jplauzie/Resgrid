#pragma once

#include <vector>
#include "stb_image_write.h"

inline void save_rgrid_png(const char* filename, double* Rgrid, int width, int height) {

    int scale = 2;
    int out_w = width * scale;
    int out_h = height * scale;

    std::vector<unsigned char> img(out_w * out_h * 3);

    auto idx = [&](int x, int y) {
        return y * width + x;
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            double center = Rgrid[idx(x, y)];
            double sum = center;
            int count = 1;

            if (x > 0) sum += Rgrid[idx(x - 1, y)], count++;
            if (x < width - 1) sum += Rgrid[idx(x + 1, y)], count++;
            if (y > 0) sum += Rgrid[idx(x, y - 1)], count++;
            if (y < height - 1) sum += Rgrid[idx(x, y + 1)], count++;

            double avg = sum / count;

            double t = (avg - 1.0) / (1000.0 - 1.0);
            if (t < 0) t = 0;
            if (t > 1) t = 1;

            // flipped color example (blue ↔ red swap if desired)
            unsigned char r = (unsigned char)(t * 255.0);
            unsigned char g = 0;
            unsigned char b = (unsigned char)((1.0 - t) * 255.0);

            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {

                    int xx = x * scale + dx;
                    int yy = y * scale + dy;

                    int i = (yy * out_w + xx) * 3;

                    img[i + 0] = r;
                    img[i + 1] = g;
                    img[i + 2] = b;
                }
            }
        }
    }

    stbi_write_png(filename, out_w, out_h, 3, img.data(), out_w * 3);
}