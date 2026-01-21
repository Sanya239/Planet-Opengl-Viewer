#pragma once

#include <array>
#include <vector>
#include <filesystem>

struct obj_data {
    struct vertex {
        std::array<float, 2> position; // номер строки и столбца
    };

    std::vector<vertex> vertices;
    std::vector<std::uint32_t> indices;
};

obj_data make_planet_grid(long long grid_size) {
    obj_data result;
    
    // Создаем вершины с номерами строки и столбца
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            obj_data::vertex v;
            v.position[0] = float(i); // номер строки
            v.position[1] = float(j); // номер столбца
            result.vertices.push_back(v);
        }
    }
    
    // Создаем индексы для треугольников
    for (int i = 0; i < grid_size - 1; ++i) {
        for (int j = 0; j < grid_size - 1; ++j) {
            uint32_t v00 = i * grid_size + j;
            uint32_t v10 = (i + 1) * grid_size + j;
            uint32_t v01 = i * grid_size + (j + 1);
            uint32_t v11 = (i + 1) * grid_size + (j + 1);

            result.indices.push_back(v00);
            result.indices.push_back(v10);
            result.indices.push_back(v01);

            result.indices.push_back(v01);
            result.indices.push_back(v10);
            result.indices.push_back(v11);
        }
    }
    
    return result;
}
