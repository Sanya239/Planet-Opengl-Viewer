#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>

#include "utils.hpp"
#include "stb_image.h"
typedef obj_data::vertex vertex;


std::string to_string(std::string_view str) {
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message) {
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error) {
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

const char vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 viewmodel;
uniform mat4 projection;
uniform mat3 normal_matrix;
uniform float time;
uniform int grid_size;
uniform sampler2D height_texture;
uniform mat4 light_space_matrix;

layout (location = 0) in vec2 position;

out vec3 normal;
out vec2 texcoord;
out vec3 frag_position;
out float height;
out vec4 frag_pos_light_space;

void main()
{
    float i = position.x;
    float j = position.y;

    float latitude = (i / float(grid_size - 1)) * 3.14159265359 - 1.57079632679;
    float longitude = (j / float(grid_size - 1)) * 2.0 * 3.14159265359;
    texcoord = vec2(1.0 - j / float(grid_size - 1), 1.0 - i / float(grid_size - 1));

    height = texture(height_texture, texcoord).r;

    float radius = 1.0 + 0.1 * height;
    float x = radius * cos(latitude) * cos(longitude);
    float y = radius * sin(latitude);
    float z = radius * cos(latitude) * sin(longitude);

    vec3 world_position = vec3(x, y, z);
    normal = normalize(world_position);

    frag_position = world_position;

    frag_pos_light_space = light_space_matrix * vec4(world_position, 1.0);

    gl_Position = projection * viewmodel * vec4(world_position, 1.0);
}
)";

const char shadow_vertex_shader_source[] =
        R"(#version 330 core

uniform mat4 light_space_matrix;
uniform mat4 model;
uniform int grid_size;
uniform sampler2D height_texture;

layout (location = 0) in vec2 position;

out float height;
void main()
{
    float i = position.x;
    float j = position.y;

    float latitude = (i / float(grid_size - 1)) * 3.14159265359 - 1.57079632679;
    float longitude = (j / float(grid_size - 1)) * 2.0 * 3.14159265359;
    vec2 texcoord = vec2(1.0 - j / float(grid_size - 1), 1.0 - i / float(grid_size - 1));

    height = texture(height_texture, texcoord).r;
    float radius = 1.0 + 0.1 * height;
    float x = radius * cos(latitude) * cos(longitude);
    float y = radius * sin(latitude);
    float z = radius * cos(latitude) * sin(longitude);

    vec3 world_position = vec3(x, y, z);
    gl_Position = light_space_matrix * model * vec4(world_position, 1.0);
}
)";

const char shadow_fragment_shader_source[] =
        R"(#version 330 core
in float height;
void main()
{
    if(abs(height)<0.002){
    discard;
    }
}
)";

const char fragment_shader_source[] =

    R"(#version 330 core

uniform sampler2D day_texture;
uniform sampler2D night_texture;
uniform sampler2D specular_texture;
uniform sampler2D height_texture;
uniform sampler2D shadow_map;
uniform mat4 light_space_matrix;
uniform vec3 light_direction;
uniform vec3 camera_position;
uniform float roughness;
uniform int grid_size;

in vec2 texcoord;
in vec3 frag_position;
in float height;
in vec4 frag_pos_light_space;

vec3 normal;

vec3 calculate_normal_from_height() {
    float height_scale = 5;

    float height_l = texture(height_texture, texcoord + vec2(-1.0 / float(grid_size)/2, 0.0)).r;
    float height_r = texture(height_texture, texcoord + vec2(1.0 / float(grid_size)/2, 0.0)).r;
    float height_d = texture(height_texture, texcoord + vec2(0.0, -1.0 / float(grid_size)/2)).r;
    float height_u = texture(height_texture, texcoord + vec2(0.0, 1.0 / float(grid_size)/2)).r;

    float dx = (height_r - height_l) * height_scale;
    float dy = (height_u - height_d) * height_scale;

    vec3 world_normal_base = normalize(frag_position);
    vec3 tangent = normalize(vec3(-world_normal_base.z, 0.0, world_normal_base.x));
    vec3 bitangent = cross(world_normal_base, tangent);

    vec3 normal_offset = tangent * dx + bitangent * dy;
    vec3 final_normal = normalize(world_normal_base + normal_offset);

    return final_normal;
}

float diffuse(vec3 direction) {
    return max(0.0, dot(normal, normalize(direction)));
}

float specular(vec3 direction) {
    vec3 n = normal;
    vec3 l = normalize(direction);
    vec3 reflected_direction = 2.0 * n * dot(n, l) - l;
    vec3 view_direction = normalize(camera_position - frag_position);
    float power = 1.0 / (roughness * roughness) - 1.0;
    float glossiness = texture(specular_texture, texcoord).r;
    return glossiness * pow(max(0.0, dot(reflected_direction, view_direction)), power);
}

float phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

float shadow_calculation(vec3 proj_coords) {
    proj_coords = proj_coords * 0.5 + 0.5;

    if (proj_coords.z > 1.0 || proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }

    float closest_depth = texture(shadow_map, proj_coords.xy).r;
    float current_depth = proj_coords.z;

    float bias = 0.002;

    return (current_depth - bias > closest_depth) ? 1.0 : 0.0;
}

vec3 calculate_normal_at(vec2 tex_coord) {
    float height_scale = 5.0;

    float height_l = texture(height_texture, tex_coord + vec2(-1.0 / float(grid_size)/2, 0.0)).r;
    float height_r = texture(height_texture, tex_coord + vec2(1.0 / float(grid_size)/2, 0.0)).r;
    float height_d = texture(height_texture, tex_coord + vec2(0.0, -1.0 / float(grid_size)/2)).r;
    float height_u = texture(height_texture, tex_coord + vec2(0.0, 1.0 / float(grid_size)/2)).r;

    float dx = (height_r - height_l) * height_scale;
    float dy = (height_u - height_d) * height_scale;

    float i = (1.0 - tex_coord.y) * float(grid_size - 1);
    float j = (1.0 - tex_coord.x) * float(grid_size - 1);

    float latitude = (i / float(grid_size - 1)) * 3.14159265359 - 1.57079632679;
    float longitude = (j / float(grid_size - 1)) * 2.0 * 3.14159265359;

    float height_val = texture(height_texture, tex_coord).r;
    float radius = 1.0 + 0.1 * height_val;
    float x = radius * cos(latitude) * cos(longitude);
    float y = radius * sin(latitude);
    float z = radius * cos(latitude) * sin(longitude);

    vec3 world_pos = vec3(x, y, z);
    vec3 world_normal_base = normalize(world_pos);
    vec3 tangent = normalize(vec3(-world_normal_base.z, 0.0, world_normal_base.x));
    vec3 bitangent = cross(world_normal_base, tangent);

    vec3 normal_offset = tangent * dx + bitangent * dy;
    return normalize(world_normal_base + normal_offset);
}

vec3 calculate_position_at(vec2 tex_coord) {
    float i = (1.0 - tex_coord.y) * float(grid_size - 1);
    float j = (1.0 - tex_coord.x) * float(grid_size - 1);

    float latitude = (i / float(grid_size - 1)) * 3.14159265359 - 1.57079632679;
    float longitude = (j / float(grid_size - 1)) * 2.0 * 3.14159265359;

    float height_val = texture(height_texture, tex_coord).r;
    float radius = 1.0 + 0.1 * height_val;
    float x = radius * cos(latitude) * cos(longitude);
    float y = radius * sin(latitude);
    float z = radius * cos(latitude) * sin(longitude);

    return vec3(x, y, z);
}

float phong_blurred(vec3 direction, vec2 tex_coord, vec3 pos) {
    float texel_size = 1.0 / float(grid_size)/2;
    float total_lighting = 0.0;
    int sample_size = 1;
    for (int x = -sample_size; x <= sample_size; x++) {
        for (int y = -sample_size; y <= sample_size; y++) {
            vec2 offset = vec2(float(x), float(y)) * texel_size;
            vec2 sample_coords = tex_coord + offset;

            sample_coords = clamp(sample_coords, 0.001, 0.999);

            vec3 sample_normal = calculate_normal_at(sample_coords);

            vec3 sample_pos = calculate_position_at(sample_coords);

            vec4 sample_light_space = light_space_matrix * vec4(sample_pos, 1.0);
            vec3 sample_proj_coords = sample_light_space.xyz / sample_light_space.w;

            float sample_shadow = shadow_calculation(sample_proj_coords);

            vec3 n = sample_normal;
            vec3 l = normalize(direction);
            float sample_diffuse = max(0.0, dot(n, l));

            vec3 reflected_direction = 2.0 * n * dot(n, l) - l;
            vec3 view_dir = normalize(camera_position - sample_pos);
            float power = 1.0 / (roughness * roughness) - 1.0;
            float sample_specular = 0.5 * pow(max(0.0, dot(reflected_direction, view_dir)), power);

            float sample_lighting = (sample_diffuse + sample_specular) * (1.0 - sample_shadow);

            float weight = 1.0;
            int abs_x = abs(x);
            int abs_y = abs(y);

            if (abs_x + abs_y == 2) {
                weight = 1.0;
            } else if (abs_x + abs_y == 1) {
                weight = 2.0;
            } else {
                weight = 4.0;
            }

            total_lighting += sample_lighting * weight;
        }
    }

    total_lighting /= 16.0;

    return total_lighting;
}

layout (location = 0) out vec4 out_color;

void main()
{
    normal = calculate_normal_from_height();

    float ambient= 0.1;
    float lighting = phong_blurred(light_direction, texcoord, frag_position) + ambient;

    vec4 albedo_tex;
    float day = max(0.0, min(1.0, 5*(lighting-ambient)));
    float night = 1- day;

    albedo_tex = texture(night_texture, texcoord)*10*night + texture(day_texture, texcoord)*day;

    vec3 albedo = albedo_tex.rgb;
    out_color = vec4(lighting * albedo, albedo_tex.a);
}
)";

GLuint create_shader(GLenum type, const char *source) {
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader) {
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}

void config_array(const obj_data &object, GLuint *vao, GLuint *vbo, GLuint *ebo) {
    glCreateVertexArrays(1, vao);
    glBindVertexArray(*vao);
    glCreateBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    glCreateBuffers(1, ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ebo);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *) 0);

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex) * object.vertices.size(), object.vertices.data(), GL_STATIC_DRAW);

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * object.indices.size(), object.indices.data(),
                 GL_STATIC_DRAW);
}


GLuint create_texture(std::string night_texture_path, bool rgba = true) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int width, height;
    auto data = stbi_load(night_texture_path.c_str(), &width, &height, NULL, rgba ? 4 : 1);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, rgba ? GL_RGBA8 : GL_R8, width, height, 0, rgba ? GL_RGBA : GL_R, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);
    }
    return texture;
}

int main() try {
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window *window = SDL_CreateWindow("Graphics course practice 5",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          800, 600,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.8f, 0.8f, 1.f, 0.f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    auto shadow_vertex_shader = create_shader(GL_VERTEX_SHADER, shadow_vertex_shader_source);
    auto shadow_fragment_shader = create_shader(GL_FRAGMENT_SHADER, shadow_fragment_shader_source);
    auto shadow_program = create_program(shadow_vertex_shader, shadow_fragment_shader);

    GLuint viewmodel_location = glGetUniformLocation(program, "viewmodel");
    GLuint projection_location = glGetUniformLocation(program, "projection");

    GLuint day_texture_location = glGetUniformLocation(program, "day_texture");
    GLuint night_texture_location = glGetUniformLocation(program, "night_texture");
    GLuint specular_texture_location = glGetUniformLocation(program, "specular_texture");
    GLuint height_texture_location = glGetUniformLocation(program, "height_texture");
    GLuint time_location = glGetUniformLocation(program, "time");
    GLuint grid_size_location = glGetUniformLocation(program, "grid_size");
    GLuint light_direction_location = glGetUniformLocation(program, "light_direction");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint roughness_location = glGetUniformLocation(program, "roughness");
    GLuint shadow_map_location = glGetUniformLocation(program, "shadow_map");
    GLuint light_space_matrix_location = glGetUniformLocation(program, "light_space_matrix");

    GLuint shadow_light_space_matrix_location = glGetUniformLocation(shadow_program, "light_space_matrix");
    GLuint shadow_model_location = glGetUniformLocation(shadow_program, "model");
    GLuint shadow_grid_size_location = glGetUniformLocation(shadow_program, "grid_size");
    GLuint shadow_height_texture_location = glGetUniformLocation(shadow_program, "height_texture");

    std::string project_root = PROJECT_ROOT;
    std::string day_texture_path = project_root + "/res/earth_day.jpg";
    std::string night_texture_path = project_root + "/res/earth_night.jpg";

    std::string specular_texture_path = project_root + "res/earth_specular.jpg";
    std::string height_texture_path = project_root + "/res/earth_height.jpg";

    long long grid_size = 1000;

    obj_data planet_data = make_planet_grid(grid_size);

    GLuint vao, vbo, ebo;
    config_array(planet_data, &vao, &vbo, &ebo);

    GLuint day_texture = create_texture(day_texture_path);

    GLuint night_texture = create_texture(night_texture_path);

    GLuint height_texture = create_texture(height_texture_path);

    GLuint specular_texture = create_texture(specular_texture_path);

    const unsigned int SHADOW_WIDTH = 16384, SHADOW_HEIGHT = 16384;
    GLuint shadow_map_fbo;
    glGenFramebuffers(1, &shadow_map_fbo);

    GLuint shadow_map_texture;
    glGenTextures(1, &shadow_map_texture);
    glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

    glBindFramebuffer(GL_FRAMEBUFFER, shadow_map_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_map_texture, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    auto last_frame_start = std::chrono::high_resolution_clock::now();

    float time = 0.f;
    float time_speed = 0.2f;

    std::map<SDL_Keycode, bool> button_down;

    float near = 0.05f;
    float far = 100.f;
   glm::vec3 camera_position(0.f, 0.f, 2.f);
    float camera_yaw = glm::pi<float>();
    float camera_pitch = 0.0f;
    float quickness = far / 100;
    bool running = true;
    while (running) {
        for (SDL_Event event; SDL_PollEvent(&event);)
            switch (event.type) {
                case SDL_QUIT:
                    running = false;
                    break;
                case SDL_WINDOWEVENT:
                    switch (event.window.event) {
                        case SDL_WINDOWEVENT_RESIZED:
                            width = event.window.data1;
                            height = event.window.data2;
                            break;
                    }
                    break;
                case SDL_KEYDOWN:
                    button_down[event.key.keysym.sym] = true;
                    break;
                case SDL_KEYUP:
                    button_down[event.key.keysym.sym] = false;
                    break;
            }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration_cast<std::chrono::duration<float> >(now - last_frame_start).count() * time_speed;
        last_frame_start = now;
        time += dt;

        if (button_down[SDLK_LEFT]) camera_yaw += 1.5f * dt;
        if (button_down[SDLK_RIGHT]) camera_yaw -= 1.5f * dt;
        if (button_down[SDLK_UP]) camera_pitch += 1.0f * dt;
        if (button_down[SDLK_DOWN]) camera_pitch -= 1.0f * dt;

        camera_pitch = glm::clamp(camera_pitch, -glm::half_pi<float>() + 0.1f, glm::half_pi<float>() - 0.1f);

        glm::vec3 camera_front(
            cos(camera_pitch) * sin(camera_yaw),
            sin(camera_pitch),
            cos(camera_pitch) * cos(camera_yaw)
        );
        camera_front = glm::normalize(camera_front);

        glm::vec3 camera_right = glm::normalize(glm::cross(camera_front, glm::vec3(0, 1, 0)));

        float speed = quickness * dt;

        if (button_down[SDLK_w]) camera_position += camera_front * speed;
        if (button_down[SDLK_s]) camera_position -= camera_front * speed;
        if (button_down[SDLK_a]) camera_position -= camera_right * speed;
        if (button_down[SDLK_d]) camera_position += camera_right * speed;
        if (button_down[SDLK_SPACE]) camera_position.y += speed;
        if (button_down[SDLK_LCTRL]) camera_position.y -= speed;

        glm::mat4 view = glm::lookAt(camera_position, camera_position + camera_front, glm::vec3(0, 1, 0));

        glm::mat4 model(1);


        glm::mat4 projection = glm::perspective(glm::pi<float>() / 3.f, (width * 1.f) / height, near, far);

        float light_height = -0.7f;
        float light_radius = 3.0f;
        float light_angle = time * 0.5f;

        glm::vec3 light_position(
            light_radius * std::cos(light_angle),
            light_height,
            light_radius * std::sin(light_angle)
        );

        glm::vec3 light_direction = glm::normalize(light_position);

        float light_near = 0.1f;
        float light_far = 10.0f;
        glm::mat4 light_projection = glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, light_near, light_far);
        glm::mat4 light_view = glm::lookAt(light_position, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 light_space_matrix = light_projection * light_view;

        glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_map_fbo);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(shadow_program);
        glUniformMatrix4fv(shadow_light_space_matrix_location, 1, GL_FALSE, glm::value_ptr(light_space_matrix));
        glUniformMatrix4fv(shadow_model_location, 1, GL_FALSE, glm::value_ptr(model));
        glUniform1i(shadow_grid_size_location, grid_size);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, height_texture);
        glUniform1i(shadow_height_texture_location, 0);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, planet_data.indices.size(), GL_UNSIGNED_INT, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program);
        glUniformMatrix4fv(viewmodel_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(light_direction_location, 1, glm::value_ptr(light_direction));
        glUniform3fv(camera_position_location, 1, glm::value_ptr(camera_position));
        glUniform1f(roughness_location, 0.5f);
        glUniform1f(time_location, time / 10);
        glUniform1i(grid_size_location, grid_size);
        glUniformMatrix4fv(light_space_matrix_location, 1, GL_FALSE, glm::value_ptr(light_space_matrix));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, day_texture);
        glUniform1i(day_texture_location, 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, night_texture);
        glUniform1i(night_texture_location, 1);

        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, specular_texture);
        glUniform1i(specular_texture_location, 2);

        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, height_texture);
        glUniform1i(height_texture_location, 3);

        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, shadow_map_texture);
        glUniform1i(shadow_map_location, 4);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, planet_data.indices.size(), GL_UNSIGNED_INT, 0);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
} catch (std::exception const &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
