#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <zlib.h>
#include <lzma.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

struct Layer {
    int in_f, out_f;
    std::vector<float> weights;
    std::vector<float> bias;
};

inline float dot_product(const float* input, const float* weights, int n) {
#ifdef __AVX2__
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v_in = _mm256_loadu_ps(&input[i]);
        __m256 v_w  = _mm256_loadu_ps(&weights[i]);
        sum = _mm256_fmadd_ps(v_in, v_w, sum);
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
    float total = _mm_cvtss_f32(sum128);
    
    for (; i < n; ++i) total += input[i] * weights[i];
    return total;
#else
    float total = 0.0f;
    #pragma omp simd reduction(+:total)
    for (int i = 0; i < n; ++i) {
        total += input[i] * weights[i];
    }
    return total;
#endif
}

class PointPredictor {
public:
    std::vector<Layer> layers;
    int max_h = 0;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        char magic[4];
        f.read(magic, 4);
        if (strncmp(magic, "LIZM", 4) != 0) return false;

        uint32_t num_layers;
        f.read((char*)&num_layers, 4);

        for (uint32_t i = 0; i < num_layers; ++i) {
            uint32_t in_f, out_f;
            f.read((char*)&in_f, 4);
            f.read((char*)&out_f, 4);

            Layer l;
            l.in_f = in_f;
            l.out_f = out_f;
            l.weights.resize(in_f * out_f);
            l.bias.resize(out_f);

            f.read((char*)l.weights.data(), l.weights.size() * 4);
            f.read((char*)l.bias.data(), l.bias.size() * 4);
            layers.push_back(l);
            
            if ((int)out_f > max_h) max_h = out_f;
            if ((int)in_f > max_h) max_h = in_f;
        }
        return true;
    }

    void predict(const float* input, float* output) const {
        // Use two alternating buffers for hidden layers
        std::vector<float> b1(max_h, 0.0f);
        std::vector<float> b2(max_h, 0.0f);
        
        const float* current_in = input;
        float* current_out = b1.data();

        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& l = layers[i];
            bool is_last = (i == layers.size() - 1);
            float* target = is_last ? output : current_out;

            for (int r = 0; r < (int)l.out_f; ++r) {
                float sum = l.bias[r] + dot_product(current_in, &l.weights[r * l.in_f], l.in_f);
                target[r] = is_last ? sum : std::max(0.0f, sum);
            }

            if (!is_last) {
                current_in = target;
                current_out = (target == b1.data()) ? b2.data() : b1.data();
            }
        }
    }
};

inline uint64_t part1by2(uint64_t n) {
    n &= 0x1fffff;
    n = (n | (n << 32)) & 0x1f00000000ffffULL;
    n = (n | (n << 16)) & 0x1f0000ff0000ffULL;
    n = (n | (n << 8))  & 0x100f00f00f00f00fULL;
    n = (n | (n << 4))  & 0x10c30c30c30c30c3ULL;
    n = (n | (n << 2))  & 0x1249249249249249ULL;
    return n;
}

struct Point {
    float x, y, z;
};

struct VoxelKey {
    uint64_t code;
    uint32_t id;
    bool operator<(const VoxelKey& o) const {
        return code < o.code;
    }
};

void voxel_sort(std::vector<Point>& points, float grid = 0.10f) {
    if (points.empty()) return;

    // Find min bounds for quantization
    float min_x = points[0].x;
    float min_y = points[0].y;
    float min_z = points[0].z;
    for (const auto& p : points) {
        if (p.x < min_x) min_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.z < min_z) min_z = p.z;
    }

    std::vector<VoxelKey> keys(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        uint64_t vx = (uint64_t)((points[i].x - min_x) / grid);
        uint64_t vy = (uint64_t)((points[i].y - min_y) / grid);
        uint64_t vz = (uint64_t)((points[i].z - min_z) / grid);
        
        keys[i].code = part1by2(vx) | (part1by2(vy) << 1) | (part1by2(vz) << 2);
        keys[i].id = (uint32_t)i;
    }
    std::stable_sort(keys.begin(), keys.end());
    std::vector<Point> sorted(points.size());
    for (size_t i = 0; i < points.size(); ++i) sorted[i] = points[keys[i].id];
    points = sorted;
}

std::vector<uint8_t> shuffle_bytes(const int32_t* src, size_t count) {
    std::vector<uint8_t> dst(count * 4);
    const uint8_t* p = (const uint8_t*)src;
    for (size_t i = 0; i < count; ++i) {
        dst[i] = p[i * 4 + 0];
        dst[i + count] = p[i * 4 + 1];
        dst[i + 2 * count] = p[i * 4 + 2];
        dst[i + 3 * count] = p[i * 4 + 3];
    }
    return dst;
}

std::vector<int32_t> unshuffle_bytes(const uint8_t* src, size_t count) {
    std::vector<int32_t> dst(count);
    uint8_t* p = (uint8_t*)dst.data();
    for (size_t i = 0; i < count; ++i) {
        p[i * 4 + 0] = src[i];
        p[i * 4 + 1] = src[i + count];
        p[i * 4 + 2] = src[i + 2 * count];
        p[i * 4 + 3] = src[i + 3 * count];
    }
    return dst;
}

std::vector<uint8_t> compress_zlib(const std::vector<uint8_t>& src) {
    std::vector<uint8_t> dst;
    dst.resize(compressBound(src.size()));
    uLongf dest_len = dst.size();
    if (compress(dst.data(), &dest_len, src.data(), src.size()) != Z_OK) return {};
    dst.resize(dest_len);
    return dst;
}

std::vector<uint8_t> decompress_zlib(const uint8_t* src, size_t src_len, size_t expected_len) {
    std::vector<uint8_t> dst(expected_len);
    uLongf dest_len = expected_len;
    if (uncompress(dst.data(), &dest_len, src, src_len) != Z_OK) return {};
    return dst;
}

std::vector<uint8_t> compress_lzma(const std::vector<uint8_t>& src) {
    std::vector<uint8_t> dst(src.size() + src.size() / 2 + 131072);
    
    lzma_mt mt_options = { 0 };
    mt_options.flags = 0;
    mt_options.block_size = 0; 
    mt_options.timeout = 0;
    mt_options.preset = 9;
    mt_options.filters = NULL;
    mt_options.check = LZMA_CHECK_CRC64;
    #pragma omp parallel
    {
        #pragma omp single
        mt_options.threads = omp_get_num_threads();
    }
    
    if (mt_options.threads == 0) mt_options.threads = 1;

    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_stream_encoder_mt(&strm, &mt_options) != LZMA_OK) {
        if (lzma_easy_encoder(&strm, 9, LZMA_CHECK_CRC64) != LZMA_OK) return {};
    }
    
    strm.next_in = src.data();
    strm.avail_in = src.size();
    strm.next_out = dst.data();
    strm.avail_out = dst.size();
    
    lzma_ret ret = lzma_code(&strm, LZMA_FINISH);
    
    if (ret != LZMA_STREAM_END) {
        lzma_end(&strm);
        return {};
    }
    
    dst.resize(strm.total_out);
    lzma_end(&strm);
    return dst;
}

std::vector<uint8_t> decompress_lzma(const uint8_t* src, size_t src_len, size_t expected_len) {
    std::vector<uint8_t> dst(expected_len);
    size_t in_pos = 0;
    size_t out_pos = 0;
    uint64_t mem_limit = UINT64_MAX;
    
    if (lzma_stream_buffer_decode(&mem_limit, 0, NULL, src, &in_pos, src_len,
                                  dst.data(), &out_pos, dst.size()) != LZMA_OK) {
        return {};
    }
    return dst;
}

struct LizipHeader {
    char magic[4];
    uint8_t comp_id; // 0=None, 1=zlib, 2=lzma
    uint8_t reserved[3];
    uint32_t num_points;
    uint32_t num_blocks;
    float scale;
    uint32_t type_flag;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "LiZIP Pure C++ Engine\nUsage: lizip <e|d> <input> <output> [model.bin] [zlib|lzma|none]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string in_path = argv[2];
    std::string out_path = argv[3];
    std::string model_path = (argc > 4) ? argv[4] : "models/mlp.bin";
    std::string comp_method = (argc > 5) ? argv[5] : "lzma";

    PointPredictor model;
    if (!model.load(model_path)) {
        std::cerr << "Error: Could not load model " << model_path << "\n";
        return 1;
    }

    const float SCALE = 100000.0f;
    const int CONTEXT = 5;
    const int BLOCK = 128;

    if (mode == "e") {
        auto start_all = std::chrono::high_resolution_clock::now();
        
        std::ifstream f(in_path, std::ios::binary);
        if (!f) { std::cerr << "Failed to open input: " << in_path << "\n"; return 1; }
        
        f.seekg(0, std::ios::end);
        size_t fsize = f.tellg();
        f.seekg(0, std::ios::beg);
        
        size_t num_pts = fsize / 20;
        std::vector<char> buffer(fsize);
        if (!f.read(buffer.data(), fsize)) {
            std::cerr << "Failed to read file\n";
            return 1;
        }

        std::vector<Point> points(num_pts);
        
        for (int i = 0; i < (int)num_pts; ++i) {
            float* b = (float*)&buffer[i * 20];
            points[i] = {b[0], b[1], b[2]};
        }
        
        voxel_sort(points);

        int n = points.size();
        int nb = (n + BLOCK - 1) / BLOCK;
        std::vector<int32_t> heads(nb * CONTEXT * 3);
        std::vector<int32_t> resids(nb * (BLOCK - CONTEXT) * 3, 0);

        #pragma omp parallel for
        for (int i = 0; i < nb; ++i) {
            int start = i * BLOCK;
            float ctx[15] = {0};
            for (int j = 0; j < CONTEXT; ++j) {
                int idx = start + j;
                if (idx < n) {
                    heads[i * CONTEXT * 3 + j * 3 + 0] = (int32_t)std::round(points[idx].x * SCALE);
                    heads[i * CONTEXT * 3 + j * 3 + 1] = (int32_t)std::round(points[idx].y * SCALE);
                    heads[i * CONTEXT * 3 + j * 3 + 2] = (int32_t)std::round(points[idx].z * SCALE);
                }
                ctx[j * 3 + 0] = heads[i * CONTEXT * 3 + j * 3 + 0] / SCALE;
                ctx[j * 3 + 1] = heads[i * CONTEXT * 3 + j * 3 + 1] / SCALE;
                ctx[j * 3 + 2] = heads[i * CONTEXT * 3 + j * 3 + 2] / SCALE;
            }

            for (int j = 0; j < (BLOCK - CONTEXT); ++j) {
                int idx = start + CONTEXT + j;
                if (idx >= n) break;

                float pred[3];
                model.predict(ctx, pred);

                int32_t t_mm[3] = {(int32_t)std::round(points[idx].x * SCALE), 
                                   (int32_t)std::round(points[idx].y * SCALE), 
                                   (int32_t)std::round(points[idx].z * SCALE)};
                int32_t p_mm[3] = {(int32_t)std::round(pred[0] * SCALE), 
                                   (int32_t)std::round(pred[1] * SCALE), 
                                   (int32_t)std::round(pred[2] * SCALE)};

                resids[i * (BLOCK - CONTEXT) * 3 + j * 3 + 0] = t_mm[0] - p_mm[0];
                resids[i * (BLOCK - CONTEXT) * 3 + j * 3 + 1] = t_mm[1] - p_mm[1];
                resids[i * (BLOCK - CONTEXT) * 3 + j * 3 + 2] = t_mm[2] - p_mm[2];

                memmove(ctx, ctx + 3, 12 * 4);
                ctx[12] = (float)t_mm[0] / SCALE;
                ctx[13] = (float)t_mm[1] / SCALE;
                ctx[14] = (float)t_mm[2] / SCALE;
            }
        }

        std::vector<int32_t> heads_delta = heads;
        for (size_t i = heads.size() - 1; i > 2; --i) heads_delta[i] -= heads_delta[i - 3];

        auto s_heads = shuffle_bytes(heads_delta.data(), heads_delta.size());
        auto s_resids = shuffle_bytes(resids.data(), resids.size());

        std::vector<uint8_t> raw_payload;
        raw_payload.insert(raw_payload.end(), s_heads.begin(), s_heads.end());
        raw_payload.insert(raw_payload.end(), s_resids.begin(), s_resids.end());

        // Measure intermediate compression stages
        std::vector<int32_t> raw_ints(n * 3);
        for(int i=0; i<n; ++i) {
            raw_ints[i*3+0] = (int32_t)std::round(points[i].x * SCALE);
            raw_ints[i*3+1] = (int32_t)std::round(points[i].y * SCALE);
            raw_ints[i*3+2] = (int32_t)std::round(points[i].z * SCALE);
        }
        std::vector<uint8_t> stage1_data((uint8_t*)raw_ints.data(), (uint8_t*)raw_ints.data() + raw_ints.size() * 4);
        
        std::vector<int32_t> combined_resids = heads;
        combined_resids.insert(combined_resids.end(), resids.begin(), resids.end());
        std::vector<uint8_t> stage2_data((uint8_t*)combined_resids.data(), (uint8_t*)combined_resids.data() + combined_resids.size() * 4);

        size_t s1_size = 0, s2_size = 0;
        if (comp_method == "lzma") {
            s1_size = compress_lzma(stage1_data).size();
            s2_size = compress_lzma(stage2_data).size();
        } else if (comp_method == "zlib") {
            s1_size = compress_zlib(stage1_data).size();
            s2_size = compress_zlib(stage2_data).size();
        }

        std::vector<uint8_t> compressed;
        uint8_t comp_id = 0;
        if (comp_method == "zlib") {
            compressed = compress_zlib(raw_payload);
            comp_id = 1;
        } else if (comp_method == "lzma") {
            compressed = compress_lzma(raw_payload);
            comp_id = 2;
        } else {
            compressed = raw_payload;
            comp_id = 0;
        }

        if (compressed.empty() && comp_id != 0) {
            std::cerr << "Compression failed!\n";
            return 1;
        }

        std::ofstream out(out_path, std::ios::binary);
        LizipHeader h;
        memcpy(h.magic, "LIZP", 4);
        h.comp_id = comp_id;
        memset(h.reserved, 0, 3);
        h.reserved[0] = (uint8_t)CONTEXT;
        h.num_points = n;
        h.num_blocks = nb;
        h.scale = SCALE;
        h.type_flag = 3;

        out.write((char*)&h, sizeof(h));
        out.write((char*)compressed.data(), compressed.size());
        
        auto end_all = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d = end_all - start_all;
        
        std::cout << "Encoded " << n << " points in " << d.count() << "s\n";
        std::cout << "BREAKDOWN:" << "\n";
        std::cout << "  Raw_Float_Size: " << (n * 20) / 1024 << " KB\n";
        std::cout << "  Quantized_Int_Size: " << (n * 12) / 1024 << " KB\n";
        std::cout << "  Stage1_Entropy_Only: " << s1_size / 1024 << " KB\n";
        std::cout << "  Stage2_MLP_Residuals: " << s2_size / 1024 << " KB\n";
        std::cout << "  Stage3_Final_Shuffled: " << compressed.size() / 1024 << " KB\n";

    } 
    else if (mode == "d") {
        std::ifstream f(in_path, std::ios::binary | std::ios::ate);
        if (!f) { std::cerr << "Failed to open input: " << in_path << "\n"; return 1; }
        
        size_t file_size = f.tellg();
        f.seekg(0, std::ios::beg);

        LizipHeader h;
        f.read((char*)&h, sizeof(h));

        if (strncmp(h.magic, "LIZP", 4) != 0) {
            f.seekg(0, std::ios::beg);
            uint32_t header[4];
            f.read((char*)header, 16);
            h.num_points = header[0];
            h.num_blocks = header[1];
            h.scale = (float)header[2];
            h.type_flag = header[3];
            h.comp_id = 1;
            std::cout << "Legacy format detected. Assuming zlib.\n";
        }

        size_t header_size = (strncmp(h.magic, "LIZP", 4) == 0) ? sizeof(LizipHeader) : 16;
        size_t compressed_size = file_size - header_size;
        
        f.seekg(header_size, std::ios::beg);
        std::vector<uint8_t> compressed(compressed_size);
        f.read((char*)compressed.data(), compressed.size());

        size_t expected_raw = (h.num_blocks * CONTEXT * 3 * 4) + (h.num_blocks * (BLOCK - CONTEXT) * 3 * 4);
        std::vector<uint8_t> raw_payload;

        if (h.comp_id == 1) {
            raw_payload = decompress_zlib(compressed.data(), compressed.size(), expected_raw);
        } else if (h.comp_id == 2) {
            raw_payload = decompress_lzma(compressed.data(), compressed.size(), expected_raw);
        } else {
            raw_payload = compressed;
        }

        if (raw_payload.size() != expected_raw) {
            std::cerr << "Decompression failed or size mismatch.\n";
            return 1;
        }

        const uint8_t* p_heads = raw_payload.data();
        const uint8_t* p_resids = p_heads + (h.num_blocks * CONTEXT * 3 * 4);

        auto h_delta = unshuffle_bytes(p_heads, h.num_blocks * CONTEXT * 3);
        for (size_t i = 3; i < h_delta.size(); ++i) h_delta[i] += h_delta[i - 3];

        auto resids = unshuffle_bytes(p_resids, h.num_blocks * (BLOCK - CONTEXT) * 3);

        std::vector<Point> rec(h.num_points);
        #pragma omp parallel for
        for (int i = 0; i < (int)h.num_blocks; ++i) {
            int start = i * BLOCK;
            float ctx[15];
            for (int j = 0; j < CONTEXT; ++j) {
                int idx = start + j;
                int32_t val[3] = {h_delta[i * CONTEXT * 3 + j * 3 + 0],
                                  h_delta[i * CONTEXT * 3 + j * 3 + 1],
                                  h_delta[i * CONTEXT * 3 + j * 3 + 2]};
                if (idx < (int)h.num_points) {
                    rec[idx] = {(float)val[0] / h.scale, (float)val[1] / h.scale, (float)val[2] / h.scale};
                }
                ctx[j * 3 + 0] = (float)val[0] / h.scale;
                ctx[j * 3 + 1] = (float)val[1] / h.scale;
                ctx[j * 3 + 2] = (float)val[2] / h.scale;
            }

            for (int j = 0; j < (BLOCK - CONTEXT); ++j) {
                int idx = start + CONTEXT + j;
                if (idx >= (int)h.num_points) break;

                float pred[3];
                model.predict(ctx, pred);

                int32_t p_mm[3] = {(int32_t)std::round(pred[0] * h.scale), 
                                   (int32_t)std::round(pred[1] * h.scale), 
                                   (int32_t)std::round(pred[2] * h.scale)};
                int32_t r_mm[3] = {resids[i * (BLOCK - CONTEXT) * 3 + j * 3 + 0],
                                   resids[i * (BLOCK - CONTEXT) * 3 + j * 3 + 1],
                                   resids[i * (BLOCK - CONTEXT) * 3 + j * 3 + 2]};

                int32_t final_mm[3] = {p_mm[0] + r_mm[0], p_mm[1] + r_mm[1], p_mm[2] + r_mm[2]};
                rec[idx] = {(float)final_mm[0] / h.scale, (float)final_mm[1] / h.scale, (float)final_mm[2] / h.scale};

                memmove(ctx, ctx + 3, 12 * 4);
                ctx[12] = rec[idx].x;
                ctx[13] = rec[idx].y;
                ctx[14] = rec[idx].z;
            }
        }

        std::ofstream out(out_path, std::ios::binary);
        for (const auto& p : rec) out.write((char*)&p, 12);
        std::cout << "Decoded " << h.num_points << " points.\n";
    }

    return 0;
}