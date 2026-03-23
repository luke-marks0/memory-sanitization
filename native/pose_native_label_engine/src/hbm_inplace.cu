#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace {
namespace cg = cooperative_groups;

constexpr const char *kDomainNamespace = "pose-db";
constexpr std::size_t kDomainNamespaceLen = 7;
constexpr const char *kSourceDomain = "pose-db/label/source/v1";
constexpr std::size_t kSourceDomainLen = 23;
constexpr const char *kInternalDomain = "pose-db/label/internal/v1";
constexpr std::size_t kInternalDomainLen = 25;
constexpr std::size_t kMaxSessionSeedLen = 64;
constexpr std::size_t kMaxDigestLen = 128;
constexpr std::size_t kMaxLabelBytes = 32;
constexpr std::size_t kMaxPayloadBytes = 256;
constexpr std::uint32_t kThreadsPerBlock = 128;
constexpr std::uint32_t kCooperativeThreadsPerBlock = 256;
constexpr bool kEnableExperimentalCooperativeFusedKernels = false;
constexpr std::size_t kBlake3BlockLen = 64;

enum Blake3Flags : std::uint8_t {
  CHUNK_START = 1 << 0,
  CHUNK_END = 1 << 1,
  ROOT = 1 << 3,
};

struct PoseOracleConfig {
  std::uint32_t output_bytes;
  std::uint32_t session_seed_len;
  std::uint32_t digest_len;
  std::uint8_t session_seed[kMaxSessionSeedLen];
  std::uint8_t graph_digest[kMaxDigestLen];
};

struct GraphCounts {
  std::vector<std::size_t> connector;
  std::vector<std::size_t> standalone;
  std::vector<std::size_t> connected;

  explicit GraphCounts(std::size_t max_level)
      : connector(std::max<std::size_t>(max_level, 1), 0),
        standalone(max_level + 1, 0),
        connected(max_level + 1, 0) {
    for (std::size_t dimension = 0; dimension < max_level; ++dimension) {
      connector[dimension] = (dimension + 1) * (static_cast<std::size_t>(1) << dimension);
    }
    standalone[0] = 1;
    connected[0] = 1;
    for (std::size_t level = 1; level <= max_level; ++level) {
      standalone[level] =
          standalone[level - 1] + connector[level - 1] + connected[level - 1];
      connected[level] =
          connected[level - 1] + (2 * connector[level - 1]) + connected[level - 1];
    }
  }

  std::size_t connector_node_count(std::size_t dimension) const {
    return connector[dimension];
  }
};

struct MergedPlan {
  std::size_t width;
  std::vector<std::uint32_t> ingress_indices;
  std::vector<std::uint32_t> center_indices;

  static MergedPlan build(std::size_t dimension) {
    const std::size_t width = static_cast<std::size_t>(1) << dimension;
    const std::size_t center_node_count = (dimension + 1) * width;
    const std::size_t ingress_node_base = center_node_count;
    const std::size_t total_local_nodes = center_node_count * 2;
    std::vector<std::uint8_t> remaining_indegree(total_local_nodes, 0);
    std::vector<std::uint32_t> ingress_indices((dimension + 1) * width, 0);
    std::vector<std::uint32_t> center_indices((dimension + 1) * width, 0);

    for (std::size_t offset = 0; offset < width; ++offset) {
      remaining_indegree[offset] = 1;
    }
    for (std::size_t layer = 1; layer <= dimension; ++layer) {
      const std::size_t layer_base = layer * width;
      const std::size_t ingress_layer_base = ingress_node_base + layer_base;
      for (std::size_t offset = 0; offset < width; ++offset) {
        remaining_indegree[layer_base + offset] = 2;
        remaining_indegree[ingress_layer_base + offset] = 2;
      }
    }

    auto release_local_successor = [&](std::size_t local_successor,
                                       std::priority_queue<std::size_t, std::vector<std::size_t>,
                                                           std::greater<std::size_t>> &ready) {
      const std::uint8_t updated = static_cast<std::uint8_t>(remaining_indegree[local_successor] - 1);
      remaining_indegree[local_successor] = updated;
      if (updated == 0) {
        ready.push(local_successor);
      }
    };

    std::priority_queue<std::size_t, std::vector<std::size_t>, std::greater<std::size_t>> ready;
    for (std::size_t local_node = ingress_node_base; local_node < ingress_node_base + width; ++local_node) {
      ready.push(local_node);
    }

    std::uint32_t next_index = 0;
    while (!ready.empty()) {
      const std::size_t local_node = ready.top();
      ready.pop();
      if (local_node < ingress_node_base) {
        const std::size_t layer = local_node / width;
        const std::size_t offset = local_node % width;
        center_indices[(layer * width) + offset] = next_index++;
        if (layer < dimension) {
          const std::size_t successor_layer_base = (layer + 1) * width;
          const std::size_t bit = static_cast<std::size_t>(1) << (dimension - 1 - layer);
          release_local_successor(successor_layer_base + offset, ready);
          release_local_successor(successor_layer_base + (offset ^ bit), ready);
        }
      } else {
        const std::size_t ingress_local = local_node - ingress_node_base;
        const std::size_t layer = ingress_local / width;
        const std::size_t offset = ingress_local % width;
        ingress_indices[(layer * width) + offset] = next_index++;
        if (layer < dimension) {
          const std::size_t successor_layer_base = ingress_node_base + ((layer + 1) * width);
          const std::size_t bit = static_cast<std::size_t>(1) << (dimension - 1 - layer);
          release_local_successor(successor_layer_base + offset, ready);
          release_local_successor(successor_layer_base + (offset ^ bit), ready);
        } else {
          release_local_successor(offset, ready);
        }
      }
    }

    return MergedPlan{
        width,
        std::move(ingress_indices),
        std::move(center_indices),
    };
  }

  std::size_t accounted_bytes() const {
    return (ingress_indices.size() + center_indices.size()) * sizeof(std::uint32_t);
  }
};

struct DeviceMergedPlan {
  std::size_t index_count;
  std::uint32_t *ingress_indices;
  std::uint32_t *center_indices;

  explicit DeviceMergedPlan(std::size_t index_count_value)
      : index_count(index_count_value), ingress_indices(nullptr), center_indices(nullptr) {}

  ~DeviceMergedPlan() {
    if (ingress_indices != nullptr) {
      cudaFree(ingress_indices);
    }
    if (center_indices != nullptr) {
      cudaFree(center_indices);
    }
  }

  std::size_t accounted_bytes() const {
    return index_count * sizeof(std::uint32_t) * 2;
  }
};

struct CudaStatus {
  int code;
  const char *message;
};

inline CudaStatus ok_status() { return CudaStatus{0, nullptr}; }

inline CudaStatus make_error(const char *message, int code = 1) {
  return CudaStatus{code, message};
}

inline bool cuda_ok(cudaError_t error, const char *action, char *error_buf, std::size_t error_buf_len) {
  if (error == cudaSuccess) {
    return true;
  }
  if (error_buf != nullptr && error_buf_len > 0) {
    std::snprintf(error_buf, error_buf_len, "%s failed: %s", action, cudaGetErrorString(error));
  }
  return false;
}

__device__ __forceinline__ std::uint32_t rotr32(std::uint32_t value, std::uint32_t count) {
  return (value >> count) | (value << (32 - count));
}

__device__ __forceinline__ void blake3_g(std::uint32_t *state, int a, int b, int c, int d,
                                         std::uint32_t x, std::uint32_t y) {
  state[a] = state[a] + state[b] + x;
  state[d] = rotr32(state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  state[d] = rotr32(state[d] ^ state[a], 8);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 7);
}

__device__ __constant__ std::uint32_t BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

__device__ __constant__ std::uint8_t BLAKE3_MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};

__device__ __forceinline__ std::uint32_t load32_le(const std::uint8_t *src) {
  return static_cast<std::uint32_t>(src[0]) |
         (static_cast<std::uint32_t>(src[1]) << 8) |
         (static_cast<std::uint32_t>(src[2]) << 16) |
         (static_cast<std::uint32_t>(src[3]) << 24);
}

__device__ __forceinline__ void store32_le(std::uint8_t *dst, std::uint32_t value) {
  dst[0] = static_cast<std::uint8_t>(value >> 0);
  dst[1] = static_cast<std::uint8_t>(value >> 8);
  dst[2] = static_cast<std::uint8_t>(value >> 16);
  dst[3] = static_cast<std::uint8_t>(value >> 24);
}

__device__ __forceinline__ void blake3_round(std::uint32_t state[16], const std::uint32_t msg[16], int round) {
  const std::uint8_t *schedule = BLAKE3_MSG_SCHEDULE[round];
  blake3_g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
  blake3_g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
  blake3_g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
  blake3_g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);
  blake3_g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
  blake3_g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
  blake3_g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
  blake3_g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

__device__ __forceinline__ void blake3_compress_pre(std::uint32_t state[16], const std::uint32_t cv[8],
                                                     const std::uint8_t block[kBlake3BlockLen],
                                                     std::uint8_t block_len, std::uint64_t counter,
                                                     std::uint8_t flags) {
  std::uint32_t block_words[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    block_words[i] = load32_le(block + (i * 4));
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    state[i] = cv[i];
  }
  state[8] = BLAKE3_IV[0];
  state[9] = BLAKE3_IV[1];
  state[10] = BLAKE3_IV[2];
  state[11] = BLAKE3_IV[3];
  state[12] = static_cast<std::uint32_t>(counter);
  state[13] = static_cast<std::uint32_t>(counter >> 32);
  state[14] = static_cast<std::uint32_t>(block_len);
  state[15] = static_cast<std::uint32_t>(flags);

  #pragma unroll
  for (int round = 0; round < 7; ++round) {
    blake3_round(state, block_words, round);
  }
}

__device__ __forceinline__ void blake3_compress_in_place(std::uint32_t cv[8],
                                                          const std::uint8_t block[kBlake3BlockLen],
                                                          std::uint8_t block_len, std::uint64_t counter,
                                                          std::uint8_t flags) {
  std::uint32_t state[16];
  blake3_compress_pre(state, cv, block, block_len, counter, flags);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    cv[i] = state[i] ^ state[i + 8];
  }
}

__device__ __forceinline__ void blake3_output_root_32(const std::uint32_t cv[8],
                                                      const std::uint8_t block[kBlake3BlockLen],
                                                      std::uint8_t block_len, std::uint64_t counter,
                                                      std::uint8_t flags, std::uint8_t out[32]) {
  std::uint32_t state[16];
  blake3_compress_pre(state, cv, block, block_len, counter, static_cast<std::uint8_t>(flags | ROOT));
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    store32_le(out + (i * 4), state[i] ^ state[i + 8]);
  }
}

__device__ __forceinline__ void blake3_hash_small(const std::uint8_t *input, std::size_t input_len,
                                                  std::uint8_t out[32]) {
  std::uint32_t cv[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    cv[i] = BLAKE3_IV[i];
  }

  std::uint8_t block[kBlake3BlockLen];
  std::size_t offset = 0;
  std::uint8_t flags = CHUNK_START;
  while (input_len - offset > kBlake3BlockLen) {
    #pragma unroll
    for (int i = 0; i < static_cast<int>(kBlake3BlockLen); ++i) {
      block[i] = input[offset + i];
    }
    blake3_compress_in_place(cv, block, static_cast<std::uint8_t>(kBlake3BlockLen), 0, flags);
    flags = 0;
    offset += kBlake3BlockLen;
  }

  const std::size_t remaining = input_len - offset;
  #pragma unroll
  for (int i = 0; i < static_cast<int>(kBlake3BlockLen); ++i) {
    block[i] = 0;
  }
  for (std::size_t i = 0; i < remaining; ++i) {
    block[i] = input[offset + i];
  }
  blake3_output_root_32(cv, block, static_cast<std::uint8_t>(remaining), 0,
                        static_cast<std::uint8_t>(flags | CHUNK_END), out);
}

__device__ __forceinline__ void append_u32_be(std::uint8_t *payload, std::size_t *cursor, std::uint32_t value) {
  payload[(*cursor)++] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
  payload[(*cursor)++] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
  payload[(*cursor)++] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
  payload[(*cursor)++] = static_cast<std::uint8_t>(value & 0xFF);
}

__device__ __forceinline__ void append_u64_be(std::uint8_t *payload, std::size_t *cursor, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    payload[(*cursor)++] = static_cast<std::uint8_t>((value >> shift) & 0xFF);
  }
}

__device__ __forceinline__ void append_bytes(std::uint8_t *payload, std::size_t *cursor,
                                             const std::uint8_t *source, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    payload[(*cursor)++] = source[i];
  }
}

__device__ __forceinline__ void append_domain_prefix(std::uint8_t *payload, std::size_t *cursor,
                                                     const char *domain, std::size_t domain_len,
                                                     std::uint32_t field_count) {
  append_bytes(payload, cursor, reinterpret_cast<const std::uint8_t *>(kDomainNamespace), kDomainNamespaceLen);
  payload[(*cursor)++] = 0;
  payload[(*cursor)++] = 0;
  payload[(*cursor)++] = 0;
  payload[(*cursor)++] = 1;
  append_u32_be(payload, cursor, static_cast<std::uint32_t>(domain_len));
  append_bytes(payload, cursor, reinterpret_cast<const std::uint8_t *>(domain), domain_len);
  append_u32_be(payload, cursor, field_count);
}

__device__ __forceinline__ std::size_t build_source_payload(std::uint8_t *payload, const PoseOracleConfig &config,
                                                            std::uint64_t node_index) {
  std::size_t cursor = 0;
  append_domain_prefix(payload, &cursor, kSourceDomain, kSourceDomainLen, 3);
  append_u32_be(payload, &cursor, config.session_seed_len);
  append_bytes(payload, &cursor, config.session_seed, config.session_seed_len);
  append_u32_be(payload, &cursor, config.digest_len);
  append_bytes(payload, &cursor, config.graph_digest, config.digest_len);
  append_u32_be(payload, &cursor, 8);
  append_u64_be(payload, &cursor, node_index);
  return cursor;
}

__device__ __forceinline__ std::size_t build_internal_payload(std::uint8_t *payload, const PoseOracleConfig &config,
                                                              std::uint64_t node_index,
                                                              const std::uint8_t *pred0,
                                                              const std::uint8_t *pred1,
                                                              std::uint32_t predecessor_count) {
  std::size_t cursor = 0;
  append_domain_prefix(payload, &cursor, kInternalDomain, kInternalDomainLen, 4 + predecessor_count);
  append_u32_be(payload, &cursor, config.session_seed_len);
  append_bytes(payload, &cursor, config.session_seed, config.session_seed_len);
  append_u32_be(payload, &cursor, config.digest_len);
  append_bytes(payload, &cursor, config.graph_digest, config.digest_len);
  append_u32_be(payload, &cursor, 8);
  append_u64_be(payload, &cursor, node_index);
  append_u32_be(payload, &cursor, 4);
  append_u32_be(payload, &cursor, predecessor_count);
  append_u32_be(payload, &cursor, config.output_bytes);
  append_bytes(payload, &cursor, pred0, config.output_bytes);
  if (predecessor_count == 2) {
    append_u32_be(payload, &cursor, config.output_bytes);
    append_bytes(payload, &cursor, pred1, config.output_bytes);
  }
  return cursor;
}

__global__ void kernel_source_labels(std::uint8_t *buffer, std::uint32_t count, std::uint64_t start_index,
                                     PoseOracleConfig config) {
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest[32];
  const std::size_t payload_len = build_source_payload(payload, config, start_index + index);
  blake3_hash_small(payload, payload_len, digest);
  std::uint8_t *out = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    out[i] = digest[i];
  }
}

__global__ void kernel_internal1_copy(const std::uint8_t *inputs, std::uint8_t *outputs, std::uint32_t count,
                                      std::uint64_t start_index, PoseOracleConfig config) {
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest[32];
  const std::uint8_t *pred0 = inputs + (static_cast<std::size_t>(index) * config.output_bytes);
  const std::size_t payload_len =
      build_internal_payload(payload, config, start_index + index, pred0, nullptr, 1);
  blake3_hash_small(payload, payload_len, digest);
  std::uint8_t *out = outputs + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    out[i] = digest[i];
  }
}

__global__ void kernel_internal1_inplace_indexed(std::uint8_t *buffer, std::uint32_t count,
                                                 const std::uint32_t *relative_indices, std::uint64_t start_index,
                                                 PoseOracleConfig config) {
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  std::uint8_t predecessor[kMaxLabelBytes];
  const std::uint8_t *input = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    predecessor[i] = input[i];
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest[32];
  const std::size_t payload_len =
      build_internal_payload(payload, config, start_index + relative_indices[index], predecessor, nullptr, 1);
  blake3_hash_small(payload, payload_len, digest);
  std::uint8_t *out = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    out[i] = digest[i];
  }
}

__global__ void kernel_internal1_inplace_contiguous(std::uint8_t *buffer, std::uint32_t count,
                                                    std::uint64_t start_index, PoseOracleConfig config) {
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) {
    return;
  }
  std::uint8_t predecessor[kMaxLabelBytes];
  const std::uint8_t *input = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    predecessor[i] = input[i];
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest[32];
  const std::size_t payload_len =
      build_internal_payload(payload, config, start_index + index, predecessor, nullptr, 1);
  blake3_hash_small(payload, payload_len, digest);
  std::uint8_t *out = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    out[i] = digest[i];
  }
}

__global__ void kernel_internal2_inplace_contiguous(std::uint8_t *buffer, std::uint32_t width, std::uint32_t bit,
                                                    std::uint64_t layer_start_index, PoseOracleConfig config) {
  const std::uint32_t left = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (left >= width || (left & bit) != 0) {
    return;
  }
  const std::uint32_t right = left ^ bit;
  std::uint8_t pred0[kMaxLabelBytes];
  std::uint8_t pred1[kMaxLabelBytes];
  const std::uint8_t *left_in = buffer + (static_cast<std::size_t>(left) * config.output_bytes);
  const std::uint8_t *right_in = buffer + (static_cast<std::size_t>(right) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    pred0[i] = left_in[i];
    pred1[i] = right_in[i];
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest_left[32];
  std::uint8_t digest_right[32];
  const std::size_t payload_len_left =
      build_internal_payload(payload, config, layer_start_index + left, pred0, pred1, 2);
  blake3_hash_small(payload, payload_len_left, digest_left);
  const std::size_t payload_len_right =
      build_internal_payload(payload, config, layer_start_index + right, pred0, pred1, 2);
  blake3_hash_small(payload, payload_len_right, digest_right);
  std::uint8_t *left_out = buffer + (static_cast<std::size_t>(left) * config.output_bytes);
  std::uint8_t *right_out = buffer + (static_cast<std::size_t>(right) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    left_out[i] = digest_left[i];
    right_out[i] = digest_right[i];
  }
}

__global__ void kernel_internal2_inplace_indexed(std::uint8_t *buffer, std::uint32_t width, std::uint32_t bit,
                                                 const std::uint32_t *relative_indices, std::uint64_t start_index,
                                                 PoseOracleConfig config) {
  const std::uint32_t left = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (left >= width || (left & bit) != 0) {
    return;
  }
  const std::uint32_t right = left ^ bit;
  std::uint8_t pred0[kMaxLabelBytes];
  std::uint8_t pred1[kMaxLabelBytes];
  const std::uint8_t *left_in = buffer + (static_cast<std::size_t>(left) * config.output_bytes);
  const std::uint8_t *right_in = buffer + (static_cast<std::size_t>(right) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    pred0[i] = left_in[i];
    pred1[i] = right_in[i];
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest_left[32];
  std::uint8_t digest_right[32];
  const std::size_t payload_len_left =
      build_internal_payload(payload, config, start_index + relative_indices[left], pred0, pred1, 2);
  blake3_hash_small(payload, payload_len_left, digest_left);
  const std::size_t payload_len_right =
      build_internal_payload(payload, config, start_index + relative_indices[right], pred0, pred1, 2);
  blake3_hash_small(payload, payload_len_right, digest_right);
  std::uint8_t *left_out = buffer + (static_cast<std::size_t>(left) * config.output_bytes);
  std::uint8_t *right_out = buffer + (static_cast<std::size_t>(right) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    left_out[i] = digest_left[i];
    right_out[i] = digest_right[i];
  }
}

__global__ void kernel_internal2_combine_buffers(std::uint8_t *ingress_buffer, const std::uint8_t *primary_inputs,
                                                 std::uint32_t width, const std::uint32_t *relative_indices,
                                                 std::uint64_t start_index, PoseOracleConfig config) {
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= width) {
    return;
  }
  std::uint8_t pred0[kMaxLabelBytes];
  std::uint8_t pred1[kMaxLabelBytes];
  const std::uint8_t *primary = primary_inputs + (static_cast<std::size_t>(index) * config.output_bytes);
  const std::uint8_t *ingress = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    pred0[i] = primary[i];
    pred1[i] = ingress[i];
  }
  std::uint8_t payload[kMaxPayloadBytes];
  std::uint8_t digest[32];
  const std::size_t payload_len =
      build_internal_payload(payload, config, start_index + relative_indices[index], pred0, pred1, 2);
  blake3_hash_small(payload, payload_len, digest);
  std::uint8_t *out = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
  for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
    out[i] = digest[i];
  }
}

__global__ void kernel_connector_inplace_cooperative(std::uint8_t *buffer, std::uint32_t width,
                                                     std::uint32_t dimension, std::uint64_t start_index,
                                                     PoseOracleConfig config) {
  cg::grid_group grid = cg::this_grid();
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < width) {
    std::uint8_t predecessor[kMaxLabelBytes];
    const std::uint8_t *input = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      predecessor[i] = input[i];
    }
    std::uint8_t payload[kMaxPayloadBytes];
    std::uint8_t digest[32];
    const std::size_t payload_len =
        build_internal_payload(payload, config, start_index + index, predecessor, nullptr, 1);
    blake3_hash_small(payload, payload_len, digest);
    std::uint8_t *out = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      out[i] = digest[i];
    }
  }
  grid.sync();
  for (std::uint32_t layer = 0; layer < dimension; ++layer) {
    const std::uint32_t bit = static_cast<std::uint32_t>(1u << (dimension - 1 - layer));
    const std::uint64_t layer_start_index = start_index + width + (static_cast<std::uint64_t>(layer) * width);
    if (index < width && (index & bit) == 0) {
      const std::uint32_t right = index ^ bit;
      std::uint8_t pred0[kMaxLabelBytes];
      std::uint8_t pred1[kMaxLabelBytes];
      const std::uint8_t *left_in = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
      const std::uint8_t *right_in = buffer + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        pred0[i] = left_in[i];
        pred1[i] = right_in[i];
      }
      std::uint8_t payload[kMaxPayloadBytes];
      std::uint8_t digest_left[32];
      std::uint8_t digest_right[32];
      const std::size_t payload_len_left =
          build_internal_payload(payload, config, layer_start_index + index, pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_left, digest_left);
      const std::size_t payload_len_right =
          build_internal_payload(payload, config, layer_start_index + right, pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_right, digest_right);
      std::uint8_t *left_out = buffer + (static_cast<std::size_t>(index) * config.output_bytes);
      std::uint8_t *right_out = buffer + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        left_out[i] = digest_left[i];
        right_out[i] = digest_right[i];
      }
    }
    grid.sync();
  }
}

__global__ void kernel_connector_copy_cooperative(const std::uint8_t *inputs, std::uint8_t *outputs,
                                                  std::uint32_t width, std::uint32_t dimension,
                                                  std::uint64_t start_index, PoseOracleConfig config) {
  cg::grid_group grid = cg::this_grid();
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < width) {
    std::uint8_t payload[kMaxPayloadBytes];
    std::uint8_t digest[32];
    const std::uint8_t *pred0 = inputs + (static_cast<std::size_t>(index) * config.output_bytes);
    const std::size_t payload_len =
        build_internal_payload(payload, config, start_index + index, pred0, nullptr, 1);
    blake3_hash_small(payload, payload_len, digest);
    std::uint8_t *out = outputs + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      out[i] = digest[i];
    }
  }
  grid.sync();
  for (std::uint32_t layer = 0; layer < dimension; ++layer) {
    const std::uint32_t bit = static_cast<std::uint32_t>(1u << (dimension - 1 - layer));
    const std::uint64_t layer_start_index = start_index + width + (static_cast<std::uint64_t>(layer) * width);
    if (index < width && (index & bit) == 0) {
      const std::uint32_t right = index ^ bit;
      std::uint8_t pred0[kMaxLabelBytes];
      std::uint8_t pred1[kMaxLabelBytes];
      const std::uint8_t *left_in = outputs + (static_cast<std::size_t>(index) * config.output_bytes);
      const std::uint8_t *right_in = outputs + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        pred0[i] = left_in[i];
        pred1[i] = right_in[i];
      }
      std::uint8_t payload[kMaxPayloadBytes];
      std::uint8_t digest_left[32];
      std::uint8_t digest_right[32];
      const std::size_t payload_len_left =
          build_internal_payload(payload, config, layer_start_index + index, pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_left, digest_left);
      const std::size_t payload_len_right =
          build_internal_payload(payload, config, layer_start_index + right, pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_right, digest_right);
      std::uint8_t *left_out = outputs + (static_cast<std::size_t>(index) * config.output_bytes);
      std::uint8_t *right_out = outputs + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        left_out[i] = digest_left[i];
        right_out[i] = digest_right[i];
      }
    }
    grid.sync();
  }
}

__global__ void kernel_merged_center_ingress_cooperative(const std::uint8_t *primary_inputs,
                                                         std::uint8_t *ingress_buffer, std::uint32_t width,
                                                         std::uint32_t dimension, std::uint64_t start_index,
                                                         const std::uint32_t *ingress_indices,
                                                         const std::uint32_t *center_indices,
                                                         PoseOracleConfig config) {
  cg::grid_group grid = cg::this_grid();
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < width) {
    std::uint8_t predecessor[kMaxLabelBytes];
    const std::uint8_t *input = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      predecessor[i] = input[i];
    }
    std::uint8_t payload[kMaxPayloadBytes];
    std::uint8_t digest[32];
    const std::size_t payload_len = build_internal_payload(
        payload, config, start_index + ingress_indices[index], predecessor, nullptr, 1);
    blake3_hash_small(payload, payload_len, digest);
    std::uint8_t *out = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      out[i] = digest[i];
    }
  }
  grid.sync();
  for (std::uint32_t layer = 1; layer <= dimension; ++layer) {
    const std::uint32_t bit = static_cast<std::uint32_t>(1u << (dimension - layer));
    const std::uint32_t *layer_indices = ingress_indices + (static_cast<std::size_t>(layer) * width);
    if (index < width && (index & bit) == 0) {
      const std::uint32_t right = index ^ bit;
      std::uint8_t pred0[kMaxLabelBytes];
      std::uint8_t pred1[kMaxLabelBytes];
      const std::uint8_t *left_in = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
      const std::uint8_t *right_in = ingress_buffer + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        pred0[i] = left_in[i];
        pred1[i] = right_in[i];
      }
      std::uint8_t payload[kMaxPayloadBytes];
      std::uint8_t digest_left[32];
      std::uint8_t digest_right[32];
      const std::size_t payload_len_left =
          build_internal_payload(payload, config, start_index + layer_indices[index], pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_left, digest_left);
      const std::size_t payload_len_right =
          build_internal_payload(payload, config, start_index + layer_indices[right], pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_right, digest_right);
      std::uint8_t *left_out = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
      std::uint8_t *right_out = ingress_buffer + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        left_out[i] = digest_left[i];
        right_out[i] = digest_right[i];
      }
    }
    grid.sync();
  }
  if (index < width) {
    std::uint8_t pred0[kMaxLabelBytes];
    std::uint8_t pred1[kMaxLabelBytes];
    const std::uint8_t *primary = primary_inputs + (static_cast<std::size_t>(index) * config.output_bytes);
    const std::uint8_t *ingress = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      pred0[i] = primary[i];
      pred1[i] = ingress[i];
    }
    std::uint8_t payload[kMaxPayloadBytes];
    std::uint8_t digest[32];
    const std::size_t payload_len =
        build_internal_payload(payload, config, start_index + center_indices[index], pred0, pred1, 2);
    blake3_hash_small(payload, payload_len, digest);
    std::uint8_t *out = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
    for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
      out[i] = digest[i];
    }
  }
  grid.sync();
  for (std::uint32_t layer = 1; layer <= dimension; ++layer) {
    const std::uint32_t bit = static_cast<std::uint32_t>(1u << (dimension - layer));
    const std::uint32_t *layer_indices = center_indices + (static_cast<std::size_t>(layer) * width);
    if (index < width && (index & bit) == 0) {
      const std::uint32_t right = index ^ bit;
      std::uint8_t pred0[kMaxLabelBytes];
      std::uint8_t pred1[kMaxLabelBytes];
      const std::uint8_t *left_in = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
      const std::uint8_t *right_in = ingress_buffer + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        pred0[i] = left_in[i];
        pred1[i] = right_in[i];
      }
      std::uint8_t payload[kMaxPayloadBytes];
      std::uint8_t digest_left[32];
      std::uint8_t digest_right[32];
      const std::size_t payload_len_left =
          build_internal_payload(payload, config, start_index + layer_indices[index], pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_left, digest_left);
      const std::size_t payload_len_right =
          build_internal_payload(payload, config, start_index + layer_indices[right], pred0, pred1, 2);
      blake3_hash_small(payload, payload_len_right, digest_right);
      std::uint8_t *left_out = ingress_buffer + (static_cast<std::size_t>(index) * config.output_bytes);
      std::uint8_t *right_out = ingress_buffer + (static_cast<std::size_t>(right) * config.output_bytes);
      for (std::uint32_t i = 0; i < config.output_bytes; ++i) {
        left_out[i] = digest_left[i];
        right_out[i] = digest_right[i];
      }
    }
    grid.sync();
  }
}

class CudaInPlaceLabeler {
 public:
  CudaInPlaceLabeler(std::size_t label_count_m, std::size_t graph_parameter_n, const PoseOracleConfig &config)
      : label_count_m_(label_count_m),
        graph_parameter_n_(graph_parameter_n),
        config_(config),
        counts_(graph_parameter_n + 1),
        plans_(graph_parameter_n + 1),
        device_plans_(graph_parameter_n + 1),
        merged_plan_bytes_(0),
        device_plan_bytes_(0),
        peak_scratch_bytes_(sizeof(PoseOracleConfig)),
        cooperative_launch_supported_(false),
        multiprocessor_count_(0) {
    int supported = 0;
    if (cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, 0) == cudaSuccess) {
      cooperative_launch_supported_ = supported != 0;
    }
    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
      multiprocessor_count_ = props.multiProcessorCount;
    }
  }

  ~CudaInPlaceLabeler() {
  }

  std::uint64_t scratch_peak_bytes() const { return peak_scratch_bytes_; }

  cudaError_t fill(std::uint8_t *target, std::size_t target_len) {
    const std::size_t left_width = static_cast<std::size_t>(1) << graph_parameter_n_;
    const std::size_t retained_from_right = label_count_m_ - left_width;
    const std::size_t expected_len = label_count_m_ * config_.output_bytes;
    if (target_len != expected_len) {
      return cudaErrorInvalidValue;
    }
    std::uint8_t *left_workspace = target;
    std::uint8_t *right_output = target + (left_width * config_.output_bytes);
    if (retained_from_right > 0) {
      const std::size_t right_copy_start = counts_.standalone[graph_parameter_n_ + 1];
      auto error = standalone_right_prefix(graph_parameter_n_ + 1, retained_from_right, left_workspace, right_copy_start);
      if (error != cudaSuccess) {
        return error;
      }
      const std::size_t right_bytes = retained_from_right * config_.output_bytes;
      error = cudaMemcpy(right_output, left_workspace, right_bytes, cudaMemcpyDeviceToDevice);
      if (error != cudaSuccess) {
        return error;
      }
      error = cudaDeviceSynchronize();
      if (error != cudaSuccess) {
        return error;
      }
    }
    return standalone_right_prefix(graph_parameter_n_ + 1, left_width, left_workspace, 0);
  }

 private:
  std::size_t label_count_m_;
  std::size_t graph_parameter_n_;
  PoseOracleConfig config_;
  GraphCounts counts_;
  std::vector<std::unique_ptr<MergedPlan>> plans_;
  std::vector<std::unique_ptr<DeviceMergedPlan>> device_plans_;
  std::uint64_t merged_plan_bytes_;
  std::uint64_t device_plan_bytes_;
  std::uint64_t peak_scratch_bytes_;
  bool cooperative_launch_supported_;
  int multiprocessor_count_;

  void record_scratch() {
    peak_scratch_bytes_ =
        std::max<std::uint64_t>(peak_scratch_bytes_, merged_plan_bytes_ + device_plan_bytes_ + sizeof(PoseOracleConfig));
  }

  const MergedPlan &plan(std::size_t dimension) {
    if (!plans_[dimension]) {
      plans_[dimension] = std::make_unique<MergedPlan>(MergedPlan::build(dimension));
      merged_plan_bytes_ += plans_[dimension]->accounted_bytes();
      record_scratch();
    }
    return *plans_[dimension];
  }

  const DeviceMergedPlan &device_plan(std::size_t dimension) {
    if (!device_plans_[dimension]) {
      const auto &host_plan = plan(dimension);
      auto plan_copy = std::make_unique<DeviceMergedPlan>(host_plan.ingress_indices.size());
      cudaError_t error = cudaMalloc(
          reinterpret_cast<void **>(&plan_copy->ingress_indices),
          host_plan.ingress_indices.size() * sizeof(std::uint32_t));
      if (error != cudaSuccess) {
        throw error;
      }
      error = cudaMalloc(
          reinterpret_cast<void **>(&plan_copy->center_indices),
          host_plan.center_indices.size() * sizeof(std::uint32_t));
      if (error != cudaSuccess) {
        throw error;
      }
      error = cudaMemcpy(
          plan_copy->ingress_indices,
          host_plan.ingress_indices.data(),
          host_plan.ingress_indices.size() * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
        throw error;
      }
      error = cudaMemcpy(
          plan_copy->center_indices,
          host_plan.center_indices.data(),
          host_plan.center_indices.size() * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice);
      if (error != cudaSuccess) {
        throw error;
      }
      device_plan_bytes_ += plan_copy->accounted_bytes();
      record_scratch();
      device_plans_[dimension] = std::move(plan_copy);
    }
    return *device_plans_[dimension];
  }

  template <typename KernelT>
  bool cooperative_launch_fits(KernelT kernel, std::size_t blocks, std::uint32_t threads) const {
    if (!kEnableExperimentalCooperativeFusedKernels || !cooperative_launch_supported_ || multiprocessor_count_ <= 0) {
      return false;
    }
    int blocks_per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, static_cast<int>(threads), 0) != cudaSuccess) {
      return false;
    }
    return static_cast<std::size_t>(blocks_per_sm) * static_cast<std::size_t>(multiprocessor_count_) >= blocks;
  }

  cudaError_t launch_source_labels(std::uint8_t *buffer, std::size_t count, std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_source_labels<<<blocks, kThreadsPerBlock>>>(buffer, static_cast<std::uint32_t>(count), start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t launch_internal1_copy(const std::uint8_t *inputs, std::uint8_t *outputs, std::size_t count,
                                    std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_internal1_copy<<<blocks, kThreadsPerBlock>>>(
        inputs, outputs, static_cast<std::uint32_t>(count), start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t launch_internal1_inplace_contiguous(std::uint8_t *buffer, std::size_t count,
                                                  std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_internal1_inplace_contiguous<<<blocks, kThreadsPerBlock>>>(
        buffer, static_cast<std::uint32_t>(count), start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t launch_internal1_inplace_indexed(std::uint8_t *buffer, std::size_t count,
                                               const std::uint32_t *device_relative_indices,
                                               std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_internal1_inplace_indexed<<<blocks, kThreadsPerBlock>>>(
        buffer, static_cast<std::uint32_t>(count), device_relative_indices, start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t launch_internal2_contiguous(std::uint8_t *buffer, std::size_t width, std::size_t bit,
                                          std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((width + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_internal2_inplace_contiguous<<<blocks, kThreadsPerBlock>>>(
        buffer, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(bit), start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t launch_connector_inplace_cooperative(std::uint8_t *buffer, std::size_t width,
                                                   std::size_t dimension, std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((width + kCooperativeThreadsPerBlock - 1) / kCooperativeThreadsPerBlock);
    if (!cooperative_launch_fits(kernel_connector_inplace_cooperative, blocks, kCooperativeThreadsPerBlock)) {
      return cudaErrorCooperativeLaunchTooLarge;
    }
    std::uint32_t width32 = static_cast<std::uint32_t>(width);
    std::uint32_t dimension32 = static_cast<std::uint32_t>(dimension);
    void *args[] = {&buffer, &width32, &dimension32, &start_index, &config_};
    auto error = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(kernel_connector_inplace_cooperative),
        dim3(blocks),
        dim3(kCooperativeThreadsPerBlock),
        args);
    if (error != cudaSuccess) {
      return error;
    }
    return cudaGetLastError();
  }

  cudaError_t launch_connector_copy_cooperative(const std::uint8_t *inputs, std::uint8_t *outputs, std::size_t width,
                                                std::size_t dimension, std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((width + kCooperativeThreadsPerBlock - 1) / kCooperativeThreadsPerBlock);
    if (!cooperative_launch_fits(kernel_connector_copy_cooperative, blocks, kCooperativeThreadsPerBlock)) {
      return cudaErrorCooperativeLaunchTooLarge;
    }
    std::uint32_t width32 = static_cast<std::uint32_t>(width);
    std::uint32_t dimension32 = static_cast<std::uint32_t>(dimension);
    std::uint8_t *mutable_inputs = const_cast<std::uint8_t *>(inputs);
    void *args[] = {&mutable_inputs, &outputs, &width32, &dimension32, &start_index, &config_};
    auto error = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(kernel_connector_copy_cooperative),
        dim3(blocks),
        dim3(kCooperativeThreadsPerBlock),
        args);
    if (error != cudaSuccess) {
      return error;
    }
    return cudaGetLastError();
  }

  cudaError_t launch_merged_center_ingress_cooperative(const std::uint8_t *primary_inputs,
                                                       std::uint8_t *ingress_workspace, std::size_t width,
                                                       std::size_t dimension, std::uint64_t start_index,
                                                       const std::uint32_t *device_ingress_indices,
                                                       const std::uint32_t *device_center_indices) {
    const auto blocks = static_cast<unsigned>((width + kCooperativeThreadsPerBlock - 1) / kCooperativeThreadsPerBlock);
    if (!cooperative_launch_fits(kernel_merged_center_ingress_cooperative, blocks, kCooperativeThreadsPerBlock)) {
      return cudaErrorCooperativeLaunchTooLarge;
    }
    std::uint32_t width32 = static_cast<std::uint32_t>(width);
    std::uint32_t dimension32 = static_cast<std::uint32_t>(dimension);
    std::uint8_t *mutable_primary_inputs = const_cast<std::uint8_t *>(primary_inputs);
    std::uint32_t *mutable_ingress_indices = const_cast<std::uint32_t *>(device_ingress_indices);
    std::uint32_t *mutable_center_indices = const_cast<std::uint32_t *>(device_center_indices);
    void *args[] = {
        &mutable_primary_inputs,
        &ingress_workspace,
        &width32,
        &dimension32,
        &start_index,
        &mutable_ingress_indices,
        &mutable_center_indices,
        &config_,
    };
    auto error = cudaLaunchCooperativeKernel(
        reinterpret_cast<void *>(kernel_merged_center_ingress_cooperative),
        dim3(blocks),
        dim3(kCooperativeThreadsPerBlock),
        args);
    if (error != cudaSuccess) {
      return error;
    }
    return cudaGetLastError();
  }

  cudaError_t launch_internal2_indexed(std::uint8_t *buffer, std::size_t width, std::size_t bit,
                                       const std::uint32_t *device_relative_indices,
                                       std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((width + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_internal2_inplace_indexed<<<blocks, kThreadsPerBlock>>>(
        buffer, static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(bit), device_relative_indices, start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t launch_combine_buffers(std::uint8_t *ingress_buffer, const std::uint8_t *primary_inputs,
                                     std::size_t width, const std::uint32_t *device_relative_indices,
                                     std::uint64_t start_index) {
    const auto blocks = static_cast<unsigned>((width + kThreadsPerBlock - 1) / kThreadsPerBlock);
    kernel_internal2_combine_buffers<<<blocks, kThreadsPerBlock>>>(
        ingress_buffer, primary_inputs, static_cast<std::uint32_t>(width), device_relative_indices, start_index, config_);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
    return cudaSuccess;
  }

  cudaError_t butterfly_layers_in_place(std::size_t dimension, std::uint8_t *buffer, std::uint64_t start_index) {
    const std::size_t width = static_cast<std::size_t>(1) << dimension;
    for (std::size_t layer = 0; layer < dimension; ++layer) {
      const std::size_t bit = static_cast<std::size_t>(1) << (dimension - 1 - layer);
      const std::uint64_t layer_start = start_index + (layer * width);
      auto error = launch_internal2_contiguous(buffer, width, bit, layer_start);
      if (error != cudaSuccess) {
        return error;
      }
    }
    return cudaSuccess;
  }

  cudaError_t connector_from_inputs_into_buffer(std::size_t dimension, const std::uint8_t *inputs,
                                                std::uint8_t *outputs, std::uint64_t start_index) {
    const std::size_t width = static_cast<std::size_t>(1) << dimension;
    auto error = launch_connector_copy_cooperative(inputs, outputs, width, dimension, start_index);
    if (error == cudaSuccess) {
      return cudaSuccess;
    }
    error = launch_internal1_copy(inputs, outputs, width, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    return butterfly_layers_in_place(dimension, outputs, start_index + width);
  }

  cudaError_t connector_in_place(std::size_t dimension, std::uint8_t *buffer, std::uint64_t start_index) {
    const std::size_t width = static_cast<std::size_t>(1) << dimension;
    auto error = launch_connector_inplace_cooperative(buffer, width, dimension, start_index);
    if (error == cudaSuccess) {
      return cudaSuccess;
    }
    error = launch_internal1_inplace_contiguous(buffer, width, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    return butterfly_layers_in_place(dimension, buffer, start_index + width);
  }

  cudaError_t merged_center_ingress_in_place(std::size_t dimension, const std::uint8_t *primary_inputs,
                                             std::uint8_t *ingress_workspace, std::uint64_t start_index) {
    const auto &merged_plan = plan(dimension);
    const auto &device_plan_copy = device_plan(dimension);
    const std::size_t width = merged_plan.width;
    auto error = launch_merged_center_ingress_cooperative(
        primary_inputs,
        ingress_workspace,
        width,
        dimension,
        start_index,
        device_plan_copy.ingress_indices,
        device_plan_copy.center_indices);
    if (error == cudaSuccess) {
      return cudaSuccess;
    }
    error = launch_internal1_inplace_indexed(
        ingress_workspace, width, device_plan_copy.ingress_indices, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    for (std::size_t layer = 1; layer <= dimension; ++layer) {
      const std::size_t bit = static_cast<std::size_t>(1) << (dimension - layer);
      error = launch_internal2_indexed(
          ingress_workspace,
          width,
          bit,
          device_plan_copy.ingress_indices + (layer * width),
          start_index);
      if (error != cudaSuccess) {
        return error;
      }
    }
    error = launch_combine_buffers(
        ingress_workspace,
        primary_inputs,
        width,
        device_plan_copy.center_indices,
        start_index);
    if (error != cudaSuccess) {
      return error;
    }
    for (std::size_t layer = 1; layer <= dimension; ++layer) {
      const std::size_t bit = static_cast<std::size_t>(1) << (dimension - layer);
      error = launch_internal2_indexed(
          ingress_workspace,
          width,
          bit,
          device_plan_copy.center_indices + (layer * width),
          start_index);
      if (error != cudaSuccess) {
        return error;
      }
    }
    return cudaSuccess;
  }

  cudaError_t connected_full(std::size_t level, std::uint8_t *buffer, std::uint64_t start_index) {
    if (level == 0) {
      return launch_internal1_inplace_contiguous(buffer, 1, start_index);
    }
    const std::size_t half_slots = static_cast<std::size_t>(1) << (level - 1);
    std::uint8_t *left = buffer;
    std::uint8_t *right = buffer + (half_slots * config_.output_bytes);
    auto error = connected_full(level - 1, left, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t merged_start = start_index + counts_.connected[level - 1];
    error = merged_center_ingress_in_place(level - 1, left, right, merged_start);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t right_start =
        merged_start + (2 * counts_.connector_node_count(level - 1));
    return connected_full(level - 1, right, right_start);
  }

  cudaError_t connected_prefix(std::size_t level, std::size_t retain, std::uint8_t *buffer,
                               std::uint64_t start_index) {
    if (retain == 0) {
      return cudaSuccess;
    }
    if (level == 0) {
      return launch_internal1_inplace_contiguous(buffer, 1, start_index);
    }
    const std::size_t half_slots = static_cast<std::size_t>(1) << (level - 1);
    std::uint8_t *left = buffer;
    std::uint8_t *right = buffer + (half_slots * config_.output_bytes);
    if (retain <= half_slots) {
      return connected_prefix(level - 1, retain, left, start_index);
    }
    auto error = connected_full(level - 1, left, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t merged_start = start_index + counts_.connected[level - 1];
    error = merged_center_ingress_in_place(level - 1, left, right, merged_start);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t right_start =
        merged_start + (2 * counts_.connector_node_count(level - 1));
    return connected_prefix(level - 1, retain - half_slots, right, right_start);
  }

  cudaError_t standalone_base(std::size_t level, std::uint8_t *buffer, std::uint64_t start_index) {
    if (level == 0) {
      return launch_source_labels(buffer, 1, start_index);
    }
    const std::size_t half_slots = static_cast<std::size_t>(1) << (level - 1);
    std::uint8_t *left = buffer;
    std::uint8_t *right = buffer + (half_slots * config_.output_bytes);
    auto error = standalone_base(level - 1, left, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t connector_start = start_index + counts_.standalone[level - 1];
    error = connector_from_inputs_into_buffer(level - 1, left, right, connector_start);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t right_start =
        connector_start + counts_.connector_node_count(level - 1);
    return connected_full(level - 1, right, right_start);
  }

  cudaError_t standalone_right_prefix(std::size_t level, std::size_t retain, std::uint8_t *workspace,
                                      std::uint64_t start_index) {
    if (retain == 0) {
      return cudaSuccess;
    }
    const std::size_t half_slots = static_cast<std::size_t>(1) << (level - 1);
    auto error = standalone_base(level - 1, workspace, start_index);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t connector_start = start_index + counts_.standalone[level - 1];
    error = connector_in_place(level - 1, workspace, connector_start);
    if (error != cudaSuccess) {
      return error;
    }
    const std::uint64_t right_start =
        connector_start + counts_.connector_node_count(level - 1);
    return connected_prefix(level - 1, retain, workspace, right_start);
  }

 public:
  cudaError_t finalize() const {
    return cudaDeviceSynchronize();
  }
};

}  // namespace

extern "C" int pose_cuda_fill_challenge_labels_in_place_blake3(
    std::size_t label_count_m,
    std::size_t graph_parameter_n,
    std::size_t output_bytes,
    int device,
    const std::uint8_t *session_seed,
    std::size_t session_seed_len,
    const std::uint8_t *graph_descriptor_digest,
    std::size_t graph_descriptor_digest_len,
    void *target_pointer,
    std::size_t target_len,
    std::uint64_t *scratch_peak_bytes_out,
    char *error_buf,
    std::size_t error_buf_len) {
  if (error_buf != nullptr && error_buf_len > 0) {
    error_buf[0] = '\0';
  }
  if (label_count_m == 0 || output_bytes == 0 || output_bytes > kMaxLabelBytes ||
      session_seed_len > kMaxSessionSeedLen || graph_descriptor_digest_len > kMaxDigestLen ||
      target_pointer == nullptr || scratch_peak_bytes_out == nullptr) {
    if (error_buf != nullptr && error_buf_len > 0) {
      std::snprintf(error_buf, error_buf_len, "invalid CUDA HBM in-place arguments");
    }
    return 1;
  }

  if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice", error_buf, error_buf_len)) {
    return 2;
  }

  PoseOracleConfig config{};
  config.output_bytes = static_cast<std::uint32_t>(output_bytes);
  config.session_seed_len = static_cast<std::uint32_t>(session_seed_len);
  config.digest_len = static_cast<std::uint32_t>(graph_descriptor_digest_len);
  std::memcpy(config.session_seed, session_seed, session_seed_len);
  std::memcpy(config.graph_digest, graph_descriptor_digest, graph_descriptor_digest_len);

  CudaInPlaceLabeler labeler(label_count_m, graph_parameter_n, config);
  cudaError_t error = cudaSuccess;
  try {
    error = labeler.fill(static_cast<std::uint8_t *>(target_pointer), target_len);
  } catch (cudaError_t caught) {
    error = caught;
  }
  if (error != cudaSuccess) {
    if (error_buf != nullptr && error_buf_len > 0) {
      std::snprintf(error_buf, error_buf_len, "HBM in-place labeling failed: %s", cudaGetErrorString(error));
    }
    return 3;
  }
  error = labeler.finalize();
  if (error != cudaSuccess) {
    if (error_buf != nullptr && error_buf_len > 0) {
      std::snprintf(error_buf, error_buf_len, "HBM in-place labeling finalize failed: %s", cudaGetErrorString(error));
    }
    return 4;
  }
  *scratch_peak_bytes_out = labeler.scratch_peak_bytes();
  return 0;
}
