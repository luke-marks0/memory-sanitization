#include "blake3.h"
#include "blake3_impl.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define POSE_BLAKE3_INTERNAL2_BATCH_MAX 32
#define POSE_BLAKE3_INTERNAL2_PAIR_BATCH_MAX (POSE_BLAKE3_INTERNAL2_BATCH_MAX / 2)

typedef blake3_chunk_state pose_blake3_chunk_state;

void pose_blake3_internal2_state_init(const uint8_t *prefix, size_t prefix_len,
                                      pose_blake3_chunk_state *out_state) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, prefix, prefix_len);
  assert(hasher.cv_stack_len == 0);
  *out_state = hasher.chunk;
}

void pose_blake3_internal2_hash_many_pairs_32(
    const pose_blake3_chunk_state *state, const uint8_t *static_suffix0,
    size_t static_suffix0_len, const uint8_t *static_suffix1,
    size_t static_suffix1_len, const uint8_t *left_node_indices,
    const uint8_t *right_node_indices, const uint8_t *block_slice,
    const uint32_t *pair_offsets, size_t num_pairs, uint8_t *out) {
  assert(state != NULL);
  assert(num_pairs <= POSE_BLAKE3_INTERNAL2_PAIR_BATCH_MAX);

  const size_t output_bytes = BLAKE3_OUT_LEN;
  const size_t num_inputs = num_pairs * 2;
  const size_t first_pred0_bytes =
      BLAKE3_BLOCK_LEN - (size_t)state->buf_len - 8 - static_suffix0_len;
  const size_t tail_pred0_bytes = output_bytes - first_pred0_bytes;
  const size_t final_block_len = tail_pred0_bytes + static_suffix1_len + output_bytes;
  const uint8_t start_flag = state->blocks_compressed == 0 ? CHUNK_START : 0;

  assert(first_pred0_bytes < output_bytes);
  assert(final_block_len < BLAKE3_BLOCK_LEN);

  uint8_t first_blocks[POSE_BLAKE3_INTERNAL2_BATCH_MAX][BLAKE3_BLOCK_LEN];
  uint8_t final_blocks[POSE_BLAKE3_INTERNAL2_PAIR_BATCH_MAX][BLAKE3_BLOCK_LEN] = {0};
  const uint8_t *input_ptrs[POSE_BLAKE3_INTERNAL2_BATCH_MAX] = {0};
  uint8_t cv_bytes[POSE_BLAKE3_INTERNAL2_BATCH_MAX * BLAKE3_OUT_LEN];

  for (size_t pair = 0; pair < num_pairs; ++pair) {
    const uint8_t *predecessor0 = block_slice + pair_offsets[pair * 2];
    const uint8_t *predecessor1 = block_slice + pair_offsets[(pair * 2) + 1];
    uint8_t *final_block = final_blocks[pair];
    memcpy(final_block, predecessor0 + first_pred0_bytes, tail_pred0_bytes);
    memcpy(final_block + tail_pred0_bytes, static_suffix1, static_suffix1_len);
    memcpy(final_block + tail_pred0_bytes + static_suffix1_len, predecessor1,
           output_bytes);

    for (size_t lane = 0; lane < 2; ++lane) {
      const size_t input = (pair * 2) + lane;
      const uint8_t *node_index =
          lane == 0 ? left_node_indices + (pair * 8) : right_node_indices + (pair * 8);
      uint8_t *first_block = first_blocks[input];
      memcpy(first_block, state->buf, state->buf_len);
      memcpy(first_block + state->buf_len, node_index, 8);
      memcpy(first_block + state->buf_len + 8, static_suffix0, static_suffix0_len);
      memcpy(first_block + state->buf_len + 8 + static_suffix0_len, predecessor0,
             first_pred0_bytes);
      input_ptrs[input] = first_block;
    }
  }

  blake3_hash_many(input_ptrs, num_inputs, 1, state->cv, state->chunk_counter,
                   false, state->flags, start_flag, 0, cv_bytes);

  for (size_t pair = 0; pair < num_pairs; ++pair) {
    for (size_t lane = 0; lane < 2; ++lane) {
      const size_t input = (pair * 2) + lane;
      uint32_t cv_words[8];
      load_key_words(cv_bytes + (input * BLAKE3_OUT_LEN), cv_words);
      blake3_compress_in_place(cv_words, final_blocks[pair], (uint8_t)final_block_len,
                               state->chunk_counter, state->flags | CHUNK_END | ROOT);
      store_cv_words(out + (input * output_bytes), cv_words);
    }
  }
}
