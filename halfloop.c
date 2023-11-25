// compile and run: g++ -Ofast -fopenmp halfloop.c -std=c++20 -Wall -Wextra -Wpedantic -march=native; ./a.out
// dependency: CPU with AVX-256 support

#include <iostream>
#include <omp.h>
#include <cstdint>
#include <chrono>
#include <unistd.h>
#include <vector>
#include <immintrin.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef __uint128_t u128;

using namespace std::chrono;
using namespace std::chrono_literals;

// compile time flags: 1 = True
// check correct value of (rk10, rk9) in the first iteration
#define CHECK_CORRECT_FIRST 0
// count various stuff (to experimentally verify our analysis)
// set to zero when benchmarking the performance!
#define COUNTERS 1
// use omp to parallelize attack
#define PARALLEL 1

// compile time const
// only check subset of {(rk10, rk9)} where
// rk10 < MAX_RK10 and rk9 < MAX_RK9
// -> complexity is MAX_RK10*MAX_RK9 <= 2**48
// for real attack use MAX_RK10 = MAX_RK9 = 0x1000000
const u64 MAX_RK10 = 0x010000;
const u64 MAX_RK9  = 0x010000;
const u64 REP  = 5;

/////////////////////////////////////////
// START OF HALFLOOP-24 IMPLEMENTATION //
/////////////////////////////////////////
static const u8 SBOX[256] = {
  0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
  0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
  0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
  0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
  0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
  0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
  0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
  0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
  0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
  0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
  0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
  0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
  0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
  0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
  0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
  0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16};

static const u8 inv_SBOX[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};
u32 sub_bytes(u32 state){
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  state = (SBOX[a0] << 16) ^ (SBOX[a1] << 8) ^ SBOX[a2];
  return state;
}

u32 inv_sub_bytes(u32 state){
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  state = (inv_SBOX[a0] << 16) ^ (inv_SBOX[a1] << 8) ^ inv_SBOX[a2];
  return state;
}

u32 rotate_rows(u32 state){
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  a1 = (a1 << 6) | (a1 >> 2);
  a2 = (a2 << 4) | (a2 >> 4);
  state = (a0 << 16) ^ (a1 << 8) ^ a2;
  return state;
}

u32 inv_rotate_rows(u32 state){
  u8 a0 = state >> 16;
  u8 a1 = (state >> 8) & 0xFF;
  u8 a2 = state & 0xFF;
  a1 = (a1 >> 6) | (a1 << 2);
  a2 = (a2 >> 4) | (a2 << 4);
  state = (a0 << 16) ^ (a1 << 8) ^ a2;
  return state;
}

u32 mix_columns(u32 state){
  u32 s = 0;
  s |= (((state >> 0) ^ (state >> 5) ^ (state >> 15) ^ (state >> 16)) & 0x1) << 0;
  s |= (((state >> 1) ^ (state >> 5) ^ (state >> 6) ^ (state >> 8) ^ (state >> 15) ^ (state >> 17)) & 0x1) << 1;
  s |= (((state >> 2) ^ (state >> 6) ^ (state >> 7) ^ (state >> 9) ^ (state >> 18)) & 0x1) << 2;
  s |= (((state >> 0) ^ (state >> 3) ^ (state >> 5) ^ (state >> 7) ^ (state >> 10) ^ (state >> 15) ^ (state >> 19)) & 0x1) << 3;
  s |= (((state >> 1) ^ (state >> 4) ^ (state >> 5) ^ (state >> 6) ^ (state >> 11) ^ (state >> 15) ^ (state >> 20)) & 0x1) << 4;
  s |= (((state >> 2) ^ (state >> 5) ^ (state >> 6) ^ (state >> 7) ^ (state >> 12) ^ (state >> 21)) & 0x1) << 5;
  s |= (((state >> 3) ^ (state >> 6) ^ (state >> 7) ^ (state >> 13) ^ (state >> 22)) & 0x1) << 6;
  s |= (((state >> 4) ^ (state >> 7) ^ (state >> 14) ^ (state >> 23)) & 0x1) << 7;
  s |= (((state >> 0) ^ (state >> 8) ^ (state >> 13) ^ (state >> 23)) & 0x1) << 8;
  s |= (((state >> 1) ^ (state >> 9) ^ (state >> 13) ^ (state >> 14) ^ (state >> 16) ^ (state >> 23)) & 0x1) << 9;
  s |= (((state >> 2) ^ (state >> 10) ^ (state >> 14) ^ (state >> 15) ^ (state >> 17)) & 0x1) << 10;
  s |= (((state >> 3) ^ (state >> 8) ^ (state >> 11) ^ (state >> 13) ^ (state >> 15) ^ (state >> 18) ^ (state >> 23)) & 0x1) << 11;
  s |= (((state >> 4) ^ (state >> 9) ^ (state >> 12) ^ (state >> 13) ^ (state >> 14) ^ (state >> 19) ^ (state >> 23)) & 0x1) << 12;
  s |= (((state >> 5) ^ (state >> 10) ^ (state >> 13) ^ (state >> 14) ^ (state >> 15) ^ (state >> 20)) & 0x1) << 13;
  s |= (((state >> 6) ^ (state >> 11) ^ (state >> 14) ^ (state >> 15) ^ (state >> 21)) & 0x1) << 14;
  s |= (((state >> 7) ^ (state >> 12) ^ (state >> 15) ^ (state >> 22)) & 0x1) << 15;
  s |= (((state >> 7) ^ (state >> 8) ^ (state >> 16) ^ (state >> 21)) & 0x1) << 16;
  s |= (((state >> 0) ^ (state >> 7) ^ (state >> 9) ^ (state >> 17) ^ (state >> 21) ^ (state >> 22)) & 0x1) << 17;
  s |= (((state >> 1) ^ (state >> 10) ^ (state >> 18) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 18;
  s |= (((state >> 2) ^ (state >> 7) ^ (state >> 11) ^ (state >> 16) ^ (state >> 19) ^ (state >> 21) ^ (state >> 23)) & 0x1) << 19;
  s |= (((state >> 3) ^ (state >> 7) ^ (state >> 12) ^ (state >> 17) ^ (state >> 20) ^ (state >> 21) ^ (state >> 22)) & 0x1) << 20;
  s |= (((state >> 4) ^ (state >> 13) ^ (state >> 18) ^ (state >> 21) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 21;
  s |= (((state >> 5) ^ (state >> 14) ^ (state >> 19) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 22;
  s |= (((state >> 6) ^ (state >> 15) ^ (state >> 20) ^ (state >> 23)) & 0x1) << 23;
  return s;
}

u32 inv_mix_columns(u32 state){
  u32 s = 0;
  s |= (((state >> 6) ^ (state >> 7) ^ (state >> 8) ^ (state >> 11) ^ (state >> 14) ^ (state >> 21)) & 0x1) << 0;
  s |= (((state >> 0) ^ (state >> 6) ^ (state >> 8) ^ (state >> 9) ^ (state >> 11) ^ (state >> 12) ^ (state >> 14) ^ (state >> 15) ^ (state >> 21) ^ (state >> 22)) & 0x1) << 1;
  s |= (((state >> 0) ^ (state >> 1) ^ (state >> 7) ^ (state >> 8) ^ (state >> 9) ^ (state >> 10) ^ (state >> 12) ^ (state >> 13) ^ (state >> 15) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 2;
  s |= (((state >> 1) ^ (state >> 2) ^ (state >> 6) ^ (state >> 7) ^ (state >> 9) ^ (state >> 10) ^ (state >> 13) ^ (state >> 16) ^ (state >> 21) ^ (state >> 23)) & 0x1) << 3;
  s |= (((state >> 2) ^ (state >> 3) ^ (state >> 6) ^ (state >> 10) ^ (state >> 17) ^ (state >> 21) ^ (state >> 22)) & 0x1) << 4;
  s |= (((state >> 3) ^ (state >> 4) ^ (state >> 7) ^ (state >> 8) ^ (state >> 11) ^ (state >> 18) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 5;
  s |= (((state >> 4) ^ (state >> 5) ^ (state >> 9) ^ (state >> 12) ^ (state >> 19) ^ (state >> 23)) & 0x1) << 6;
  s |= (((state >> 5) ^ (state >> 6) ^ (state >> 10) ^ (state >> 13) ^ (state >> 20)) & 0x1) << 7;
  s |= (((state >> 5) ^ (state >> 14) ^ (state >> 15) ^ (state >> 16) ^ (state >> 19) ^ (state >> 22)) & 0x1) << 8;
  s |= (((state >> 5) ^ (state >> 6) ^ (state >> 8) ^ (state >> 14) ^ (state >> 16) ^ (state >> 17) ^ (state >> 19) ^ (state >> 20) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 9;
  s |= (((state >> 6) ^ (state >> 7) ^ (state >> 8) ^ (state >> 9) ^ (state >> 15) ^ (state >> 16) ^ (state >> 17) ^ (state >> 18) ^ (state >> 20) ^ (state >> 21) ^ (state >> 23)) & 0x1) << 10;
  s |= (((state >> 0) ^ (state >> 5) ^ (state >> 7) ^ (state >> 9) ^ (state >> 10) ^ (state >> 14) ^ (state >> 15) ^ (state >> 17) ^ (state >> 18) ^ (state >> 21)) & 0x1) << 11;
  s |= (((state >> 1) ^ (state >> 5) ^ (state >> 6) ^ (state >> 10) ^ (state >> 11) ^ (state >> 14) ^ (state >> 18)) & 0x1) << 12;
  s |= (((state >> 2) ^ (state >> 6) ^ (state >> 7) ^ (state >> 11) ^ (state >> 12) ^ (state >> 15) ^ (state >> 16) ^ (state >> 19)) & 0x1) << 13;
  s |= (((state >> 3) ^ (state >> 7) ^ (state >> 12) ^ (state >> 13) ^ (state >> 17) ^ (state >> 20)) & 0x1) << 14;
  s |= (((state >> 4) ^ (state >> 13) ^ (state >> 14) ^ (state >> 18) ^ (state >> 21)) & 0x1) << 15;
  s |= (((state >> 0) ^ (state >> 3) ^ (state >> 6) ^ (state >> 13) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 16;
  s |= (((state >> 0) ^ (state >> 1) ^ (state >> 3) ^ (state >> 4) ^ (state >> 6) ^ (state >> 7) ^ (state >> 13) ^ (state >> 14) ^ (state >> 16) ^ (state >> 22)) & 0x1) << 17;
  s |= (((state >> 0) ^ (state >> 1) ^ (state >> 2) ^ (state >> 4) ^ (state >> 5) ^ (state >> 7) ^ (state >> 14) ^ (state >> 15) ^ (state >> 16) ^ (state >> 17) ^ (state >> 23)) & 0x1) << 18;
  s |= (((state >> 1) ^ (state >> 2) ^ (state >> 5) ^ (state >> 8) ^ (state >> 13) ^ (state >> 15) ^ (state >> 17) ^ (state >> 18) ^ (state >> 22) ^ (state >> 23)) & 0x1) << 19;
  s |= (((state >> 2) ^ (state >> 9) ^ (state >> 13) ^ (state >> 14) ^ (state >> 18) ^ (state >> 19) ^ (state >> 22)) & 0x1) << 20;
  s |= (((state >> 0) ^ (state >> 3) ^ (state >> 10) ^ (state >> 14) ^ (state >> 15) ^ (state >> 19) ^ (state >> 20) ^ (state >> 23)) & 0x1) << 21;
  s |= (((state >> 1) ^ (state >> 4) ^ (state >> 11) ^ (state >> 15) ^ (state >> 20) ^ (state >> 21)) & 0x1) << 22;
  s |= (((state >> 2) ^ (state >> 5) ^ (state >> 12) ^ (state >> 21) ^ (state >> 22)) & 0x1) << 23;
  return s;
}

// LUT for linear layer
// these are filled in the generate_tables
static u8 LUT_L_INV_MSB_0[256] = {0};
static u8 LUT_L_INV_MSB_1[256] = {0};
static u8 LUT_L_INV_MSB_2[256] = {0};
static u8 LUT_L_INV_MIDDLESB_0[256] = {0};
static u8 LUT_L_INV_MIDDLESB_1[256] = {0};
static u8 LUT_L_INV_MIDDLESB_2[256] = {0};
static u8 LUT_L_INV_LSB_0[256] = {0};
static u8 LUT_L_INV_LSB_1[256] = {0};
static u8 LUT_L_INV_LSB_2[256] = {0};
static u8 LUT_L_MSB_0[256] = {0};
static u8 LUT_L_MSB_1[256] = {0};
static u8 LUT_L_MSB_2[256] = {0};
static u8 LUT_L_MIDDLESB_0[256] = {0};
static u8 LUT_L_MIDDLESB_1[256] = {0};
static u8 LUT_L_MIDDLESB_2[256] = {0};
static u8 LUT_L_LSB_0[256] = {0};
static u8 LUT_L_LSB_1[256] = {0};
static u8 LUT_L_LSB_2[256] = {0};
static u32 LUT_L_FROM_MSB[256] = {0};

static inline u8 L_INV_MSB(u32 s){
  return LUT_L_INV_MSB_2[(u8) s] ^ LUT_L_INV_MSB_1[(u8) (s >> 8)] ^ LUT_L_INV_MSB_0[(u8) (s >> 16)];
}
static inline u8 L_INV_MIDDLESB(u32 s){
  return LUT_L_INV_MIDDLESB_2[(u8) s] ^ LUT_L_INV_MIDDLESB_1[(u8) (s >> 8)] ^ LUT_L_INV_MIDDLESB_0[(u8) (s >> 16)];
}
static inline u8 L_INV_LSB(u32 s){
  return LUT_L_INV_LSB_2[(u8) s] ^ LUT_L_INV_LSB_1[(u8) (s >> 8)] ^ LUT_L_INV_LSB_0[(u8) (s >> 16)];
}
static inline u32 inv_linear_layer(u32 s){
  return L_INV_LSB(s) ^ (L_INV_MIDDLESB(s) << 8) ^ (L_INV_MSB(s) << 16);
}

static inline u8 L_MSB(u32 s){
  return LUT_L_MSB_2[(u8) s] ^ LUT_L_MSB_1[(u8) (s >> 8)] ^ LUT_L_MSB_0[(u8) (s >> 16)];
}
static inline u8 L_MIDDLESB(u32 s){
  return LUT_L_MIDDLESB_2[(u8) s] ^ LUT_L_MIDDLESB_1[(u8) (s >> 8)] ^ LUT_L_MIDDLESB_0[(u8) (s >> 16)];
}
static inline u8 L_LSB(u32 s){
  return LUT_L_LSB_2[(u8) s] ^ LUT_L_LSB_1[(u8) (s >> 8)] ^ LUT_L_LSB_0[(u8) (s >> 16)];
}
static inline u32 linear_layer(u32 s){
  return L_LSB(s) ^ (L_MIDDLESB(s) << 8) ^ (L_MSB(s) << 16);
}

void generate_tables(){
  // Build LUTs
  for(u32 s = 0; s < 0x100; s++){
    LUT_L_INV_MSB_2[s] =      ((u8) (inv_rotate_rows(inv_mix_columns(s <<  0)) >> 16));
    LUT_L_INV_MSB_1[s] =      ((u8) (inv_rotate_rows(inv_mix_columns(s <<  8)) >> 16));
    LUT_L_INV_MSB_0[s] =      ((u8) (inv_rotate_rows(inv_mix_columns(s << 16)) >> 16));
    LUT_L_INV_MIDDLESB_2[s] = ((u8) (inv_rotate_rows(inv_mix_columns(s <<  0)) >>  8));
    LUT_L_INV_MIDDLESB_1[s] = ((u8) (inv_rotate_rows(inv_mix_columns(s <<  8)) >>  8));
    LUT_L_INV_MIDDLESB_0[s] = ((u8) (inv_rotate_rows(inv_mix_columns(s << 16)) >>  8));
    LUT_L_INV_LSB_2[s] =      ((u8) (inv_rotate_rows(inv_mix_columns(s <<  0)) >>  0));
    LUT_L_INV_LSB_1[s] =      ((u8) (inv_rotate_rows(inv_mix_columns(s <<  8)) >>  0));
    LUT_L_INV_LSB_0[s] =      ((u8) (inv_rotate_rows(inv_mix_columns(s << 16)) >>  0));

    LUT_L_FROM_MSB[s]  =      (mix_columns(rotate_rows(s << 16)));

    LUT_L_MSB_2[s] =      ((u8) (mix_columns(rotate_rows(s <<  0)) >> 16));
    LUT_L_MSB_1[s] =      ((u8) (mix_columns(rotate_rows(s <<  8)) >> 16));
    LUT_L_MSB_0[s] =      ((u8) (mix_columns(rotate_rows(s << 16)) >> 16));
    LUT_L_MIDDLESB_2[s] = ((u8) (mix_columns(rotate_rows(s <<  0)) >>  8));
    LUT_L_MIDDLESB_1[s] = ((u8) (mix_columns(rotate_rows(s <<  8)) >>  8));
    LUT_L_MIDDLESB_0[s] = ((u8) (mix_columns(rotate_rows(s << 16)) >>  8));
    LUT_L_LSB_2[s] =      ((u8) (mix_columns(rotate_rows(s <<  0)) >>  0));
    LUT_L_LSB_1[s] =      ((u8) (mix_columns(rotate_rows(s <<  8)) >>  0));
    LUT_L_LSB_0[s] =      ((u8) (mix_columns(rotate_rows(s << 16)) >>  0));
  }

}

u32 inv_round_with_MC(u32 state, u32 round_key){
  state = state ^ round_key;
  state = inv_linear_layer(state);
  state = inv_sub_bytes(state);
  return state;
}
u32 inv_round_with_MC_inv_key(u32 state, u32 inv_round_key){
  state = inv_linear_layer(state);
  state = state ^ inv_round_key;
  state = inv_sub_bytes(state);
  return state;
}

u32 inv_round_no_MC(u32 state, u32 round_key){
  state = state ^ round_key;
  state = inv_rotate_rows(state);
  state = inv_sub_bytes(state);
  return state;
  //state = state ^ round_key;
  //return ((u32) inv_NO_MC_LUT_0[state] << 16) ^ ((u32) inv_NO_MC_LUT_1[state] << 8) ^ ((u32) inv_NO_MC_LUT_2[state]);
}

u32 round_with_MC(u32 state, u32 round_key){
  state = sub_bytes(state);
  state = linear_layer(state);
  state = state ^ round_key;
  return state;
}
u32 round_no_MC(u32 state, u32 round_key){
  state = sub_bytes(state);
  state = rotate_rows(state);
  state = state ^ round_key;
  return state;
}

u32 g(u32 key_word, u32 rc){
  u8 byte0 = key_word >> 24;
  u8 byte1 = (key_word >> 16) & 0xFF;
  u8 byte2 = (key_word >> 8) & 0xFF;
  u8 byte3 = (key_word >> 0) & 0xFF;
  key_word = ((SBOX[byte1] ^ rc) << 24) ^ (SBOX[byte2] << 16) ^ (SBOX[byte3] << 8) ^ SBOX[byte0];
  return key_word;
}

u32* key_schedule(u32 rk[11], u128 master_key, u64 seed){
  master_key = master_key ^ ((u128) seed << 64);
  rk[0] = (master_key >> 104) & 0xFFFFFF;
  rk[1] = (master_key >> 80) & 0xFFFFFF;
  rk[2] = (master_key >> 56) & 0xFFFFFF;
  rk[3] = (master_key >> 32) & 0xFFFFFF;
  rk[4] = (master_key >> 8) & 0xFFFFFF;
  rk[5] = (master_key & 0xFF) << 16;
  master_key ^= (u128) g(master_key & 0xFFFFFFFF, 1) << 96;
  master_key ^= ((master_key >> 96) & 0xFFFFFFFF) << 64;
  master_key ^= ((master_key >> 64) & 0xFFFFFFFF) << 32;
  master_key ^= ((master_key >> 32) & 0xFFFFFFFF) <<  0;
  rk[5] |= (master_key >> 112) & 0xFFFF;
  rk[6] = (master_key >> 88) & 0xFFFFFF;
  rk[7] = (master_key >> 64) & 0xFFFFFF;
  rk[8] = (master_key >> 40) & 0xFFFFFF;
  rk[9] = (master_key >> 16) & 0xFFFFFF;
  rk[10] = ((master_key >> 0) & 0xFFFF) << 8;
  master_key ^= (u128) g(master_key & 0xFFFFFFFF, 2) << 96;
  rk[10] |= (master_key >> 120) & 0xFF;
  return rk;
}

u32 encrypt(u32 state, u128 master_key, u64 seed){
  u32 rk[11] = {0};
  key_schedule(rk, master_key, seed);
  state = state ^ rk[0];
  for(int i = 1; i < 10; i++){
    state = round_with_MC(state, rk[i]);
  }
  state = round_no_MC(state, rk[10]);
  return state;
}


u32 decrypt(u32 state, u128 master_key, u64 seed){
  u32 rk[11] = {0};
  key_schedule(rk, master_key, seed);
  state = inv_round_no_MC(state, rk[10]);
  for(int i = 9; i > 0; i--){
    state = inv_round_with_MC(state, rk[i]);
  }
  state = state ^ rk[0];
  return state;
}

void test(){
 /* // Tests */
  u32 state = 0x7e47ce;
  if (sub_bytes(state) == 0xf3a08b) std::cout << "sub_bytes: OK!" << std::endl;
  else std::cout << "sub_bytes: BAD!" << std::endl;

  state = 0xf3a08b;
  if (inv_sub_bytes(state) == 0x7e47ce) std::cout << "Inverse sub_bytes: OK!" << std::endl;
  else std::cout << "Inverse sub_bytes: BAD!" << std::endl;

  state = 0xf3a08b;
  if (rotate_rows(state) == 0xf328b8) std::cout << "rotate_rows: OK!" << std::endl;
  else std::cout << "rotate_rows: BAD!" << std::endl;

  state = 0xf328b8;
  if (inv_rotate_rows(state) == 0xf3a08b) std::cout << "Inverse rotate_rows: OK!" << std::endl;
  else std::cout << "Inverse rotate_rows: BAD!" << std::endl;

  state = 0xf328b8;
  if (mix_columns(state) == 0x6936ac) std::cout << "mix_columns: OK!" << std::endl;
  else std::cout << "mix_columns: BAD!" << std::endl;

  state = 0x6936ac;
  if (inv_mix_columns(state) == 0xf328b8) std::cout << "Inverse mix_columns: OK!" << std::endl;
  else std::cout << "Inverse mix_columns: BAD!" << std::endl;

  u128 key = ((u128) 0x2b7e151628aed2a6 << 64) ^ 0xabf7158809cf4f3cULL;
  u64 seed = 0x543bd88000017550;
  u32 plain = 0x010203;
  u32 cipher = 0xf28c1e;
  if (encrypt(plain, key, seed) == cipher) std::cout << "Encrypt: OK!" << std::endl;
  else std::cout << "Encrypt: BAD!" << std::endl;

  if (decrypt(cipher, key, seed) == plain) std::cout << "Decrypt: OK!" << std::endl;
  else std::cout << "Decrypt: BAD!" << std::endl;
}
/////////////////////////////////////////
// END OF HALFLOOP-24 IMPLEMENTATION   //
/////////////////////////////////////////


/////////////////////////////////////////
// START OF AFFINE SUBSPACE STUFF      //
/////////////////////////////////////////
typedef __m256i subset_t;

void subset_print(auto name, const subset_t &var) {
  std::cout << name << ": 0b";
  u64 element = _mm256_extract_epi64(var, 3);
  for (int j = 63; j >= 0; j--) {
    u32 bit = (element >> j) & 1;
    std::cout << bit;
  }
  std::cout << " ";
  element = _mm256_extract_epi64(var, 2);
  for (int j = 63; j >= 0; j--) {
    u32 bit = (element >> j) & 1;
    std::cout << bit;
  }
  std::cout << " ";
  element = _mm256_extract_epi64(var, 1);
  for (int j = 63; j >= 0; j--) {
    u32 bit = (element >> j) & 1;
    std::cout << bit;
  }
  std::cout << " ";
  element = _mm256_extract_epi64(var, 0);
  for (int j = 63; j >= 0; j--) {
    u32 bit = (element >> j) & 1;
    std::cout << bit;
  }
  std::cout << std::endl;
}

inline subset_t subset_intersect(const subset_t &a, const subset_t &b){
  return _mm256_and_si256(a, b);
}

inline subset_t subset_union(const subset_t &a, const subset_t &b){
  return _mm256_or_si256(a, b);
}

inline u16 subset_size(const subset_t &a){
  u16 count = 0;
  u64 chunk = _mm256_extract_epi64(a, 0);
  count += __builtin_popcountll(chunk);
  chunk = _mm256_extract_epi64(a, 1);
  count += __builtin_popcountll(chunk);
  chunk = _mm256_extract_epi64(a, 2);
  count += __builtin_popcountll(chunk);
  chunk = _mm256_extract_epi64(a, 3);
  count += __builtin_popcountll(chunk);
  return count;
}

inline bool subset_is_empty(const subset_t &a){
  return _mm256_testz_si256(a, a);
}

subset_t subset_add_element(const subset_t &a, const u8 elm){
  // compute union of a and {elm}
  u64 mask[4] = {0};
  mask[elm/64] = (u64) 1 << (elm % 64);
  __m256i _mask = _mm256_set_epi64x(mask[3], mask[2], mask[1], mask[0]);
  return _mm256_or_si256(a, _mask);
}

subset_t subset_shift(const subset_t &b, const u8 shift){
  auto a = b;
  // compute a \oplus shift
  if((shift >> 7) & 0x1){
    a = _mm256_permute2x128_si256(a, a, 1);
  }
  if((shift >> 6) & 0x1){
    a = _mm256_permute4x64_epi64(a, _MM_SHUFFLE(2, 3, 0, 1));
  }
  if((shift >> 5) & 0x1){
    a = _mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
  }
  if((shift >> 4) & 0x1){
    a = _mm256_shufflelo_epi16(a, _MM_SHUFFLE(2, 3, 0, 1));
    a = _mm256_shufflehi_epi16(a, _MM_SHUFFLE(2, 3, 0, 1));
  }
  if((shift >> 3) & 0x1){
    const __m256i mask = _mm256_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    a = _mm256_shuffle_epi8(a, mask);
  }
  if((shift >> 2) & 0x1){
    const __m256i maskHigh = _mm256_set1_epi8((char) 0xF0);
    const __m256i maskLow = _mm256_set1_epi8(0x0F);
    const __m256i high = _mm256_and_si256(a, maskHigh);
    const __m256i low = _mm256_and_si256(a, maskLow);
    a = _mm256_or_si256(_mm256_srli_epi16(high, 4), _mm256_slli_epi16(low, 4));
  }
  if((shift >> 1) & 0x1){
    const __m256i maskHigh = _mm256_set1_epi8((char) 0xCC);
    const __m256i maskLow = _mm256_set1_epi8(0x33);
    const __m256i high = _mm256_and_si256(a, maskHigh);
    const __m256i low = _mm256_and_si256(a, maskLow);
    a = _mm256_or_si256(_mm256_srli_epi16(high, 2), _mm256_slli_epi16(low, 2));
  }
  if((shift >> 0) & 0x1){
    const __m256i maskHigh = _mm256_set1_epi8((char) 0xAA);
    const __m256i maskLow = _mm256_set1_epi8(0x55);
    const __m256i high = _mm256_and_si256(a, maskHigh);
    const __m256i low = _mm256_and_si256(a, maskLow);
    a = _mm256_or_si256(_mm256_srli_epi16(high, 1), _mm256_slli_epi16(low, 1));
  }
  return a;
}

std::vector<u8> subset_get_elements(const subset_t &a){
  std::vector<u8> e;
  u64 chunk = _mm256_extract_epi64(a, 0);
  while(chunk != 0){
    u8 idx = __builtin_ctzll(chunk) + 0*64;
    e.push_back(idx);
    chunk &= (chunk - 1);
  }
  chunk = _mm256_extract_epi64(a, 1);
  while(chunk != 0){
    u8 idx = __builtin_ctzll(chunk) + 1*64;
    e.push_back(idx);
    chunk &= (chunk - 1);
  }
  chunk = _mm256_extract_epi64(a, 2);
  while(chunk != 0){
    u8 idx = __builtin_ctzll(chunk) + 2*64;
    e.push_back(idx);
    chunk &= (chunk - 1);
  }
  chunk = _mm256_extract_epi64(a, 3);
  while(chunk != 0){
    u8 idx = __builtin_ctzll(chunk) + 3*64;
    e.push_back(idx);
    chunk &= (chunk - 1);
  }
  return e;
}

inline subset_t subset_init_empty(){
  return _mm256_setzero_si256();
}

inline subset_t subset_init_full(){
  return _mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,  0xFFFFFFFFFFFFFFFF);
}

/////////////////////////////////////////
// END OF AFFINE SUBSPACE STUFF        //
/////////////////////////////////////////


/////////////////////////////////////////
// START OF NEW ATTACK                 //
/////////////////////////////////////////

const  u8 N_PAIRS = 3;
u32 normalize_round_key(u32 round_key, u64 seed, u8 round){
  // self inverse
  switch (round) {
  case 0: return round_key ^ (seed >> 40);
  case 1: return round_key ^ ((seed >> 16) & 0xFFFFFF);
  case 2: return round_key ^ ((seed & 0xFFFF) << 8);
  case 3: return round_key;
  case 4: return round_key;
  case 5: return round_key ^ (seed >> 48);
  case 6: return round_key ^ (((seed >> 32) & 0xFFFF) << 16) ^ ((seed >> 56) ^ ((seed >> 40) & 0xFF));
  case 7: return round_key ^ ((seed >> 32) & 0xFFFFFF) ^ (seed & 0xFFFFFF);
  case 8: return round_key ^ ((seed >> 40) & 0xFFFFFF) ^ ((seed >> 8) & 0xFFFFFF);
  case 9: return round_key ^ ((seed >> 16) & 0xFFFFFF) ^ ((seed & 0xFF) << 16) ^ (seed >> 48);
  case 10: return round_key ^ ((((seed >> 32) & 0xFFFF) ^ (seed & 0xFFFF)) << 8) ^ (seed >> 56);
  default: return 0;
  }
}

u32 normalize_round_key_10(u32 round_key, u8 last_byte_round_key_9, u64 seed){
  return round_key ^ ((((seed >> 32) & 0xFFFF) ^ (seed & 0xFFFF)) << 8) ^ (seed >> 56) ^ SBOX[last_byte_round_key_9] ^ SBOX[last_byte_round_key_9 ^ ((seed >> 48) & 0xFF) ^ ((seed >> 16) & 0xFF)];
}
struct pair_t{
  u32 p : 24; // plaintext
  u64 t; // tweak = seed
  u8 d; // delta
  u32 c : 24; // ciphertext
  u32 c_prime : 24; // ciphertext'
};

void new_attack(){

  // step 0: fix key
  std::cout << "Step 0: Fix key" << std::endl;
  int error;
  u128 key;
  error = getentropy(&key, 16);
  if(error) std::cout << "BAD RNG" << std::endl;
  std::cout << "master key: 0x" << std::hex << (u64) (key >> 64) << (u64) key << std::endl;
  // ROUND KEYS FOR SHORTCUTS later
  u32 RK[11] = {0}; key_schedule(RK, key, 0);
  for(int i = 0; i < 11; i++){
    std::cout << "RK[" << i << "] = 0x" << std::hex << RK[i] << std::endl;
  }
  std::cout << "L^(-1)(RK[7])_0 = 0x" << std::hex << (inv_linear_layer(RK[7]) >> 16) << std::endl;
  // ---------------------
  std::cout << std::endl;

  // step 1: gather data
  // (in CPA setting)
  auto start = steady_clock::now();
  std::cout << "Step 1: Generating data:" << std::endl;
  pair_t PAIRS[N_PAIRS];
  for(u8 i = 0; i < N_PAIRS; i++){
    // pick random plaintext, tweak and (one byte) input difference
    u64 seed = 0;
    u32 plain = 0;
    u8 in_diff = 0;
    error = getentropy(&seed, 8);
    error = getentropy(&plain, 3);
    // generate in_diff s.t. in the end N_PAIRS different in_diff are used
    u8 new_in_diff;
    do {
      error = getentropy(&in_diff, 1);
      new_in_diff = true;
      if (in_diff == 0) new_in_diff = false;
      for(u8 j = 0; j < i; j++){
        if (in_diff == PAIRS[j].d) new_in_diff = false;
      }
    } while (!new_in_diff);

    PAIRS[i].p = plain;
    PAIRS[i].t = seed;
    PAIRS[i].d = in_diff;
    PAIRS[i].c = encrypt(plain, key, seed);
    PAIRS[i].c_prime = encrypt(plain ^ (u32) in_diff, key, seed ^ ((u64) in_diff << 40));
  }
  if(error) std::cout << "BAD RNG" << std::endl;
  auto stop = steady_clock::now();
  auto duration = duration_cast<seconds>(stop - start);
  std::cout << "Took " << std::dec << (2*N_PAIRS) << " queries and " << std::dec << duration.count() << "s" << std::endl;
  std::cout << std::endl;


  // step 2: precomputations
  start = steady_clock::now();
  std::cout << "Step 2: Precomputations" << std::endl;

  // Build DDT with specific values
  auto DDTV_out = new std::vector<u8> [256][256];
  for(u32 x = 0; x < 256; x++){
    for(u32 din = 0; din < 256; din++){
      u32 dout = SBOX[x] ^ SBOX[x ^ din];
      DDTV_out[din][dout].push_back(SBOX[(u8) x]);
    }
  }

  auto DDTV_out_shifted = new subset_t [256][256][256];
  for(unsigned int x = 0; x < 0x100; x++){
    for(unsigned int y = 0; y < 0x100; y++){
      for(unsigned int c = 0; c < 0x100; c++){
        DDTV_out_shifted[x][y][c] = subset_init_empty();
        for(u8 elm : DDTV_out[x][y]){
          DDTV_out_shifted[x][y][c] = subset_add_element(DDTV_out_shifted[x][y][c], elm ^ c);
        }
      }
    }
  }

  // precompute y for which delta_x -S-> delat_y is possible
  std::vector<u8> *POSSIBLE_DELTA_Y = new std::vector<u8> [256];
  for(unsigned int x = 0; x < 0x100; x++){
    for(unsigned int y = 0; y < 0x100; y++){
      if(!DDTV_out[x][y].empty()){
        POSSIBLE_DELTA_Y[x].push_back(y);
      }
    }
  }
  auto T = new subset_t [N_PAIRS][1 << 24][3];
  for(int i = 0; i < N_PAIRS; i++){
    for(u32 delta_z7 = 0; delta_z7 < (1 << 24); delta_z7++){
      for(int j = 0; j < 3; j++){
        T[i][delta_z7][j] = subset_init_empty();
      }
    }
  }
  for(int i = 0; i < N_PAIRS; i++){
    u8 din = PAIRS[i].d;
    for(u8 dout : POSSIBLE_DELTA_Y[din]){

      u32 delta_x7 = LUT_L_FROM_MSB[dout] ^ ((u32) din << 8);
      u8 delta_x7_2 = (u8) delta_x7;
      u8 delta_x7_1 = (u8) (delta_x7 >> 8);
      u8 delta_x7_0 = (u8) (delta_x7 >> 16);

      for(u8 delta_y7_0 : POSSIBLE_DELTA_Y[delta_x7_0]){
        for(u8 delta_y7_1 : POSSIBLE_DELTA_Y[delta_x7_1]){
          for(u8 delta_y7_2 : POSSIBLE_DELTA_Y[delta_x7_2]){
            u32 delta_y7 = ((u32) delta_y7_0 << 16) ^ ((u32) delta_y7_1 << 8) ^ (u32) delta_y7_2;
            u32 delta_z7 = linear_layer(delta_y7);
            T[i][delta_z7][0] = subset_union(T[i][delta_z7][0], DDTV_out_shifted[delta_x7_0][delta_y7_0][0]);
            T[i][delta_z7][1] = subset_union(T[i][delta_z7][1], DDTV_out_shifted[delta_x7_1][delta_y7_1][0]);
            T[i][delta_z7][2] = subset_union(T[i][delta_z7][2], DDTV_out_shifted[delta_x7_2][delta_y7_2][0]);
          }
        }
      }
    }
  }
  delete(DDTV_out);
  delete[](POSSIBLE_DELTA_Y);
  stop = steady_clock::now();
  duration = duration_cast<seconds>(stop - start);
  std::cout << "Took " << std::dec << duration.count() << "s" << std::endl;
  std::cout << std::endl;

  // step 3
  // enumerate (rk9, rk10) to find candidates ((L^-1 rk7)_0, rk8, rk9, rk10)
  start = steady_clock::now();
  std::cout << "Step 3: Identify key candidates" << std::endl;

  std::cout << "Checking " << MAX_RK10 * MAX_RK9 << " of 2**48 candidates for (rk9, rk10)." << std::endl;
  std::cout << "Using " << (u32) N_PAIRS << " pairs." << std::endl;

  // precompute normalization terms
  u32 norm_8_[N_PAIRS];
  u8 norm_8[3][N_PAIRS];
  for(int i = 0; i < N_PAIRS; i++){
    norm_8_[i] = inv_linear_layer(normalize_round_key(0, PAIRS[i].t, 8));
    norm_8[0][i] = (u8) (norm_8_[i] >> 16);
    norm_8[1][i] = (u8) (norm_8_[i] >> 8);
    norm_8[2][i] = (u8) norm_8_[i];
  }

  #if PARALLEL == 1
  std::cout << "omp_get_num_procs():   " << omp_get_num_procs() << std::endl;
  std::cout << "omp_get_max_threads(): " << omp_get_max_threads() << std::endl;
  #endif

  #if CHECK_CORRECT_FIRST == 1
  bool flag = true;
  #endif

  #if COUNTERS
  u64 CNT_rk8[3] = {0};
  u64 CNT_survives_rk8 = 0;
  u64 CNT_survives_Dy6 = 0;
  u64 CNT_survives_rk7 = 0;
  #endif

  #if PARALLEL == 1
    #if CHECK_CORRECT_FIRST == 1
      #if COUNTERS == 1
        #pragma omp parallel for shared(flag) reduction(+:CNT_rk8[:3], CNT_survives_rk8, CNT_survives_Dy6, CNT_survives_rk7)
      #else
        #pragma omp parallel for shared(flag)
      #endif
    #else
      #if COUNTERS == 1
        #pragma omp parallel for reduction(+:CNT_rk8[:3], CNT_survives_rk8, CNT_survives_Dy6, CNT_survives_rk7)
      #else
        #pragma omp parallel for
      #endif
    #endif
  #endif
  for(u32 rk10_ = 0; rk10_ < MAX_RK10; rk10_++){ // normalised keys
    for(u32 L_inv_rk9_ = 0; L_inv_rk9_ < MAX_RK9; L_inv_rk9_++){

      #if CHECK_CORRECT_FIRST == 1
      // correct guess instead of zero guess
      // correct means round keys for tweak PAIRS[0].t
      if (flag && (omp_get_thread_num() == 0)) {
        L_inv_rk9_ = inv_linear_layer(normalize_round_key(RK[9], PAIRS[0].t, 9));
        rk10_ = normalize_round_key_10(RK[10], (u8) linear_layer(L_inv_rk9_), PAIRS[0].t);
      }
      #endif

      // de-normalise keys
      u32 L_inv_rk9[N_PAIRS], rk10[N_PAIRS], rk10_PRIME[N_PAIRS];
      L_inv_rk9[0] = L_inv_rk9_;
      rk10[0] = rk10_;
      rk10_PRIME[0] = rk10[0] ^ ((u32) PAIRS[0].d << 16);
      for(int i = 1; i < N_PAIRS; i++){
        L_inv_rk9[i] = L_inv_rk9[0] ^ inv_linear_layer(normalize_round_key(0, PAIRS[0].t ^ PAIRS[i].t, 9));
        rk10[i] = normalize_round_key_10(normalize_round_key_10(rk10[0], (u8) linear_layer(L_inv_rk9[0]), PAIRS[0].t), (u8) linear_layer(L_inv_rk9[i]), PAIRS[i].t);
        rk10_PRIME[i] = rk10[i] ^ ((u32) PAIRS[i].d << 16);
      }

      // compute Delta_y7 from c, c', rk9, rk10
      u32 x8[N_PAIRS], x8_PRIME[N_PAIRS], delta_z7[N_PAIRS], v8_[N_PAIRS];
      u8 v8[3][N_PAIRS];
      subset_t *bytes_pair[N_PAIRS];
      for(int i = 0; i < N_PAIRS; i++){

        x8[i] = inv_round_with_MC_inv_key(inv_round_no_MC(PAIRS[i].c, rk10[i]), L_inv_rk9[i]);
        x8_PRIME[i] = inv_round_with_MC_inv_key(inv_round_no_MC(PAIRS[i].c_prime, rk10_PRIME[i]), L_inv_rk9[i]);

        delta_z7[i] = x8[i] ^ x8_PRIME[i] ^ ((u32) PAIRS[i].d);

        bytes_pair[i] = T[i][delta_z7[i]];

        v8_[i] = inv_linear_layer(x8[i]);
        v8[0][i] = (u8) (v8_[i] >> 16);
        v8[1][i] = (u8) (v8_[i] >> 8);
        v8[2][i] = (u8) v8_[i];
      }

      #if COUNTERS == 1
      subset_t intersection[3];
      for(int j = 0; j < 3; j++){
        // fast rejection with byte j
        CNT_rk8[j] += subset_size(bytes_pair[0][j]);
        intersection[j] = bytes_pair[0][j];
        for(int i = 1; i < N_PAIRS; i++){
          CNT_rk8[j] += subset_size(bytes_pair[i][j]);
          subset_t b = subset_shift(bytes_pair[i][j], v8[j][0] ^ norm_8[j][0] ^ v8[j][i] ^ norm_8[j][i]);
          intersection[j] = subset_intersect(intersection[j], b);
        }
      }
      for(int j = 0; j < 3; j++) if(subset_is_empty(intersection[j])) goto bad_guess_;
      CNT_survives_rk8++;
      #else
      subset_t intersection[3];
      for(int j = 0; j < 3; j++){
        // fast rejection with byte j
        intersection[j] = bytes_pair[0][j];
        for(int i = 1; i < N_PAIRS; i++){
          subset_t b = subset_shift(bytes_pair[i][j], v8[j][0] ^ norm_8[j][0] ^ v8[j][i] ^ norm_8[j][i]);
          intersection[j] = subset_intersect(intersection[j], b);
        }
        if(subset_is_empty(intersection[j])) goto bad_guess_;
      }
      #endif

      for(u8 rk8_0 : subset_get_elements(intersection[0])){
        rk8_0 ^= v8[0][0] ^ norm_8[0][0];
        for(u8 rk8_1 : subset_get_elements(intersection[1])){
          rk8_1 ^= v8[1][0] ^ norm_8[1][0];
          for(u8 rk8_2 : subset_get_elements(intersection[2])){
            rk8_2 ^= v8[2][0] ^ norm_8[2][0];
            u32 rk8;
            rk8 = linear_layer(((u32) rk8_0 << 16) ^ ((u32) rk8_1 << 8) ^ ((u32) rk8_2));
            subset_t L_inv_rk7_0;
            L_inv_rk7_0 = subset_init_full();
            for(int i = 0; i < N_PAIRS; i++){
              u32 rk8_normalised = normalize_round_key(rk8, PAIRS[i].t, 8);
              u32 rk8_PRIME_normalised = rk8_normalised ^ ((u32) PAIRS[i].d);
              u32 v7 = inv_linear_layer(inv_round_with_MC(x8[i], rk8_normalised));
              u32 v7_PRIME = inv_linear_layer(inv_round_with_MC(x8_PRIME[i], rk8_PRIME_normalised) ^ ((u32) PAIRS[i].d << 8));
              if(((v7 ^ v7_PRIME) & 0x00FFFF) != 0) goto next_rk_8;
              #if COUNTERS == 1
              CNT_survives_Dy6++;
              #endif
              u8 delta_v7_0 = (u8) ((v7 ^ v7_PRIME) >> 16);
              u8 norm_7_0 = (u8) (inv_linear_layer(normalize_round_key(0, PAIRS[i].t, 7)) >> 16);
              u8 v7_0 = (u8) (v7 >> 16);
              L_inv_rk7_0 = subset_intersect(L_inv_rk7_0, DDTV_out_shifted[PAIRS[i].d][delta_v7_0][v7_0 ^ norm_7_0]);
            }
            for(u8 L_inv_rk7_0_ : subset_get_elements(L_inv_rk7_0)){
              #if COUNTERS == 1
              CNT_survives_rk7++;
              #endif
              std::cout << "Candidate: L_inv_rk7_0 = 0x" << std::hex << (u32) L_inv_rk7_0_ << ", rk8 = 0x" << rk8;
              std::cout << ", rk9 = 0x" << normalize_round_key(linear_layer(L_inv_rk9_), PAIRS[0].t, 9);
              std::cout << ", rk10 = 0x" << normalize_round_key_10(rk10_, (u8) linear_layer(L_inv_rk9_), PAIRS[0].t) << std::endl;
            }
          next_rk_8:;
          }
        }
      }
    bad_guess_:;

      #if CHECK_CORRECT_FIRST == 1
      // set keys back to zero and flip flag in first iteration
      if (flag && (omp_get_thread_num() == 0)) {
        L_inv_rk9_ = 0;
        rk10_ = 0;
        flag = false;
      }
      #endif
    }
  }
  stop = steady_clock::now();
  auto duration_ns = duration_cast<nanoseconds>(stop - start);
  std::cout << "Took      " << std::dec << duration_ns.count() << "ns = " << (MAX_RK10 * MAX_RK9) << " * " << duration_ns.count()/(MAX_RK10 * MAX_RK9) << "ns" << std::endl;
  #if COUNTERS == 1
  std::cout << "Notice that the timings are effected by the counting! To benchamrk performance set COUTNERS to 0" << std::endl;
  std::cout << std::endl;
  std::cout << "Average number of candidaets for rk^{(8)}_0: " << (double) CNT_rk8[0] / (MAX_RK10 * MAX_RK9) / N_PAIRS << std::endl;
  std::cout << "Average number of candidaets for rk^{(8)}_1: " << (double) CNT_rk8[1] / (MAX_RK10 * MAX_RK9) / N_PAIRS << std::endl;
  std::cout << "Average number of candidaets for rk^{(8)}_2: " << (double) CNT_rk8[2] / (MAX_RK10 * MAX_RK9) / N_PAIRS << std::endl;
  std::cout << "Survived rk8 filter: " << (double) CNT_survives_rk8 / (MAX_RK10 * MAX_RK9) << std::endl;
  std::cout << "Survived Delta y6 filter: " << (double) CNT_survives_Dy6 / (MAX_RK10 * MAX_RK9) << std::endl;
  std::cout << "Survived rk7 filter: " << (double) CNT_survives_rk7 / (MAX_RK10 * MAX_RK9) << std::endl;
  #endif
  std::cout << std::endl;

  delete(DDTV_out_shifted);
  delete(T);
  // Step4: Brute force remaining key bits: same as in [DDLS22] and therefore omitted
}
/////////////////////////////////////////
// END OF NEW ATTACK                   //
/////////////////////////////////////////

// this function is used to generate the data for the figure
// regarding the number of candidates for rk8
void compute_number_of_rk8_candidates(){
  // for every non-zero delta and every Delta y_7 compute the number of possible rk8
  std::cout << "Computing distributions of |RK^{(8)}_j|" << std::endl;

  // compute DDT
  auto DDT = new u32 [256][256];
  for(u32 x = 0; x < 256; x++){
    for(u32 din = 0; din < 256; din++){
      u32 dout = SBOX[x] ^ SBOX[x ^ din];
      DDT[din][dout]++;
    }
  }

  for(int j = 0; j < 4; j++){
    std::cout << "Running computation for j = " << std::dec << j << std::endl;
    auto N_CAND = new u16[256][1 << 24];
    for(u32 delta = 1; delta < 256; delta++){
      std::cout << "#" << std::flush;
      for(u32 gamma = 1; gamma < 256; gamma++){
        if(DDT[delta][gamma] == 0) continue;
        u32 delta_x7 = mix_columns(rotate_rows(gamma << 16)) ^ (delta << 8);
        u8 delta_x7_0 = (u8) (delta_x7 >> 16);
        u8 delta_x7_1 = (u8) (delta_x7 >> 8);
        u8 delta_x7_2 = (u8) delta_x7;

        for(u32 delta_y7_0 = 0; delta_y7_0 < 256; delta_y7_0++){
          if(DDT[delta_x7_0][delta_y7_0] == 0) continue;

          for(u32 delta_y7_1 = 0; delta_y7_1 < 256; delta_y7_1++){
            if(DDT[delta_x7_1][delta_y7_1] == 0) continue;

            for(u32 delta_y7_2 = 0; delta_y7_2 < 256; delta_y7_2++){
              if(DDT[delta_x7_2][delta_y7_2] == 0) continue;

              u32 delta_y7 = (delta_y7_0 << 16) ^ (delta_y7_1 << 8) ^ delta_y7_2;
              if(j == 0) N_CAND[delta][delta_y7] += DDT[delta_x7_0][delta_y7_0];
              else if(j == 1) N_CAND[delta][delta_y7] += DDT[delta_x7_1][delta_y7_1];
              else if(j == 2) N_CAND[delta][delta_y7] += DDT[delta_x7_2][delta_y7_2];
              else if(j == 3) N_CAND[delta][delta_y7] += DDT[delta_x7_0][delta_y7_0]*DDT[delta_x7_1][delta_y7_1]*DDT[delta_x7_2][delta_y7_2];
            }
          }
        }
      }
    }
    std::cout << std::endl << "finished computation" << std::endl;
    u32 max = 0;
    double avg = 0;
    u32 hist[4097] = {0}; // 4097 is max_j so enough for all j
    u32 cnt_zero = 0;
    for(u32 delta = 1; delta < 256; delta++){
      for(u32 delta_y7 = 0; delta_y7 < (1 << 24); delta_y7++){
        u16 n = N_CAND[delta][delta_y7];
        if(n > max) max = n;
        avg += n;
        hist[n]++;
        if(n == 0) cnt_zero++;
      }
    }
    avg = avg / ((double) 255 * (1 << 24));
    std::cout << "MAX: " << std::dec << max << std::endl;
    std::cout << "Avg: " << std::dec << avg << std::endl;
    std::cout << "#Zeros: " << std::dec << cnt_zero << std::endl;

    for(int i = 0; i < 4097; i++){
      if(hist[i] == 0) continue;
      std::cout << "HIST_" << std::dec << j << "[" << i << "] = " << hist[i] << std::endl;
    }
    delete(N_CAND);
  }
}

int main() {
  /////////////////
  generate_tables(); // never remove!
  test();
  /////////////////

  // generate data for figures in papaer
  // compute_number_of_rk8_candidates();
  // return 0;

  std::cout << std::endl;
  std::cout << "FLAGS: " << std::endl;
  std::cout << "  - CHECK_CORRECT_FIRST: " << CHECK_CORRECT_FIRST << std::endl;
  std::cout << "  - COUNTERS: " << COUNTERS << std::endl;
  std::cout << "  - PARALLEL: " << PARALLEL << std::endl;
  std::cout << "Running the attack " << std::dec << REP << " times..." << std::endl;
  std::cout << std::endl;

  // experimentally verify our new attack
  for(unsigned int i = 0; i < REP; i++){
    std::cout << "Run " << std::dec << i << ":" << std::endl;
    new_attack();
  }
  return 0;
}
