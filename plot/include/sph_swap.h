#ifndef SPH_SWAP_H__
#define SPH_SWAP_H__

#define SPH_SWAP32(v)                           \
    ((SPH_ROTL32(v, 8) & SPH_C32(0x00FF00FF)) | \
     (SPH_ROTL32(v, 24) & SPH_C32(0xFF00FF00)))

#define SPH_SWAP64(v) (((uint64_t)SPH_SWAP32(SPH_T32(v)) << 32) | (uint64_t)SPH_SWAP32(SPH_T32(v >> 32)))

#endif