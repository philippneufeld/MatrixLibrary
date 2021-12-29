// Copyright 2021, Philipp Neufeld

#ifndef ML_Memory_AlignedAlloc_H_
#define ML_Memory_AlignedAlloc_H_

// Includes
#include <cstdint>
#include <cstdlib>

namespace ML
{
  template<typename T>
  T* MLAlignedAlloc(std::size_t size, std::size_t alignment)
  {
    if (size == 0)
      return nullptr;
    
    // padd the memory by so much that the pointer can be aligned and 
    // a void* pointer can be stored in front of the aligned block
    std::size_t padding = alignment + sizeof(void*);
    void* mem = std::malloc(size * sizeof(T) + padding);

    // find and aligned memory address
    std::uintptr_t uptr = ((reinterpret_cast<std::uintptr_t>(mem) + padding) / alignment) * alignment;

    // store originally allocated memory pointer in front of the aligned block
    *(reinterpret_cast<void**>(uptr) - 1) = mem; 

    // return the aligned pointer
    return reinterpret_cast<T*>(uptr);
  }

    template<typename T>
    void MLAlignedFree(T*& ptr)
    {
      if (ptr)
      {
        // retrieve the original memory pointer and free it
        std::free(*(reinterpret_cast<void**>(ptr) - 1));
      }
      ptr = nullptr;
    }
}

#endif
