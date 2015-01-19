/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2014 Facebook, Inc. (http://www.facebook.com)     |
   +----------------------------------------------------------------------+
   | This source file is subject to version 3.01 of the PHP license,      |
   | that is bundled with this package in the file LICENSE, and is        |
   | available through the world-wide-web at the following url:           |
   | http://www.php.net/license/3_01.txt                                  |
   | If you did not receive a copy of the PHP license and are unable to   |
   | obtain it through the world-wide-web, please send a note to          |
   | license@php.net so we can mail you a copy immediately.               |
   +----------------------------------------------------------------------+
*/

#ifndef incl_HPHP_TRACING_COLLECTOR_IMPL_H_
#define incl_HPHP_TRACING_COLLECTOR_IMPL_H_

#include "hphp/runtime/base/tracing-collector.h"
#include "hphp/util/thread-local.h"
#include "hphp/runtime/base/base-includes.h"
#include "hphp/runtime/base/memory-manager.h"
#include "hphp/runtime/base/heap-tracer.h"
#include "hphp/runtime/vm/srckey.h"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <map>


namespace std {
  template<class E>class hash {
    using sfinae = typename std::enable_if<std::is_enum<E>::value, E>::type;
  public:
    size_t operator()(const E&e) const {
      return std::hash<typename std::underlying_type<E>::type>()(e);
    }
  };
};

namespace HPHP {

///////////////////////////////////////////////////////////////////////////////


typedef std::vector<SrcKey> AllocStackTrace;

class MarkSweepCollector {
public:
  typedef ThreadLocalSingleton<MarkSweepCollector> TlsWrapper;
  static void Create(void *);
  static void Delete(MarkSweepCollector *);
  static void OnThreadExit(MarkSweepCollector *);

public:
  MarkSweepCollector() {}
  ~MarkSweepCollector() {}

  int64_t collect();

  /**
   * Mark an object as having a destructor that will need to
   * be called when it is collected.
   */
  void markDestructable(const ObjectData *obj);

  void markObjectLive(void *ptr, DataType t);

  void printStack();

private:
  static constexpr size_t alignBits = MemoryManager::kSlabAlignment - 1;

  struct SlabData {
    MemoryManager::Slab slab;
    std::bitset<MemoryManager::kBlocksPerSlab> usedBlocks;
  };

  /*
   * Grab the data about current slabs from the memory
   * manager and process it into something useful
   */
  void prepareSlabData();

  /*
   * Clean out any data that can't be reused next time
   * we collect
   */
  void cleanData();

  /*
   * Find the slab data for any pointer.
   * Returns NULL if the pointer is not within a valid slab
   */
  SlabData *getSlabData(void *ptr);

  int sweepHeap() {return 1;}

  void markReachable(void *ptr);

  /*
   * Returns true if this pointer is within a reachable
   * block. Does not mean that the pointer itself is
   * reachable.
   */
  bool isBlockReachable(void *ptr);

  void checkTraceSet();

  //A vector is probably not the best data structure for this,
  //depending on the size of the data.
  //Something that is segregated based on memory block will be
  //easier to handle at collection time.
  std::vector<const ObjectData *> destructable;

  std::vector<SlabData> m_slabs;

  std::unordered_map<void *, size_t> slabLookup;

  std::unordered_set<std::pair<void *, DataType> > liveSet;

  std::unordered_set<std::pair<void *, DataType> > traceSet;

  std::unordered_map<void *, AllocStackTrace> allocTraces;

  HeapTracer tracer;
};

MarkSweepCollector &gc();

///////////////////////////////////////////////////////////////////////////////

}

#endif
