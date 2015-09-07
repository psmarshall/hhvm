/*
   +----------------------------------------------------------------------+
   | HipHop for PHP                                                       |
   +----------------------------------------------------------------------+
   | Copyright (c) 2010-2015 Facebook, Inc. (http://www.facebook.com)     |
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
#include "hphp/runtime/base/memory-manager.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

#include "hphp/runtime/base/builtin-functions.h"
#include "hphp/runtime/base/exceptions.h"
#include "hphp/runtime/base/memory-profile.h"
#include "hphp/runtime/base/runtime-option.h"
#include "hphp/runtime/base/stack-logger.h"
#include "hphp/runtime/base/surprise-flags.h"
#include "hphp/runtime/base/sweepable.h"
#include "hphp/runtime/base/thread-info.h"
#include "hphp/runtime/server/http-server.h"

#include "hphp/util/alloc.h"
#include "hphp/util/logger.h"
#include "hphp/util/process.h"
#include "hphp/util/trace.h"

#include <folly/Random.h>
#include <folly/ScopeGuard.h>
#include "hphp/runtime/base/memory-manager-defs.h"

namespace HPHP {

TRACE_SET_MOD(mm);

//////////////////////////////////////////////////////////////////////

std::atomic<MemoryManager::ReqProfContext*>
  MemoryManager::s_trigger{nullptr};

// generate mmap flags for contiguous heap
uint32_t getRequestHeapFlags() {
  struct stat buf;

  // check if MAP_UNITIALIZED is supported
  auto mapUninitializedSupported =
    (stat("/sys/kernel/debug/fb_map_uninitialized", &buf) == 0);
  auto mmapFlags = MAP_NORESERVE | MAP_ANON | MAP_PRIVATE;

  /* Check whether mmap(2) supports the MAP_UNINITIALIZED flag. */
 if (mapUninitializedSupported) {
    mmapFlags |= MAP_UNINITIALIZED;
  }
 return mmapFlags;
}

static auto s_mmapFlags = getRequestHeapFlags();

#ifdef USE_JEMALLOC
bool MemoryManager::s_statsEnabled = false;
size_t MemoryManager::s_cactiveLimitCeiling = 0;

static size_t threadAllocatedpMib[2];
static size_t threadDeallocatedpMib[2];
static size_t statsCactiveMib[2];
static pthread_once_t threadStatsOnce = PTHREAD_ONCE_INIT;

void MemoryManager::threadStatsInit() {
  if (!mallctlnametomib) return;
  size_t miblen = sizeof(threadAllocatedpMib) / sizeof(size_t);
  if (mallctlnametomib("thread.allocatedp", threadAllocatedpMib, &miblen)) {
    return;
  }
  miblen = sizeof(threadDeallocatedpMib) / sizeof(size_t);
  if (mallctlnametomib("thread.deallocatedp", threadDeallocatedpMib, &miblen)) {
    return;
  }
  miblen = sizeof(statsCactiveMib) / sizeof(size_t);
  if (mallctlnametomib("stats.cactive", statsCactiveMib, &miblen)) {
    return;
  }
  MemoryManager::s_statsEnabled = true;

  // In threadStats() we wish to solve for cactiveLimit in:
  //
  //   footprint + cactiveLimit + headRoom == MemTotal
  //
  // However, headRoom comes from RuntimeOption::ServerMemoryHeadRoom, which
  // isn't initialized until after the code here runs.  Therefore, compute
  // s_cactiveLimitCeiling here in order to amortize the cost of introspecting
  // footprint and MemTotal.
  //
  //   cactiveLimit == (MemTotal - footprint) - headRoom
  //
  //   cactiveLimit == s_cactiveLimitCeiling - headRoom
  // where
  //   s_cactiveLimitCeiling == MemTotal - footprint
  size_t footprint = Process::GetCodeFootprint(Process::GetProcessId());
  size_t MemTotal  = 0;
#ifndef __APPLE__
  size_t pageSize = size_t(sysconf(_SC_PAGESIZE));
  MemTotal = size_t(sysconf(_SC_PHYS_PAGES)) * pageSize;
#else
  int mib[2] = { CTL_HW, HW_MEMSIZE };
  u_int namelen = sizeof(mib) / sizeof(mib[0]);
  size_t len = sizeof(MemTotal);
  sysctl(mib, namelen, &MemTotal, &len, nullptr, 0);
#endif
  if (MemTotal > footprint) {
    MemoryManager::s_cactiveLimitCeiling = MemTotal - footprint;
  }
}

inline
void MemoryManager::threadStats(uint64_t*& allocated, uint64_t*& deallocated,
                                size_t*& cactive, size_t& cactiveLimit) {
  pthread_once(&threadStatsOnce, threadStatsInit);
  if (!MemoryManager::s_statsEnabled) return;

  size_t len = sizeof(allocated);
  if (mallctlbymib(threadAllocatedpMib,
                   sizeof(threadAllocatedpMib) / sizeof(size_t),
                   &allocated, &len, nullptr, 0)) {
    not_reached();
  }

  len = sizeof(deallocated);
  if (mallctlbymib(threadDeallocatedpMib,
                   sizeof(threadDeallocatedpMib) / sizeof(size_t),
                   &deallocated, &len, nullptr, 0)) {
    not_reached();
  }

  len = sizeof(cactive);
  if (mallctlbymib(statsCactiveMib,
                   sizeof(statsCactiveMib) / sizeof(size_t),
                   &cactive, &len, nullptr, 0)) {
    not_reached();
  }

  int64_t headRoom = RuntimeOption::ServerMemoryHeadRoom;
  // Compute cactiveLimit based on s_cactiveLimitCeiling, as computed in
  // threadStatsInit().
  if (headRoom != 0 && headRoom < MemoryManager::s_cactiveLimitCeiling) {
    cactiveLimit = MemoryManager::s_cactiveLimitCeiling - headRoom;
  } else {
    cactiveLimit = std::numeric_limits<size_t>::max();
  }
}
#endif

static void* MemoryManagerInit() {
  MemoryManager::TlsWrapper tls;
  return (void*)tls.getNoCheck;
}

void* MemoryManager::TlsInitSetup = MemoryManagerInit();

void MemoryManager::Create(void* storage) {
  new (storage) MemoryManager();
}

void MemoryManager::Delete(MemoryManager* mm) {
  mm->~MemoryManager();
}

void MemoryManager::OnThreadExit(MemoryManager* mm) {
  mm->~MemoryManager();
}

MemoryManager::MemoryManager()
    : m_lineCursor(nullptr)
    , m_lineLimit(nullptr)
    , m_blockCursor(nullptr)
    , m_blockLimit(nullptr)
    , m_sweeping(false) {
#ifdef USE_JEMALLOC
  threadStats(m_allocated, m_deallocated, m_cactive, m_cactiveLimit);
#endif
  resetStatsImpl(true);
  m_stats.maxBytes = std::numeric_limits<int64_t>::max();
  // make the circular-lists empty.
  m_strings.next = m_strings.prev = &m_strings;
  m_bypassSlabAlloc = RuntimeOption::DisableSmallAllocator;
}

void MemoryManager::addExceptionRoot(ExtendedException* exn) {
  m_exceptionRoots.push_back(exn);
  // The key is the index into the exception root list (biased by 1).
  exn->m_key.m_index = m_exceptionRoots.size();
}

void MemoryManager::removeExceptionRoot(ExtendedException* exn) {
  // Swap the exception being removed with the last exception in the list (which
  // might be the same one). This lets us remove from the vector in constant
  // time.
  assert(exn->m_key.m_index > 0 &&
         exn->m_key.m_index <= m_exceptionRoots.size());
  auto& removed = m_exceptionRoots[exn->m_key.m_index-1];
  assert(removed == exn);
  auto& back = m_exceptionRoots.back();
  assert(back->m_key.m_index == m_exceptionRoots.size());
  back->m_key = removed->m_key;
  removed->m_key.m_index = 0;
  removed = back;
  m_exceptionRoots.pop_back();
}

void MemoryManager::dropRootMaps() {
  m_objectRoots = nullptr;
  m_resourceRoots = nullptr;
  m_exceptionRoots = std::vector<ExtendedException*>();
}

void MemoryManager::deleteRootMaps() {
  if (m_objectRoots) {
    req::destroy_raw(m_objectRoots);
    m_objectRoots = nullptr;
  }
  if (m_resourceRoots) {
    req::destroy_raw(m_resourceRoots);
    m_resourceRoots = nullptr;
  }
  m_exceptionRoots = std::vector<ExtendedException*>();
}

void MemoryManager::resetRuntimeOptions() {
  TRACE(2, "resetRuntimeOptions called\n");
  if (debug) {
    deleteRootMaps();
    checkHeap();
    // check that every allocation in heap has been freed before reset
    // iterate([&](Header* h) {
    //   assert(h->kind() == HeaderKind::Free || h->hdr_.mrb);
    //   if (h->hdr_.mrb) {
    //     TRACE(1, "leaked due to mrb: hdr %p\n", h);
    //   }
    // });
  }
  MemoryManager::TlsWrapper::destroy(); // ~MemoryManager()
  MemoryManager::TlsWrapper::getCheck(); // new MemeoryManager()
}

void MemoryManager::resetStatsImpl(bool isInternalCall) {
#ifdef USE_JEMALLOC
  FTRACE(1, "resetStatsImpl({}) pre:\n", isInternalCall);
  FTRACE(1, "usage: {}\nalloc: {}\npeak usage: {}\npeak alloc: {}\n",
    m_stats.usage, m_stats.alloc, m_stats.peakUsage, m_stats.peakAlloc);
  FTRACE(1, "total alloc: {}\nje alloc: {}\nje dealloc: {}\n",
    m_stats.totalAlloc, m_prevAllocated, m_prevDeallocated);
  FTRACE(1, "je debt: {}\n\n", m_stats.jemallocDebt);
#else
  FTRACE(1, "resetStatsImpl({}) pre:\n"
    "usage: {}\nalloc: {}\npeak usage: {}\npeak alloc: {}\n\n",
    isInternalCall,
    m_stats.usage, m_stats.alloc, m_stats.peakUsage, m_stats.peakAlloc);
#endif
  if (isInternalCall) {
    m_statsIntervalActive = false;
    m_stats.usage = 0;
    m_stats.alloc = 0;
    m_stats.peakUsage = 0;
    m_stats.peakAlloc = 0;
    m_stats.totalAlloc = 0;
    m_stats.peakIntervalUsage = 0;
    m_stats.peakIntervalAlloc = 0;
#ifdef USE_JEMALLOC
    m_enableStatsSync = false;
  } else if (!m_enableStatsSync) {
#else
  } else {
#endif
    // This is only set by the jemalloc stats sync which we don't enable until
    // after this has been called.
    assert(m_stats.totalAlloc == 0);
#ifdef USE_JEMALLOC
    assert(m_stats.jemallocDebt >= m_stats.alloc);
#endif

    // The effect of this call is simply to ignore anything we've done *outside*
    // the MemoryManager allocator after we initialized to avoid attributing
    // shared structure initialization that happens during hphp_thread_init()
    // to this session.

    // We don't want to clear the other values because we do already have some
    // small-sized allocator usage and live slabs and wiping now will result in
    // negative values when we try to reconcile our accounting with jemalloc.
#ifdef USE_JEMALLOC
    // Anything that was definitively allocated by the MemoryManager allocator
    // should be counted in this number even if we're otherwise zeroing out
    // the count for each thread.
    m_stats.totalAlloc = s_statsEnabled ? m_stats.jemallocDebt : 0;

    m_enableStatsSync = s_statsEnabled;
#else
    m_stats.totalAlloc = 0;
#endif
  }
#ifdef USE_JEMALLOC
  if (s_statsEnabled) {
    m_stats.jemallocDebt = 0;
    m_prevDeallocated = *m_deallocated;
    m_prevAllocated = *m_allocated;
  }
#endif
#ifdef USE_JEMALLOC
  FTRACE(1, "resetStatsImpl({}) post:\n", isInternalCall);
  FTRACE(1, "usage: {}\nalloc: {}\npeak usage: {}\npeak alloc: {}\n",
    m_stats.usage, m_stats.alloc, m_stats.peakUsage, m_stats.peakAlloc);
  FTRACE(1, "total alloc: {}\nje alloc: {}\nje dealloc: {}\n",
    m_stats.totalAlloc, m_prevAllocated, m_prevDeallocated);
  FTRACE(1, "je debt: {}\n\n", m_stats.jemallocDebt);
#else
  FTRACE(1, "resetStatsImpl({}) post:\n"
    "usage: {}\nalloc: {}\npeak usage: {}\npeak alloc: {}\n\n",
    isInternalCall,
    m_stats.usage, m_stats.alloc, m_stats.peakUsage, m_stats.peakAlloc);
#endif
}

void MemoryManager::refreshStatsHelperExceeded() {
  setSurpriseFlag(MemExceededFlag);
  m_couldOOM = false;
  if (RuntimeOption::LogNativeStackOnOOM) {
    log_native_stack("Exceeded memory limit");
  }
}

#ifdef USE_JEMALLOC
void MemoryManager::refreshStatsHelperStop() {
  HttpServer::Server->stop();
  // Increase the limit to the maximum possible value, so that this method
  // won't be called again.
  m_cactiveLimit = std::numeric_limits<size_t>::max();
}
#endif

/*
 * Refresh stats to reflect directly malloc()ed memory, and determine
 * whether the request memory limit has been exceeded.
 *
 * The stats parameter allows the updates to be applied to either
 * m_stats as in refreshStats() or to a separate MemoryUsageStats
 * struct as in getStatsSafe().
 *
 * The template variable live controls whether or not MemoryManager
 * member variables are updated and whether or not to call helper
 * methods in response to memory anomalies.
 */
template<bool live>
void MemoryManager::refreshStatsImpl(MemoryUsageStats& stats) {
#ifdef USE_JEMALLOC
  // Incrementally incorporate the difference between the previous and current
  // deltas into the memory usage statistic.  For reference, the total
  // malloced memory usage could be calculated as such, if delta0 were
  // recorded in resetStatsImpl():
  //
  //   int64 musage = delta - delta0;
  //
  // Note however, the slab allocator adds to m_stats.jemallocDebt
  // when it calls malloc(), so that this function can avoid
  // double-counting the malloced memory. Thus musage in the example
  // code may well substantially exceed m_stats.usage.
  if (m_enableStatsSync) {
    uint64_t jeDeallocated = *m_deallocated;
    uint64_t jeAllocated = *m_allocated;

    // We can't currently handle wrapping so make sure this isn't happening.
    assert(jeAllocated >= 0 &&
           jeAllocated <= std::numeric_limits<int64_t>::max());
    assert(jeDeallocated >= 0 &&
           jeDeallocated <= std::numeric_limits<int64_t>::max());

    // This is the delta between the current and the previous jemalloc reading.
    int64_t jeMMDeltaAllocated =
      int64_t(jeAllocated) - int64_t(m_prevAllocated);

    FTRACE(1, "Before stats sync:\n");
    FTRACE(1, "je alloc:\ncurrent: {}\nprevious: {}\ndelta with MM: {}\n",
      jeAllocated, m_prevAllocated, jeAllocated - m_prevAllocated);
    FTRACE(1, "je dealloc:\ncurrent: {}\nprevious: {}\ndelta with MM: {}\n",
      jeDeallocated, m_prevDeallocated, jeDeallocated - m_prevDeallocated);
    FTRACE(1, "usage: {}\ntotal (je) alloc: {}\nje debt: {}\n",
      stats.usage, stats.totalAlloc, stats.jemallocDebt);

    if (!contiguous_heap) {
      // Since these deltas potentially include memory allocated from another
      // thread but deallocated on this one, it is possible for these nubmers to
      // go negative.
      int64_t jeDeltaAllocated =
        int64_t(jeAllocated) - int64_t(jeDeallocated);
      int64_t mmDeltaAllocated =
        int64_t(m_prevAllocated) - int64_t(m_prevDeallocated);
      FTRACE(1, "je delta:\ncurrent: {}\nprevious: {}\n",
          jeDeltaAllocated, mmDeltaAllocated);

      // Subtract the old jemalloc adjustment (delta0) and add the current one
      // (delta) to arrive at the new combined usage number.
      stats.usage += jeDeltaAllocated - mmDeltaAllocated;
      // Remove the "debt" accrued from allocating the slabs so we don't double
      // count the slab-based allocations.
      stats.usage -= stats.jemallocDebt;
    }

    stats.jemallocDebt = 0;
    // We need to do the calculation instead of just setting it to jeAllocated
    // because of the MaskAlloc capability.
    stats.totalAlloc += jeMMDeltaAllocated;
    if (live) {
      m_prevAllocated = jeAllocated;
      m_prevDeallocated = jeDeallocated;
    }

    FTRACE(1, "After stats sync:\n");
    FTRACE(1, "usage: {}\ntotal (je) alloc: {}\n\n",
      stats.usage, stats.totalAlloc);
}
#endif
  if (stats.usage > stats.peakUsage) {
    // NOTE: the peak memory usage monotonically increases, so there cannot
    // be a second OOM exception in one request.
    assert(stats.maxBytes > 0);
    if (live && m_couldOOM && stats.usage > stats.maxBytes) {
      refreshStatsHelperExceeded();
    }
    // Check whether the process's active memory limit has been exceeded, and
    // if so, stop the server.
    //
    // Only check whether the total memory limit was exceeded if this request
    // is at a new high water mark.  This check could be performed regardless
    // of this request's current memory usage (because other request threads
    // could be to blame for the increased memory usage), but doing so would
    // measurably increase computation for little benefit.
#ifdef USE_JEMALLOC
    // (*m_cactive) consistency is achieved via atomic operations.  The fact
    // that we do not use an atomic operation here means that we could get a
    // stale read, but in practice that poses no problems for how we are
    // using the value.
    if (live && s_statsEnabled && *m_cactive > m_cactiveLimit) {
      refreshStatsHelperStop();
    }
#endif
    stats.peakUsage = stats.usage;
  }
  if (live && m_statsIntervalActive) {
    if (stats.usage > stats.peakIntervalUsage) {
      stats.peakIntervalUsage = stats.usage;
    }
    if (stats.alloc > stats.peakIntervalAlloc) {
      stats.peakIntervalAlloc = stats.alloc;
    }
  }
}

template void MemoryManager::refreshStatsImpl<true>(MemoryUsageStats& stats);
template void MemoryManager::refreshStatsImpl<false>(MemoryUsageStats& stats);

void MemoryManager::sweep() {
  assert(!sweeping());
  if (debug) checkHeap();
  TRACE(2, "MemoryManager::sweep() calling collect()\n");
  collect();
  m_sweeping = true;
  SCOPE_EXIT { m_sweeping = false; };
  DEBUG_ONLY size_t num_sweepables = 0, num_natives = 0;

  // iterate until both sweep lists are empty. Entries can be added or
  // removed from either list during sweeping.
  do {
    while (!m_sweepables.empty()) {
      num_sweepables++;
      auto obj = m_sweepables.next();
      obj->unregister();
      obj->sweep();
    }
    while (!m_natives.empty()) {
      num_natives++;
      assert(m_natives.back()->sweep_index == m_natives.size() - 1);
      auto node = m_natives.back();
      m_natives.pop_back();
      auto obj = Native::obj(node);
      auto ndi = obj->getVMClass()->getNativeDataInfo();
      ndi->sweep(obj);
      // trash the native data but leave the header and object parsable
      assert(memset(node+1, kSmallFreeFill, node->obj_offset - sizeof(*node)));
    }
  } while (!m_sweepables.empty());

  DEBUG_ONLY auto napcs = m_apc_arrays.size();
  FTRACE(1, "sweep: sweepable {} native {} apc array {}\n",
         num_sweepables,
         num_natives,
         napcs);
  if (debug) checkHeap();

  // decref apc arrays referenced by this request.  This must happen here
  // (instead of in resetAllocator), because the sweep routine may use
  // g_context.
  while (!m_apc_arrays.empty()) {
    auto a = m_apc_arrays.back();
    m_apc_arrays.pop_back();
    a->sweep();
  }
}

void MemoryManager::resetAllocator() {
  assert(m_natives.empty() && m_sweepables.empty());
  // decref apc strings referenced by this request
  DEBUG_ONLY auto nstrings = StringData::sweepAll();

  // cleanup root maps
  dropRootMaps();

  // free the heap
  m_heap.reset();

  m_lineCursor = m_lineLimit = nullptr;

  resetStatsImpl(true);
  FTRACE(1, "reset: strings {}\n", nstrings);
}

void MemoryManager::flush() {
  always_assert(empty());
  m_heap.flush();
  m_apc_arrays = std::vector<APCLocalArray*>();
  m_natives = std::vector<NativeNode*>();
  m_exceptionRoots = std::vector<ExtendedException*>();
}

/*
 * req::malloc & friends implementation notes
 *
 * There are three kinds of allocations:
 *
 *  a) Big allocations.  (size >= kMaxSmallSize)
 *
 *     In this case we behave as a wrapper around the normal libc
 *     malloc/free.  We insert a BigNode header at the front of the
 *     allocation in order to find these at sweep time (end of
 *     request) so we can give them back to libc.
 *
 *  b) Size-tracked small allocations.
 *
 *     This is used for the generic case, for callers who can't tell
 *     us the size of the allocation at free time.
 *
 *     In this situation, we put a SmallNode header at the front of
 *     the block that tells us the size for when we need to free it
 *     later.  We differentiate this from a BigNode using the size
 *     field in either structure (they overlap at the same address).
 *
 *  c) Size-untracked small allocation
 *
 *     Many callers have an easy time telling you how big the object
 *     was when they need to free it.  In this case we can avoid the
 *     SmallNode, which saves us some memory and also let's us give
 *     out 16-byte aligned pointers easily.
 *
 *     We know when we have one of these because it has to be freed
 *     through a different entry point.  (E.g. MM().freeSmallSize() or
 *     MM().freeBigSize().)
 *
 * When small blocks are freed (case b and c), they're placed in the
 * appropriate size-segregated freelist.  Large blocks are immediately
 * passed back to libc via free.
 *
 * There are currently two kinds of freelist entries: entries where
 * there is already a valid SmallNode on the list (case b), and
 * entries where there isn't (case c).  The reason for this is that
 * that way, when allocating for case b, you don't need to store the
 * SmallNode size again.  Much of the heap is going through case b at
 * the time of this writing, so it is a measurable regression to try
 * to just combine the free lists, but presumably we can move more to
 * case c and combine the lists eventually.
 */

inline void* MemoryManager::malloc(size_t nbytes) {
  auto const nbytes_padded = nbytes + sizeof(SmallNode);
  if (LIKELY(nbytes_padded) <= kMaxMediumSize) {
    auto const ptr = static_cast<SmallNode*>(mallocSmallSize(nbytes_padded));
    ptr->padbytes = nbytes_padded;
    ptr->hdr.kind = HeaderKind::SmallMalloc;
    return ptr + 1;
  }
  return mallocBig(nbytes);
}

union MallocNode {
  BigNode big;
  SmallNode small;
};

static_assert(sizeof(SmallNode) == sizeof(BigNode), "");

inline void MemoryManager::free(void* ptr) {
  assert(ptr != 0);
  auto const n = static_cast<MallocNode*>(ptr) - 1;
  auto const padbytes = n->small.padbytes;
  if (LIKELY(padbytes <= kMaxMediumSize)) {
    return freeSmallSize(&n->small, n->small.padbytes);
  }
  m_heap.freeBig(ptr);
}

inline void* MemoryManager::realloc(void* ptr, size_t nbytes) {
  FTRACE(3, "MemoryManager::realloc: {} to {}\n", ptr, nbytes);
  assert(nbytes > 0);
  auto const n = static_cast<MallocNode*>(ptr) - 1;
  if (LIKELY(n->small.padbytes <= kMaxMediumSize)) {
    void* newmem = req::malloc(nbytes);
    auto const copySize = std::min(
      n->small.padbytes - sizeof(SmallNode),
      nbytes
    );
    newmem = memcpy(newmem, ptr, copySize);
    req::free(ptr);
    return newmem;
  }
  // Ok, it's a big allocation.
  if (debug) eagerGCCheck();
  auto block = m_heap.resizeBig(ptr, nbytes);
  refreshStats();
  return block.ptr;
}

const char* header_names[] = {
  "Packed", "Struct", "Mixed", "Empty", "Apc", "Globals", "Proxy",
  "String", "Resource", "Ref",
  "Object", "ResumableObj", "AwaitAllWH",
  "Vector", "Map", "Set", "Pair", "ImmVector", "ImmMap", "ImmSet",
  "Resumable", "Native", "SmallMalloc", "BigMalloc", "BigObj",
  "Free", "Hole"
};
static_assert(sizeof(header_names)/sizeof(*header_names) == NumHeaderKinds, "");

// initialize a Hole header in the unused memory between m_front and m_limit
void MemoryManager::initHole(void* ptr, uint32_t size) {
  if (size == 0) return;
  assert(size >= sizeof(FreeNode));
  if (debug) memset(ptr, kSmallFreeFill, size);
  auto hdr = static_cast<FreeNode*>(ptr);
  hdr->hdr.kind = HeaderKind::Hole;
  hdr->size() = size;
}

// test iterating objects in slabs
void MemoryManager::checkHeap() {

  // initialise a freenode at the end of the current block
  // TODO could be redundant
  if (m_lineCursor) {
    uint32_t space = uintptr_t(m_lineLimit) - uintptr_t(m_lineCursor);
    assert(space == 0 || space >= sizeof(FreeNode));
    initHole(m_lineCursor, space);
  }

  size_t markedLines = 0, totalLines = 0;
  std::unordered_set<FreeNode*> free_blocks;
  std::unordered_set<APCLocalArray*> apc_arrays;
  std::unordered_set<StringData*> apc_strings;

  forEachLine([&](void* p, uint8_t markByte) {
    ++totalLines;
    if (markByte > 0 ) ++markedLines;
    TRACE(3, "checkHeap: line %p\n markByte(%d)\n", p, markByte);
  });

  forEachHeader([&](Header* h) {
    switch (h->kind()) {
      case HeaderKind::Apc:
        apc_arrays.insert(&h->apc_);
        break;
      case HeaderKind::String:
        if (h->str_.isShared()) apc_strings.insert(&h->str_);
        break;
      case HeaderKind::Free:
      case HeaderKind::Packed:
      case HeaderKind::Struct:
      case HeaderKind::Mixed:
      case HeaderKind::Empty:
      case HeaderKind::Globals:
      case HeaderKind::Proxy:
      case HeaderKind::Object:
      case HeaderKind::ResumableObj:
      case HeaderKind::AwaitAllWH:
      case HeaderKind::Vector:
      case HeaderKind::Map:
      case HeaderKind::Set:
      case HeaderKind::Pair:
      case HeaderKind::ImmVector:
      case HeaderKind::ImmMap:
      case HeaderKind::ImmSet:
      case HeaderKind::Resource:
      case HeaderKind::Ref:
      case HeaderKind::ResumableFrame:
      case HeaderKind::NativeData:
      case HeaderKind::SmallMalloc:
      case HeaderKind::BigMalloc:
        break;
      case HeaderKind::BigObj:
      case HeaderKind::Hole:
        assert(false && "forEachHeader skips these kinds");
        break;
    }
  });

  // check the apc array list
  for (auto a : m_apc_arrays) {
    assert(apc_arrays.find(a) != apc_arrays.end());
    apc_arrays.erase(a);
  }
  assert(apc_arrays.empty());

  // check the apc string list
  for (StringDataNode *next, *n = m_strings.next; n != &m_strings; n = next) {
    next = n->next;
    auto const s = StringData::node2str(n);
    assert(s->isShared());
    assert(apc_strings.find(s) != apc_strings.end());
    apc_strings.erase(s);
  }
  assert(apc_strings.empty());

  TRACE(1, "checkHeap: %lu totalLines %lu markedLines\n", totalLines,
    markedLines);
}

void* MemoryManager::sequentialAllocate(void*& cursor, void* limit, 
                                        uint32_t bytes) {
  if (!cursor) {
    TRACE(3, "sequentialAllocate called with nullptr cursor for %d bytes\n",
      bytes);
    return nullptr;
  }
  assert((uintptr_t(cursor) & kSmallSizeAlignMask) == 0);
  assert((uintptr_t(limit) & kSmallSizeAlignMask) == 0);
  assert(uintptr_t(cursor) <= uintptr_t(limit));

  auto space = uintptr_t(limit) - uintptr_t(cursor);
  assert(space <= kBlockSize);
  if (bytes <= space) {
    // round to 16-byte alignment
    auto aligned_bytes = MemoryManager::align(bytes);

    void* p = cursor;
    cursor = (void*)(uintptr_t(cursor) + aligned_bytes);

    assert(uintptr_t(p) + aligned_bytes == uintptr_t(cursor));
    // the other assertions imply p is aligned here
    TRACE(3, "sequentialAllocate fit %d bytes in %lu space\n", aligned_bytes,
      space);
    return p;
  } else if (space != 0) {
    assert(space >= 16); // otherwise we can't fit a freenode header
    initHole(cursor, space);
  }
  
  TRACE(2, "sequentialAllocate could not fit %d bytes in %lu space\n", bytes,
    space);
  return nullptr;
}

/*
 * Find the next run of free lines in the current block and return and set the
 * cursor, or if no such run exists, return and set cursor to nullptr.
 */
void* MemoryManager::getNextLineInBlock() {
  ImmixBlock b = m_heap.currentBlock();
  if (b.ptr == nullptr) {
    m_lineCursor = nullptr;
    return nullptr;
  }
  // We don't ever backtrack into the current block, so we start at our current 
  // limit
  auto start = (uintptr_t(m_lineLimit) - uintptr_t(b.ptr)) / kLineSize;
  TRACE(2, "getNextLineInBlock attempted, start(%lu)\n", start);
  return getFreeLines(b, start, m_lineCursor, m_lineLimit);
}

void* MemoryManager::getFreeLines(const ImmixBlock& block, uint32_t start,
                                  void*& cursor, void*& limit) {
  uint32_t linesInBlock = block.size / kLineSize;
  assert(start <= linesInBlock);

  // loop until 1 before the last line as the last line is implicitly marked
  // if the line before it is marked.
  for (uint32_t line = start; line < linesInBlock - 1; line++) {
    if (block.lineMap[line] == 0) {
      if (line == start || block.lineMap[line - 1] == 0) {
        cursor = (void*)(uintptr_t(block.ptr) + line * kLineSize);

        // continue and find the end of this run of free lines
        for (uint32_t endLine = line + 1; endLine < linesInBlock; endLine++) {
          if (block.lineMap[endLine] != 0) {
            limit = (void*)(uintptr_t(block.ptr) + endLine * kLineSize);

            TRACE(3, "getFreeLines cursor %p limit %p\n", cursor, limit);
            return cursor;
          }
        }
        // reached the end of the block without finding a marked line
        limit = (void*)(uintptr_t(block.ptr) + block.size);

        TRACE(3, "getFreeLines cursor %p limit(end) %p\n", cursor, limit);
        return cursor;
      }
    }
  }
  cursor = nullptr;
  return nullptr;
}

/* 
 * Request the next block from the heap and move cursor to the next run of
 * free lines within the block, if any exist. Return nullptr and set cursor to 
 * nullptr if we have exhausted all recyclable blocks.
 */
void* MemoryManager::getNextRecyclableBlock() {
  while (true) {
    auto block = m_heap.getNextRecyclableBlock();
    assert(block.overflow == 0);
    if (block.ptr == nullptr) {
      TRACE(2, "getNextRecyclableBlock returned null\n");
      m_lineCursor = nullptr;
      return nullptr;
    }
    void* ret = getFreeLines(block, /*start=*/0, m_lineCursor, m_lineLimit);
    if (ret) {
      TRACE(2, "getNextRecyclableBlock returned %p size(%lu)\n", m_lineCursor,
        uintptr_t(m_lineLimit) - uintptr_t(m_lineCursor));
      return ret;
    } 
  }
}

void* MemoryManager::getFreeBlock(void*& cursor, void*& limit,
                                  bool forOverflow) {
  if (debug && RuntimeOption::EvalCheckHeapOnAlloc) checkHeap();
  auto block = m_heap.allocSlab(kBlockSize, forOverflow);
  assert((uintptr_t(block.ptr) & kSmallSizeAlignMask) == 0);

  cursor = block.ptr;
  limit = (void*)(uintptr_t(block.ptr) + block.size);

  TRACE(2, "getFreeBlock %p size %lu\n", block.ptr, block.size);

  return block.ptr;
}

/* 
 * Finds the next line to allocate small objects into.
 * Will look in the current block first, then the next
 * recyclable block, then get a fresh block.
 */
void* MemoryManager::allocSlowHot(uint32_t bytes) {
  TRACE(3, "allocSlowHot bytes(%d)\n", bytes);
  getNextLineInBlock();
  if (m_lineCursor == nullptr) {
    getNextRecyclableBlock();
    if (m_lineCursor == nullptr) {
      auto block = getFreeBlock(m_lineCursor, m_lineLimit, false);
      if (block == nullptr) {
        return nullptr; //OOM
      }
    }
  }
  return mallocSmallSize(bytes);
}

/* 
 * pre: bytes is the size of a medium allocation
 */
void* MemoryManager::overflowAlloc(uint32_t bytes) {
  assert(bytes > kLineSize && bytes <= kMaxMediumSize);

  void* p = sequentialAllocate(m_blockCursor, m_blockLimit, bytes);
  if (p != nullptr) {
    // TODO refactor
    uint32_t space = uintptr_t(m_blockLimit) - uintptr_t(m_blockCursor);
    assert(space == 0 || space >= sizeof(FreeNode));
    initHole(m_blockCursor, space);
    FTRACE(2, "overflowAlloc into block: {} -> {}\n", bytes, p);
    return p;
  }
  // get a fresh block and allocate into it instead
  auto block = getFreeBlock(m_blockCursor, m_blockLimit, true);
  if (block == nullptr) {
    FTRACE(1, "overflowAlloc OOM: {}\n", bytes);
    return nullptr; //OOM
  }
  auto ret = sequentialAllocate(m_blockCursor, m_blockLimit, bytes);
  // TODO refactor
  uint32_t space = uintptr_t(m_blockLimit) - uintptr_t(m_blockCursor);
  assert(space == 0 || space >= sizeof(FreeNode));
  initHole(m_blockCursor, space);
  FTRACE(2, "overflowAlloc into *new* block: {} -> {}\n", bytes, ret);
  return ret;
}

inline void MemoryManager::updateBigStats() {
  // If we are using jemalloc, it is keeping track of allocations outside of
  // the slabs and the usage so we should force this after an allocation that
  // was too large for one of the existing slabs. When we're not using jemalloc
  // this check won't do anything so avoid the extra overhead.
  if (use_jemalloc || UNLIKELY(m_stats.usage > m_stats.maxBytes)) {
    refreshStats();
  }
}

NEVER_INLINE
void* MemoryManager::mallocBig(size_t nbytes) {
  assert(nbytes > 0);
  auto block = m_heap.allocBig(nbytes, HeaderKind::BigMalloc);
  updateBigStats();
  return block.ptr;
}

template NEVER_INLINE
MemBlock MemoryManager::mallocBigSize<true>(size_t);
template NEVER_INLINE
MemBlock MemoryManager::mallocBigSize<false>(size_t);

template<bool callerSavesActualSize> NEVER_INLINE
MemBlock MemoryManager::mallocBigSize(size_t bytes) {
  if (debug) eagerGCCheck();

  auto block = m_heap.allocBig(bytes, HeaderKind::BigObj);
  auto szOut = block.size;
#ifdef USE_JEMALLOC
  // NB: We don't report the SweepNode size in the stats.
  auto const delta = callerSavesActualSize ? szOut : bytes;
  m_stats.usage += int64_t(delta);
  // Adjust jemalloc otherwise we'll double count the direct allocation.
  m_stats.borrow(delta);
#else
  m_stats.usage += bytes;
#endif
  updateBigStats();
  auto ptrOut = block.ptr;
  FTRACE(3, "mallocBigSize: {} ({} requested, {} usable)\n",
         ptrOut, bytes, szOut);
  return {ptrOut, szOut};
}

NEVER_INLINE
void* MemoryManager::callocBig(size_t totalbytes) {
  if (debug) eagerGCCheck();
  assert(totalbytes > 0);
  auto block = m_heap.callocBig(totalbytes);
  updateBigStats();
  return block.ptr;
}

// req::malloc api entry points, with support for malloc/free corner cases.
namespace req {
void* malloc(size_t nbytes) {
  auto const size = std::max(nbytes, size_t(1));
  return MM().malloc(size);
}

void* calloc(size_t count, size_t nbytes) {
  auto const totalBytes = std::max<size_t>(count * nbytes, 1);
  if (totalBytes <= kMaxMediumSize) {
    return memset(req::malloc(totalBytes), 0, totalBytes);
  }
  return MM().callocBig(totalBytes);
}

void* realloc(void* ptr, size_t nbytes) {
  if (!ptr) return req::malloc(nbytes);
  if (!nbytes) {
    req::free(ptr);
    return nullptr;
  }
  return MM().realloc(ptr, nbytes);
}

void free(void* ptr) {
  if (ptr) MM().free(ptr);
}
} // namespace req

//////////////////////////////////////////////////////////////////////

void MemoryManager::addNativeObject(NativeNode* node) {
  if (debug) for (DEBUG_ONLY auto n : m_natives) assert(n != node);
  node->sweep_index = m_natives.size();
  m_natives.push_back(node);
}

void MemoryManager::removeNativeObject(NativeNode* node) {
  assert(node->sweep_index < m_natives.size());
  assert(m_natives[node->sweep_index] == node);
  auto index = node->sweep_index;
  auto last = m_natives.back();
  m_natives[index] = last;
  m_natives.pop_back();
  last->sweep_index = index;
}

void MemoryManager::addApcArray(APCLocalArray* a) {
  a->m_sweep_index = m_apc_arrays.size();
  m_apc_arrays.push_back(a);
}

void MemoryManager::removeApcArray(APCLocalArray* a) {
  assert(a->m_sweep_index < m_apc_arrays.size());
  assert(m_apc_arrays[a->m_sweep_index] == a);
  auto index = a->m_sweep_index;
  auto last = m_apc_arrays.back();
  m_apc_arrays[index] = last;
  m_apc_arrays.pop_back();
  last->m_sweep_index = index;
}

void MemoryManager::addSweepable(Sweepable* obj) {
  obj->enlist(&m_sweepables);
}

// defined here because memory-manager.h includes sweepable.h
Sweepable::Sweepable() {
  MM().addSweepable(this);
}

//////////////////////////////////////////////////////////////////////

void MemoryManager::logAllocation(void* p, size_t bytes) {
  MemoryProfile::logAllocation(p, bytes);
}

void MemoryManager::logDeallocation(void* p) {
  MemoryProfile::logDeallocation(p);
}

void MemoryManager::resetCouldOOM(bool state) {
  clearSurpriseFlag(MemExceededFlag);
  m_couldOOM = state;
}


///////////////////////////////////////////////////////////////////////////////
// Request profiling.

bool MemoryManager::triggerProfiling(const std::string& filename) {
  auto trigger = new ReqProfContext();
  trigger->flag = true;
  trigger->filename = filename;

  ReqProfContext* expected = nullptr;

  if (!s_trigger.compare_exchange_strong(expected, trigger)) {
    delete trigger;
    return false;
  }
  return true;
}

void MemoryManager::requestInit() {
  auto trigger = s_trigger.exchange(nullptr);

  // If the trigger has already been claimed, do nothing.
  if (trigger == nullptr) return;

  always_assert(MM().empty());

  // Initialize the request-local context from the trigger.
  auto& profctx = MM().m_profctx;
  assert(!profctx.flag);

  MM().m_bypassSlabAlloc = true;
  profctx = *trigger;
  delete trigger;

#ifdef USE_JEMALLOC
  bool active = true;
  size_t boolsz = sizeof(bool);

  // Reset jemalloc stats.
  if (mallctl("prof.reset", nullptr, nullptr, nullptr, 0)) {
    return;
  }

  // Enable jemalloc thread-local heap dumps.
  if (mallctl("prof.active",
              &profctx.prof_active, &boolsz,
              &active, sizeof(bool))) {
    profctx = ReqProfContext{};
    return;
  }
  if (mallctl("thread.prof.active",
              &profctx.thread_prof_active, &boolsz,
              &active, sizeof(bool))) {
    mallctl("prof.active", nullptr, nullptr,
            &profctx.prof_active, sizeof(bool));
    profctx = ReqProfContext{};
    return;
  }
#endif
}

void MemoryManager::requestShutdown() {
  auto& profctx = MM().m_profctx;

  if (!profctx.flag) return;

#ifdef USE_JEMALLOC
  jemalloc_pprof_dump(profctx.filename, true);

  mallctl("thread.prof.active", nullptr, nullptr,
          &profctx.thread_prof_active, sizeof(bool));
  mallctl("prof.active", nullptr, nullptr,
          &profctx.prof_active, sizeof(bool));
#endif

  MM().m_bypassSlabAlloc = RuntimeOption::DisableSmallAllocator;
  profctx = ReqProfContext{};
}

void MemoryManager::eagerGCCheck() {
  if (RuntimeOption::EvalEagerGCProbability > 0 &&
      !g_context.isNull() &&
      folly::Random::oneIn(RuntimeOption::EvalEagerGCProbability)) {
    setSurpriseFlag(PendingGCFlag);
  }
}

///////////////////////////////////////////////////////////////////////////////

void BigHeap::reset() {
  TRACE(1, "BigHeap-reset: slabs %lu bigs %lu\n", m_slabs.size(),
        m_bigs.size());
  for (auto slab : m_slabs) {
    free(slab.ptr);
  }
  m_slabs.clear();
  m_pos = -1;
  for (auto n : m_bigs) {
    free(n);
  }
  m_bigs.clear();
}

void BigHeap::dump() {
  // TRACE(2, "BigHeap dump:\n\n");
  // auto charCounter = 0, blockCounter = 0;
  // MM().forEachHeader([&](Header* h) {
  //   auto size = MemoryManager::align(h->size());
  //   for (uintptr_t p = uintptr_t(h); p < uintptr_t(h) + size; p += 16) {
  //     if (charCounter == 64) {
  //       TRACE(2, "\n");
  //       charCounter = 0;
  //     }
  //     if (blockCounter == 2048) {
  //       TRACE(2, "\n");
  //       blockCounter = 0;
  //     }
  //     if (h->kind() == HeaderKind::Hole) {
  //       TRACE(2, "-");
  //     } else {
  //       TRACE(2, "x");
  //     }
  //     charCounter++;
  //     blockCounter++;
  //   }
  // });
}

void BigHeap::flush() {
  assert(empty());
  m_slabs = std::vector<ImmixBlock>{};
  m_bigs = std::vector<BigNode*>{};
}

MemBlock BigHeap::allocSlab(size_t size, bool forOverflow) {
  void* slab = safe_malloc(size);
  // probably not needed
  ImmixBlock b = ImmixBlock{slab, size};
  if (forOverflow) b.overflow = 1;
  m_slabs.push_back(b);
  if (!forOverflow) m_pos = m_slabs.size() - 1;
  return {slab, size};
}

ImmixBlock BigHeap::getNextRecyclableBlock() {
  if (!m_slabs.empty() && m_pos < m_slabs.size() - 1) {
    while (m_slabs[++m_pos].overflow != 0) {
      if (m_pos == m_slabs.size() - 1) return ImmixBlock{nullptr, 0};
    }
    assert(m_pos < m_slabs.size());
    assert(m_slabs[m_pos].overflow == 0);
    return m_slabs[m_pos];
  }
  return ImmixBlock{nullptr, 0};
}

void BigHeap::enlist(BigNode* n, HeaderKind kind, size_t size) {
  n->nbytes = size;
  n->hdr.kind = kind;
  n->index() = m_bigs.size();
  m_bigs.push_back(n);
}

MemBlock BigHeap::allocBig(size_t bytes, HeaderKind kind) {
#ifdef USE_JEMALLOC
  auto n = static_cast<BigNode*>(mallocx(bytes + sizeof(BigNode), 0));
  auto cap = sallocx(n, 0);
#else
  auto cap = bytes + sizeof(BigNode);
  auto n = static_cast<BigNode*>(safe_malloc(cap));
#endif
  enlist(n, kind, cap);
  return {n + 1, cap - sizeof(BigNode)};
}

MemBlock BigHeap::callocBig(size_t nbytes) {
  auto cap = nbytes + sizeof(BigNode);
  auto const n = static_cast<BigNode*>(safe_calloc(cap, 1));
  enlist(n, HeaderKind::BigMalloc, cap);
  return {n + 1, nbytes};
}

bool BigHeap::contains(void* ptr) const {
  auto const ptrInt = reinterpret_cast<uintptr_t>(ptr);
  auto it = std::find_if(std::begin(m_slabs), std::end(m_slabs),
    [&] (ImmixBlock slab) {
      auto const baseInt = reinterpret_cast<uintptr_t>(slab.ptr);
      return ptrInt >= baseInt && ptrInt < baseInt + slab.size;
    }
  );
  return it != std::end(m_slabs);
}

void BigHeap::markLineForSmall(const void* p) {
  auto const ptrInt = reinterpret_cast<uintptr_t>(p);
  auto it = std::find_if(std::begin(m_slabs), std::end(m_slabs),
    [&] (ImmixBlock slab) {
      auto const baseInt = reinterpret_cast<uintptr_t>(slab.ptr);
      return ptrInt >= baseInt && ptrInt < baseInt + slab.size;
    }
  );
  assert(it != std::end(m_slabs));
  auto lineNum = (ptrInt - uintptr_t(it->ptr)) / kLineSize;
  it->lineMap[lineNum] = 1; //mark the line
}

void BigHeap::markLinesForMedium(const void* p, uint32_t size) {
  assert(size > kLineSize);
  assert(size <= kMaxMediumSize);
  auto const ptrInt = reinterpret_cast<uintptr_t>(p);
  auto it = std::find_if(std::begin(m_slabs), std::end(m_slabs),
    [&] (ImmixBlock slab) {
      auto const baseInt = reinterpret_cast<uintptr_t>(slab.ptr);
      return ptrInt >= baseInt && ptrInt < baseInt + slab.size;
    }
  );
  assert(it != std::end(m_slabs));
  auto lineNum = (ptrInt - uintptr_t(it->ptr)) / kLineSize;
  auto firstLine = lineNum;
  auto lastLine = ((ptrInt + size) - uintptr_t(it->ptr)) / kLineSize;
  for(; lineNum <= lastLine; lineNum++) {
    it->lineMap[lineNum] = 1; //mark the line
  }
  FTRACE(2, "Marked {} lines for medium size={}\n",
    lastLine - firstLine + 1, size);
}

// p should point to start of a line
void BigHeap::markBlockContaining(const void* p) {
  auto const ptrInt = reinterpret_cast<uintptr_t>(p);
  auto it = std::find_if(std::begin(m_slabs), std::end(m_slabs),
    [&] (ImmixBlock slab) {
      auto const baseInt = reinterpret_cast<uintptr_t>(slab.ptr);
      return ptrInt >= baseInt && ptrInt < baseInt + slab.size;
    }
  );
  assert(it != std::end(m_slabs));
  it->marked = 1;
}

ImmixBlock BigHeap::currentBlock() {
  if (m_pos != -1) {
    assert(m_pos < m_slabs.size());
    return m_slabs[m_pos];
  }
  return ImmixBlock{nullptr, 0};
}

void BigHeap::resetBlockPointer() {
  m_pos = -1;
}

void BigHeap::freeUnusedBlocks() {
  m_slabs.erase(
    std::remove_if(
      m_slabs.begin(),
      m_slabs.end(),
      [](const ImmixBlock& block) {
        for (uint8_t lineMark : block.lineMap) {
          if (lineMark != 0) {
            return false;
          }
        }
        free(block.ptr);
        TRACE(2, "Freed block %p\n", block.ptr);
        return true;
      }
    ),
    m_slabs.end()
  );
}

NEVER_INLINE
void BigHeap::freeBig(void* ptr) {
  auto n = static_cast<BigNode*>(ptr) - 1;
  auto i = n->index();
  auto last = m_bigs.back();
  last->index() = i;
  m_bigs[i] = last;
  m_bigs.pop_back();
  free(n);
}

MemBlock BigHeap::resizeBig(void* ptr, size_t newsize) {
  // Since we don't know how big it is (i.e. how much data we should memcpy),
  // we have no choice but to ask malloc to realloc for us.
  auto const n = static_cast<BigNode*>(ptr) - 1;
  auto const newNode = static_cast<BigNode*>(
    safe_realloc(n, newsize + sizeof(BigNode))
  );
  if (newNode != n) {
    m_bigs[newNode->index()] = newNode;
  }
  return {newNode + 1, newsize};
}

/////////////////////////////////////////////////////////////////////////
//Contiguous Heap

void ContiguousHeap::reset() {
  m_requestCount++;

  // if there is a new peak, store it
  if (m_peak < m_used) {
    m_peak = m_used;
    // convert usage to MB.. used later for comparison with water marks
    m_heapUsage = ((uintptr_t)m_peak - (uintptr_t) m_base) >> 20;
  }

  // should me reset?
  bool resetHeap = false;

  // check if we are above low water mark
  if (m_heapUsage > RuntimeOption::HeapLowWaterMark) {
    // check if we are above below water mark
    if (m_heapUsage > RuntimeOption::HeapHighWaterMark) {
      // we are above high water mark... always reset
      resetHeap = true;
    } else {
      // if between watermarks, free based on request count and usage
      int requestCount = RuntimeOption::HeapResetCountBase;

      // Assumption : low and high water mark are power of 2 aligned
      for( auto resetStep = RuntimeOption::HeapHighWaterMark / 2 ;
           resetStep > m_heapUsage ;
           resetStep /= 2 ) {
        requestCount *= RuntimeOption::HeapResetCountMultiple;
      }
      if (requestCount <= m_requestCount) {
        resetHeap = true;
      }
    }
  }


  if (resetHeap) {
    auto oldPeak = m_peak;
    m_peak -= ((m_peak - m_base) / 2);
    m_peak = (char*)((uintptr_t)m_peak & ~(s_pageSize - 1));
    if (madvise(m_peak,
                (uintptr_t)oldPeak - (uintptr_t)m_peak,
                MADV_DONTNEED) == 0)
    {
      m_requestCount = 0;
      TRACE(1, "ContiguousHeap-reset: bytes %lu\n",
            (uintptr_t)m_end - (uintptr_t)m_peak);
    } else {
      TRACE(1,
          "ContiguousHeap-reset: madvise failed, trying again next request");
    }
  } else {
    TRACE(1, "ContiguousHeap-reset: nothing release");
  }
  m_used = m_base;
  m_freeList.next = nullptr;
  m_freeList.size() = 0;
  always_assert(m_base);
  m_slabs.clear();
  m_bigs.clear();
}

void ContiguousHeap::flush() {
  madvise(m_base, m_peak-m_base, MADV_DONTNEED);
  m_used = m_peak = m_base;
  m_freeList.size() = 0;
  m_freeList.next = nullptr;
  m_slabs = std::vector<ImmixBlock>{};
  m_bigs = std::vector<BigNode*>{};
}

MemBlock ContiguousHeap::allocSlab(size_t size) {
  size_t cap;
  void* slab = heapAlloc(size, cap);
  m_slabs.push_back({slab, cap});
  return {slab, cap};
}

MemBlock ContiguousHeap::allocBig(size_t bytes, HeaderKind kind) {
  size_t cap;
  auto n = static_cast<BigNode*>(heapAlloc(bytes + sizeof(BigNode), cap));
  enlist(n, kind, cap);
  return {n + 1, cap - sizeof(BigNode)};
}

MemBlock ContiguousHeap::callocBig(size_t nbytes) {
  size_t cap;
  auto const n = static_cast<BigNode*>(
        heapAlloc(nbytes + sizeof(BigNode), cap));
  memset(n, 0, cap);
  enlist(n, HeaderKind::BigMalloc, cap);
  return {n + 1, cap - sizeof(BigNode)};
}

bool ContiguousHeap::contains(void* ptr) const {
  auto const ptrInt = reinterpret_cast<uintptr_t>(ptr);
  return ptrInt >= reinterpret_cast<uintptr_t>(m_base) &&
         ptrInt <  reinterpret_cast<uintptr_t>(m_used);
}

NEVER_INLINE
void ContiguousHeap::freeBig(void* ptr) {
  // remove from big list
  auto n = static_cast<BigNode*>(ptr) - 1;
  auto i = n->index();
  auto last = m_bigs.back();
  auto size = n->nbytes;
  last->index() = i;
  m_bigs[i] = last;
  m_bigs.pop_back();

  // free heap space
  // freed nodes are stored in address ordered freelist
  auto free = &m_freeList;
  auto node = reinterpret_cast<FreeNode*>(n);
  node->size() = size;
  while (free->next != nullptr && free->next < node) {
    free = free->next;
  }
  // Coalesce Nodes if possible with adjacent free nodes
  if ((uintptr_t)free + free->size() + node->size() == (uintptr_t)free->next) {
    free->size() += node->size() + free->next->size();
    free->next = free->next->next;
  } else if ((uintptr_t)free + free->size() == (uintptr_t)ptr) {
    free->size() += node->size();
  } else if ((uintptr_t)node + node->size() == (uintptr_t)free->next){
    node->next = free->next->next;
    node->size() += free->next->size();
    free->next = node;
  } else {
    node->next = free->next;
    free->next = node;
  }
}

MemBlock ContiguousHeap::resizeBig(void* ptr, size_t newsize) {
  // Since we don't know how big it is (i.e. how much data we should memcpy),
  // we have no choice but to ask malloc to realloc for us.
  auto const n = static_cast<BigNode*>(ptr) - 1;
  size_t cap = 0;
  BigNode* newNode = nullptr;
  if (n->nbytes >= newsize + sizeof(BigNode)) {
    newNode = n;
  } else {
    newNode = static_cast<BigNode*>(
      heapAlloc(newsize + sizeof(BigNode),cap)
    );
    memcpy(newNode, ptr, n->nbytes);
    newNode->nbytes = cap;
  }
  if (newNode != n) {
    m_bigs[newNode->index()] = newNode;
    freeBig(n);
  }
  return {newNode + 1, n->nbytes - sizeof(BigNode)};
}

void* ContiguousHeap::heapAlloc(size_t nbytes, size_t &cap) {
  if (UNLIKELY(!m_base)) {
    // Lazy allocation of heap
    createRequestHeap();
  }

  void* ptr = nullptr;
  auto alignedSize = (nbytes + s_pageSize - 1) & ~(s_pageSize - 1);

  // freeList is address ordered first fit
  auto prev = &m_freeList;
  auto cur = m_freeList.next;
  while (cur != nullptr ) {
    if (cur->size() >= alignedSize &&
        cur->size() < alignedSize + kMaxMediumSize) {
      // found freed heap node that fits allocation and doesn't need to split
      ptr = cur;
      prev->next = cur->next;
      cap = cur->size();
      return ptr;
    }
    if (cur->size() > alignedSize) {
      // split free heap node
      prev->next = reinterpret_cast<FreeNode*>(((char*)cur) + alignedSize);
      prev->next->next = cur->next;
      prev->next->size() = cur->size() - alignedSize;
      ptr = cur;
      cap = alignedSize;
      return ptr;
    }
    prev = cur;
    cur = cur->next;
  }
  ptr = (void*)m_used;
  m_used += alignedSize;
  cap = alignedSize;
  if (UNLIKELY(m_used > m_end)) {
    always_assert_flog(0,
        "Heap address space exhausted\nbase:{}\nend:{}\nused{}",
        m_base, m_end, m_used);
    // Throw exception when t4840214 is fixed
    // throw FatalErrorException("Request heap out of memory");
  } else if (UNLIKELY(m_used > m_OOMMarker)) {
    setSurpriseFlag(MemExceededFlag);
  }
  return ptr;
}

void ContiguousHeap::createRequestHeap() {
  // convert to bytes
  m_contiguousHeapSize = RuntimeOption::HeapSizeMB * 1024 * 1024;

  if (( m_base = (char*)mmap(NULL,
                            m_contiguousHeapSize,
                            PROT_WRITE | PROT_READ,
                            s_mmapFlags,
                            -1,
                            0)) != MAP_FAILED) {
    m_used = m_base;
  } else {
    always_assert_flog(0, "Heap Creation Failed");
  }
  m_end = m_base + m_contiguousHeapSize;
  m_peak = m_base;
  m_OOMMarker = m_end - (m_contiguousHeapSize/2);
  m_freeList.next = nullptr;
  m_freeList.size() = 0;
}

ContiguousHeap::~ContiguousHeap(){
  flush();
}
}
