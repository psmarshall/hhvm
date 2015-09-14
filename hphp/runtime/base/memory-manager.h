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

#ifndef incl_HPHP_MEMORY_MANAGER_H_
#define incl_HPHP_MEMORY_MANAGER_H_

#include <array>
#include <vector>
#include <utility>
#include <set>
#include <unordered_map>
#include <bitset>

#include <folly/Memory.h>

#include "hphp/util/alloc.h" // must be included before USE_JEMALLOC is used
#include "hphp/util/compilation-flags.h"
#include "hphp/util/trace.h"
#include "hphp/util/thread-local.h"

#include "hphp/runtime/base/memory-usage-stats.h"
#include "hphp/runtime/base/request-event-handler.h"
#include "hphp/runtime/base/runtime-option.h"
#include "hphp/runtime/base/sweepable.h"
#include "hphp/runtime/base/header-kind.h"
#include "hphp/runtime/base/req-ptr.h"

// used for mmapping contiguous heap space
// If used, anonymous pages are not cleared when mapped with mmap. It is not
// enabled by default and should be checked before use
#define       MAP_UNINITIALIZED 0x4000000 /* XXX Fragile. */

namespace HPHP {
struct APCLocalArray;
struct MemoryManager;
struct ObjectData;
struct ResourceData;
struct ExtendedException;
struct Header;

//////////////////////////////////////////////////////////////////////

/*
 * Request local memory in HHVM is managed by a thread local object
 * called MemoryManager.
 *
 * The object may be accessed with MM(), but higher-level apis are
 * also provided below.
 *
 * The MemoryManager serves the following funcitons in hhvm:
 *
 *   - Managing request-local memory.
 *
 *   - Tracking "sweepable" objects--i.e. objects that must run custom
 *     cleanup code if they are still live at the end of the request.
 *
 *   - Accounting for usage of memory "by this request", whether it
 *     goes through the request-local allocator, or the underlying
 *     malloc implementation.  (This feature is gated on being
 *     compiled with jemalloc.)
 */
MemoryManager& MM();

//////////////////////////////////////////////////////////////////////

/*
 * req::malloc api for request-scoped memory
 *
 * This is the most generic entry point to the request local
 * allocator.  If you easily know the size of the allocation at free
 * time, it might be more efficient to use MM() apis directly.
 *
 * These functions behave like C's malloc/free, but get memory from
 * the current thread's MemoryManager instance.  At request-end, any
 * un-freed memory is explicitly freed (and in debug, garbage filled).
 * If any pointers to this memory survive beyond a request, they'll be
 * dangling pointers.
 *
 * These functions only guarantee 8-byte alignment for the returned
 * pointer.
 */

namespace req {

void* malloc(size_t nbytes);
void* calloc(size_t count, size_t bytes);
void* realloc(void* ptr, size_t nbytes);
void  free(void* ptr);

/*
 * request-heap (de)allocators for non-POD C++-style stuff. Runs constructors
 * and destructors.
 *
 * Unlike the normal operator delete, req::destroy_raw() requires ~T() must
 * be nothrow and that p is not null.
 */
template<class T, class... Args> T* make_raw(Args&&...);
template<class T> void destroy_raw(T* p);

/*
 * Allocate an array of objects.  Similar to req::malloc, but with
 * support for constructors.
 *
 * Note that explicitly calling req::destroy_raw will run the destructors,
 * but if you let the allocator sweep it the destructors will not be
 * called.
 *
 * Unlike the normal operator delete, req::destroy_raw_array requires
 * ~T() must be nothrow.
 */
template<class T> T* make_raw_array(size_t count);
template<class T> void destroy_raw_array(T* t, size_t count);

//////////////////////////////////////////////////////////////////////

// STL-style allocator for the request-heap allocator.  (Unfortunately we
// can't use allocator_traits yet.)
//
// You can also use req::Allocator as a model of folly's
// SimpleAllocator where appropriate.
//

template <class T>
struct Allocator {
  typedef T              value_type;
  typedef T*             pointer;
  typedef const T*       const_pointer;
  typedef T&             reference;
  typedef const T&       const_reference;
  typedef std::size_t    size_type;
  typedef std::ptrdiff_t difference_type;

  template <class U>
  struct rebind {
    typedef Allocator<U> other;
  };

  pointer address(reference value) const {
    return &value;
  }
  const_pointer address(const_reference value) const {
    return &value;
  }

  Allocator() noexcept {}
  Allocator(const Allocator&) noexcept {}
  template<class U> Allocator(const Allocator<U>&) noexcept {}
  ~Allocator() noexcept {}

  size_type max_size() const {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

  pointer allocate(size_type num, const void* = 0) {
    pointer ret = (pointer)req::malloc(num * sizeof(T));
    return ret;
  }

  template<class U, class... Args>
  void construct(U* p, Args&&... args) {
    ::new ((void*)p) U(std::forward<Args>(args)...);
  }

  void destroy(pointer p) {
    p->~T();
  }

  void deallocate(pointer p, size_type num) {
    req::free(p);
  }

  template<class U> bool operator==(const Allocator<U>&) const {
    return true;
  }

  template<class U> bool operator!=(const Allocator<U>&) const {
    return false;
  }
};

}

//////////////////////////////////////////////////////////////////////

/*
 * Slabs are consumed via bump allocation.  The individual allocations are
 * quantized into a fixed set of size classes, the sizes of which are an
 * implementation detail documented here to shed light on the algorithms that
 * compute size classes.  Request sizes are rounded up to the nearest size in
 * the relevant SMALL_SIZES table; e.g. 17 is rounded up to 32.  There are
 * 4 size classes for each doubling of size
 * (ignoring the alignment-constrained smallest size classes), which limits
 * internal fragmentation to 20%.
 *
 * SMALL_SIZES: Complete table of SMALL_SIZE(index, lg_grp, lg_delta, ndelta,
 *              lg_delta_lookup, ncontig) tuples.
 *   index: Size class index.
 *   lg_grp: Lg group base size (no deltas added).
 *   lg_delta: Lg delta to previous size class.
 *   ndelta: Delta multiplier.  size == 1<<lg_grp + ndelta<<lg_delta
 *   lg_delta_lookup: Same as lg_delta if a lookup table size class, 'no'
 *                    otherwise.
 *   ncontig: Number of contiguous regions to batch allocate in the slow path
 *            due to the corresponding free list being empty.  Must be greater
 *            than zero, and small enough that the contiguous regions fit within
 *            one slab.
 */

constexpr uint32_t kMaxSmallSizeLookup = 4096;

constexpr unsigned kLgSlabSize = 21;
constexpr uint32_t kSlabSize = uint32_t{1} << kLgSlabSize;
constexpr unsigned kLgSmallSizeQuantum = 4;
constexpr uint32_t kSmallSizeAlign = 1u << kLgSmallSizeQuantum;
constexpr uint32_t kSmallSizeAlignMask = kSmallSizeAlign - 1;

constexpr unsigned kLgSizeClassesPerDoubling = 2;

constexpr unsigned kLineSize = 128;
constexpr unsigned kBlockSize = 32768;
constexpr unsigned kMaxMediumSize = 8192; //8kB is max medium object (1/4 block)
constexpr unsigned kLinesPerBlock = kBlockSize / kLineSize;

constexpr unsigned kSmallPreallocCountLimit = 8;
constexpr uint32_t kSmallPreallocBytesLimit = uint32_t{1} << 9;

/*
 * Constants for the various debug junk-filling of different types of
 * memory in hhvm.
 *
 * jemalloc uses 0x5a to fill freed memory, so we use 0x6a for the
 * request-local allocator so it is easy to tell the difference when
 * debugging.  There's also 0x7a for junk-filling some cases of
 * ex-TypedValue memory (evaluation stack).
 */
constexpr char kSmallFreeFill   = 0x6a;
constexpr char kTVTrashFill     = 0x7a; // used by interpreter
constexpr char kTVTrashFill2    = 0x7b; // used by req::ptr dtors
constexpr char kTVTrashJITStk   = 0x7c; // used by the JIT for stack slots
constexpr char kTVTrashJITFrame = 0x7d; // used by the JIT for stack frames
constexpr char kTVTrashJITHeap  = 0x7e; // used by the JIT for heap
constexpr uintptr_t kSmallFreeWord = 0x6a6a6a6a6a6a6a6aLL;
constexpr uintptr_t kMallocFreeWord = 0x5a5a5a5a5a5a5a5aLL;

//////////////////////////////////////////////////////////////////////

// Header MemoryManager uses for StringDatas that wrap APCHandle
struct StringDataNode {
  StringDataNode* next;
  StringDataNode* prev;
};

// This is the header MemoryManager uses to remember large allocations
// so they can be auto-freed in MemoryManager::reset()
struct BigNode {
  size_t nbytes;
  HeaderWord<> hdr;
  uint32_t& index() { return hdr.hi32; }
};

// Header used for small req::malloc allocations (but not *Size allocs)
struct SmallNode {
  size_t padbytes;
  HeaderWord<> hdr;
};

// all FreeList entries are parsed by inspecting this header.
struct FreeNode {
  FreeNode* next;
  HeaderWord<> hdr;
  uint32_t& size() { return hdr.hi32; }
  uint32_t size() const { return hdr.hi32; }
};

// header for HNI objects with NativeData payloads. see native-data.h
// for details about memory layout.
struct NativeNode {
  uint32_t sweep_index; // index in MM::m_natives
  uint32_t obj_offset; // byte offset from this to ObjectData*
  HeaderWord<> hdr;
};

// header for Resumable objects. See layout comment in resumable.h
struct ResumableNode {
  size_t framesize;
  HeaderWord<> hdr;
};

// POD type for tracking arbitrary memory ranges
struct MemBlock {
  void* ptr;
  size_t size; // bytes
};

typedef std::bitset<kBlockSize / 16> live_map;

struct ImmixBlock {
  void* ptr;
  size_t size; // should always be kBlockSize
  uint8_t lineMap[kLinesPerBlock] = {}; // 256 lines per block for 32kB block
  uint8_t marked;
  uint8_t overflow;
  // 1 bit per 16 bytes/128 bits in the heap
  live_map map;
  ImmixBlock(void* ptr, size_t size) : ptr(ptr), size(size), marked(0), overflow(0) {}

  void setMapBit(const void* p) {
    assert(uintptr_t(p) >= uintptr_t(ptr));
    assert(uintptr_t(p) < uintptr_t(ptr) + size);
    auto bit_pos = ((uintptr_t(p) - uintptr_t(ptr)) / 16);
    map[bit_pos] = true;
  }

  bool testMapBit(const void* p) {
    assert(uintptr_t(p) >= uintptr_t(ptr));
    assert(uintptr_t(p) < uintptr_t(ptr) + size);
    auto bit_pos = ((uintptr_t(p) - uintptr_t(ptr)) / 16);
    return map[bit_pos];
  }
};

///////////////////////////////////////////////////////////////////////////////

/*
 * Allocator for slabs and big blocks.
 */
struct BigHeap {
  BigHeap() {m_pos = -1;}
  bool empty() const {
    return m_slabs.empty() && m_bigs.empty();
  }

  // return true if ptr points into one of the slabs
  bool contains(void* ptr) const;
  bool containsBig(const void* ptr) const;

  void markLineForSmall(const void* p);
  void markLinesForMedium(const void* p, uint32_t size);
  void markBlockContaining(const void* p);

  void resetBlockPointer();
  void freeUnusedBlocks();

  void setMapBit(const void* p);
  void setMapBitSlow(const void* p);
  bool testMapBit(const void* p);
  void dumpMapBits();

  // allocate a MemBlock of at least size bytes, track in m_slabs.
  MemBlock allocSlab(size_t size, bool forOverflow);

  // the next recyclable block
  ImmixBlock getNextRecyclableBlock();

  ImmixBlock currentBlock();

  // allocation api for big blocks. These get a BigNode header and
  // are tracked in m_bigs
  MemBlock allocBig(size_t size, HeaderKind kind);
  MemBlock callocBig(size_t size);
  MemBlock resizeBig(void* p, size_t size);
  void freeBig(void*);

  // free all slabs and big blocks
  void reset();

  // Release auxiliary structures to prepare to be idle for a while
  void flush();

  // allow whole-heap iteration
  template<class Fn> void iterate(Fn);
  // iterate over slabs (immix lines and blocks) only
  template<class Fn> void iterateSlabs(Fn);
  // iterate immix lines with line mark
  template<class Fn> void forEachLine(Fn);

 protected:
  void enlist(BigNode*, HeaderKind kind, size_t size);

 protected:
  std::vector<ImmixBlock> m_slabs;
  int32_t m_pos;
  std::vector<BigNode*> m_bigs;
};

///////////////////////////////////////////////////////////////////////////////

/*
 * ContiguousHeap handles allocations and provides a contiguous address space
 * for requests.
 *
 * To turn on build with CONTIGUOUS_HEAP = 1.
 */
struct ContiguousHeap : BigHeap {
  bool contains(void* ptr) const;

  MemBlock allocSlab(size_t size);

  MemBlock allocBig(size_t size, HeaderKind kind);
  MemBlock callocBig(size_t size);
  MemBlock resizeBig(void* p, size_t size);
  void freeBig(void*);

  void reset();

  void flush();

  ~ContiguousHeap();

 private:
  // Contiguous Heap Pointers
  char* m_base = nullptr;
  char* m_used;
  char* m_end;
  char* m_peak;
  char* m_OOMMarker;
  FreeNode m_freeList;

  // Contiguous Heap Counters
  uint32_t m_requestCount;
  size_t m_heapUsage;
  size_t m_contiguousHeapSize;

 private:
  void* heapAlloc(size_t nbytes, size_t &cap);
  void  createRequestHeap();
};

///////////////////////////////////////////////////////////////////////////////

struct MemoryManager {
  /*
   * Lifetime managed with a ThreadLocalSingleton.  Use MM() to access
   * the current thread's MemoryManager.
   */
  using TlsWrapper = ThreadLocalSingleton<MemoryManager>;

  static void Create(void*);
  static void Delete(MemoryManager*);
  static void OnThreadExit(MemoryManager*);

  /////////////////////////////////////////////////////////////////////////////

  /*
   * Id that is used when registering roots with the memory manager.
   */
  using RootId = size_t;

  /*
   * This is an RAII wrapper to temporarily mask counting allocations from
   * stats tracking in a scoped region.
   *
   * Usage:
   *   MemoryManager::MaskAlloc masker(MM());
   */
  struct MaskAlloc;

  /*
   * An RAII wrapper to suppress OOM checking in a region.
   */
  struct SuppressOOM;

  /////////////////////////////////////////////////////////////////////////////
  // Allocation.

  static uint32_t align(uint32_t bytes);
  static void* align(const void* p);

  /*
   * Return a lower bound estimate of the capacity that will be returned for
   * the requested size.
   */
  static uint32_t estimateCap(uint32_t requested);



  /*
   * Allocate/deallocate a small memory block in a given small size class.
   * You must be able to tell the deallocation function how big the
   * allocation was.
   *
   * The size passed to mallocSmallSize does not need to be an exact
   * size class (although stats accounting may undercount in this
   * case).  The size passed to freeSmallSize must be the exact size
   * that was passed to mallocSmallSize for that allocation.
   *
   * The returned pointer is guaranteed to be 16-byte aligned.
   *
   * Pre: size > 0 && size <= kMaxSmallSize
   */
  void* mallocSmallSize(uint32_t size);
  void freeSmallSize(void* p, uint32_t size);

  /*
   * Allocate/deallocate memory that is too big for the small size classes.
   *
   * Returns a pointer and the actual size of the allocation, which
   * amay be larger than the requested size.  The returned pointer is
   * guaranteed to be 16-byte aligned.
   *
   * The size passed to freeBigSize must either be the size that was
   * passed to mallocBigSize, or the value that was returned as the
   * actual allocation size.
   *
   * Pre: size > kMaxSmallSize
   */
  template<bool callerSavesActualSize>
  MemBlock mallocBigSize(size_t size);
  void freeBigSize(void* vp, size_t size);

  /*
   * Allocate/deallocate objects when the size is not known to be
   * above or below kMaxSmallSize without a runtime check.
   *
   * These functions use the same underlying allocator as
   * malloc{Small,Big}Size, and it is safe to return allocations using
   * one of those apis as long as the appropriate preconditions on the
   * size are met.
   *
   * The size passed to objFree must be the size passed in to
   * objMalloc.
   *
   * Pre: size > 0
   */
  void* objMalloc(size_t size);
  void objFree(void* vp, size_t size);

  /////////////////////////////////////////////////////////////////////////////
  // Cleanup.

  /*
   * Prepare for being idle for a while by releasing or madvising as much as
   * possible.
   */
  void flush();

  /*
   * Release all the request-local allocations.
   *
   * Zeros all the free lists and may return some underlying storage to the
   * system allocator.  This also resets all internally-stored memory usage
   * stats.
   *
   * This is called after sweep in the end-of-request path.
   */
  void resetAllocator();

  /*
   * Reset all runtime options for MemoryManager.
   */
  void resetRuntimeOptions();

  /////////////////////////////////////////////////////////////////////////////
  // Heap introspection.

  /*
   * Return true if there are no allocated slabs.
   */
  bool empty() const;

  /*
   * Whether `p' points into memory owned by `m_heap'.  checkContains() will
   * assert that it does.
   */
  bool contains(void* p) const;
  bool checkContains(void* p) const;

  /////////////////////////////////////////////////////////////////////////////
  // Stats.

  /*
   * Get access to the current memory allocation stats, without refreshing them
   * first.
   */
  MemoryUsageStats& getStatsNoRefresh();

  /*
   * Get most recent stats, updating the tracked stats in the MemoryManager
   * object.
   */
  MemoryUsageStats& getStats();

  /*
   * Get most recent stats data, as one would with getStats(), but without
   * altering the underlying data stored in the MemoryManager.
   *
   * Used for obtaining debug info.
   */
  MemoryUsageStats getStatsCopy();

  /*
   * Open and close respectively a stats-tracking interval.
   *
   * Return whether or not the tracking state was changed as a result of the
   * call.
   */
  bool startStatsInterval();
  bool stopStatsInterval();

  /*
   * How much memory this thread has allocated or deallocated.
   */
  int64_t getAllocated() const;
  int64_t getDeallocated() const;

  /*
   * Reset all stats that are synchronzied externally from the memory manager.
   *
   * Used between sessions and to signal that external sync is now safe to
   * begin (after shared structure initialization that should not be counted is
   * complete.)
   */
  void resetExternalStats();

  /////////////////////////////////////////////////////////////////////////////
  // OOMs.

  /*
   * Whether an allocation of `size' would run the request out of memory.
   *
   * This behaves just like the OOM check in refreshStatsImpl().  If the
   * m_couldOOM flag is already unset, we return false, but if otherwise we
   * would exceed the limit, we unset the flag and register an OOM fatal
   * (though we do not modify the MM's stats).
   */
  bool preAllocOOM(int64_t size);

  /*
   * Unconditionally register an OOM fatal. Still respects the m_couldOOM flag.
   */
  void forceOOM();

  /*
   * Reset whether or not we should raise an OOM fatal if we exceed the memory
   * limit for the request.
   *
   * After an OOM fatal, the memory manager refuses to raise another OOM error
   * until this flag has been reset, to try to avoid getting OOMs during the
   * initial OOM processing.
   */
  void resetCouldOOM(bool state = true);

  /////////////////////////////////////////////////////////////////////////////
  // Sweeping.

  /*
   * Returns true iff a sweep is in progress---i.e., is the current thread
   * running inside a call to MemoryManager::sweep()?
   *
   * It is legal to call this function even when the current thread's
   * MemoryManager may not be set up (i.e. between requests).
   */
  static bool sweeping();

  /*
   * During session shutdown, before resetAllocator(), this phase runs through
   * the sweep lists, running cleanup for anything that needs to run custom
   * tear down logic before we throw away the request-local memory.
   */
  void sweep();

  /*
   * Methods for maintaining dedicated sweep lists of sweepable NativeData
   * objects, APCLocalArray instances, and Sweepables.
   */
  void addNativeObject(NativeNode*);
  void removeNativeObject(NativeNode*);
  void addApcArray(APCLocalArray*);
  void removeApcArray(APCLocalArray*);
  void addSweepable(Sweepable*);

  /////////////////////////////////////////////////////////////////////////////
  // Request profiling.

  /*
   * Trigger heap profiling in the next request.
   *
   * Allocate the s_trigger atomic so that the next request can consume it.  If
   * an unconsumed trigger exists, do nothing and return false; else return
   * true.
   */
  static bool triggerProfiling(const std::string& filename);

  /*
   * Do per-request initialization.
   *
   * Attempt to consume the profiling trigger, and copy it to m_profctx if we
   * are successful.  Also enable jemalloc heap profiling.
   */
  static void requestInit();

  /*
   * Do per-request shutdown.
   *
   * Dump a jemalloc heap profiling, then reset the profiler.
   */
  static void requestShutdown();

  /////////////////////////////////////////////////////////////////////////////

  /*
   * Returns ptr to head node of m_strings linked list. This used by
   * StringData during a reset, enlist, and delist
   */
  StringDataNode& getStringList();

  /*
   * Methods for maintaining maps of root objects keyed by RootIds.
   *
   * The id/object associations are only valid for a single request.  This
   * interface is useful for extensions that cannot physically hold on to a
   * req::ptr, etc. or other handle class.
   */
  template <typename T> RootId addRoot(req::ptr<T>&& ptr);
  template <typename T> RootId addRoot(const req::ptr<T>& ptr);
  template <typename T> req::ptr<T> lookupRoot(RootId tok) const;
  template <typename T> bool removeRoot(const req::ptr<T>& ptr);
  template <typename T> bool removeRoot(const T* ptr);
  template <typename T> req::ptr<T> removeRoot(RootId token);
  template <typename F> void scanRootMaps(F& m) const;
  template <typename F> void scanSweepLists(F& m) const;

  // Opaque type used to allow for quick removal of exception roots. Should be
  // embedded in ExtendedException.
  struct ExceptionRootKey {
    std::size_t m_index = 0;
  };

  // Add/remove exceptions as GC roots.
  void addExceptionRoot(ExtendedException* exn);
  void removeExceptionRoot(ExtendedException* exn);

  /*
   * Heap iterator methods.
   */
  template<class Fn> void iterate(Fn);
  template<class Fn> void iterateSlabs(Fn);   
  template<class Fn> void forEachObject(Fn);
  template<class Fn> void forEachLine(Fn);

  /*
   * Line marking
   */
  void markLineForSmall(const void* p);
  void markLinesForMedium(const void* p, uint32_t size);
  void markBlockContaining(const void* p);

  void goToFirstRecyclableBlock();
  void freeUnusedBlocks();
  /*
   * Run the experimental collector.
   */
  void collect();

  /////////////////////////////////////////////////////////////////////////////

private:
  friend void* req::malloc(size_t nbytes);
  friend void* req::calloc(size_t count, size_t bytes);
  friend void* req::realloc(void* ptr, size_t nbytes);
  friend void  req::free(void* ptr);

  // head node of the doubly-linked list of Sweepables
  struct SweepableList : Sweepable {
    SweepableList() : Sweepable(Init{}) {}
    void sweep() override {}
    void* owner() override { return nullptr; }
  };

  template <typename T>
  using RootMap =
    std::unordered_map<
      RootId,
      req::ptr<T>,
      std::hash<RootId>,
      std::equal_to<RootId>,
      req::Allocator<std::pair<const RootId,req::ptr<T>>>
    >;

  /*
   * Request-local heap profiling context.
   */
  struct ReqProfContext {
    bool flag{false};
    bool prof_active{false};
    bool thread_prof_active{false};
    std::string filename;
  };

  /////////////////////////////////////////////////////////////////////////////

private:
  MemoryManager();
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

private:
  void  updateBigStats();
  void* mallocBig(size_t nbytes);
  void* callocBig(size_t nbytes);
  void* malloc(size_t nbytes);
  void* realloc(void* ptr, size_t nbytes);
  void  free(void* ptr);

  /* immix */
  void* sequentialAllocate(void*& cursor, void* limit, uint32_t bytes);
  void* getNextLineInBlock();
  void* getFreeLines(const ImmixBlock& block, uint32_t start,
                     void*& cursor, void*& limit);
  void* getNextRecyclableBlock();
  void* getFreeBlock(void*& cursor, void*& limit, bool forOverflow);
  void* allocSlowHot(uint32_t bytes);
  void* overflowAlloc(uint32_t bytes);

  static uint32_t bsr(uint32_t x);

  static void threadStatsInit();
  static void threadStats(uint64_t*&, uint64_t*&, size_t*&, size_t&);
  void refreshStats();
  template<bool live> void refreshStatsImpl(MemoryUsageStats& stats);
  void refreshStatsHelperExceeded();
  void refreshStatsHelperStop();

  void resetStatsImpl(bool isInternalCall);

  void logAllocation(void*, size_t);
  void logDeallocation(void*);

  void checkHeap();
  void initFree();

  void dropRootMaps();
  void deleteRootMaps();

  void eagerGCCheck();

  template <typename T>
  typename std::enable_if<
    std::is_base_of<ResourceData,T>::value,
    RootMap<ResourceData>&
  >::type getRootMap() {
    if (UNLIKELY(!m_resourceRoots)) {
      m_resourceRoots = req::make_raw<RootMap<ResourceData>>();
    }
    return *m_resourceRoots;
  }

  template <typename T>
  typename std::enable_if<
    std::is_base_of<ObjectData,T>::value,
    RootMap<ObjectData>&
  >::type getRootMap() {
    if (UNLIKELY(!m_objectRoots)) {
      m_objectRoots = req::make_raw<RootMap<ObjectData>>();
    }
    return *m_objectRoots;
  }

  template <typename T>
  typename std::enable_if<
    std::is_base_of<ResourceData,T>::value,
    const RootMap<ResourceData>&
  >::type getRootMap() const {
    if (UNLIKELY(!m_resourceRoots)) {
      m_resourceRoots = req::make_raw<RootMap<ResourceData>>();
    }
    return *m_resourceRoots;
  }

  template <typename T>
  typename std::enable_if<
    std::is_base_of<ObjectData,T>::value,
    const RootMap<ObjectData>&
  >::type getRootMap() const {
    if (UNLIKELY(!m_objectRoots)) {
      m_objectRoots = req::make_raw<RootMap<ObjectData>>();
    }
    return *m_objectRoots;
  }

  /////////////////////////////////////////////////////////////////////////////

private:
  TRACE_SET_MOD(mm);

  void* m_lineCursor;
  void* m_lineLimit;
  void* m_blockCursor;
  void* m_blockLimit;

  StringDataNode m_strings; // in-place node is head of circular list
  std::vector<APCLocalArray*> m_apc_arrays;
  MemoryUsageStats m_stats;
#if CONTIGUOUS_HEAP
  ContiguousHeap m_heap;
#else
  BigHeap m_heap;
#endif
  std::vector<NativeNode*> m_natives;
  SweepableList m_sweepables;

  mutable RootMap<ResourceData>* m_resourceRoots{nullptr};
  mutable RootMap<ObjectData>* m_objectRoots{nullptr};
  mutable std::vector<ExtendedException*> m_exceptionRoots;

  bool m_sweeping;
  bool m_statsIntervalActive;
  bool m_couldOOM{true};
  bool m_bypassSlabAlloc;

  ReqProfContext m_profctx;
  static std::atomic<ReqProfContext*> s_trigger;

  static void* TlsInitSetup;

#ifdef USE_JEMALLOC
  // pointers to jemalloc-maintained allocation counters
  uint64_t* m_allocated;
  uint64_t* m_deallocated;
  uint64_t m_prevAllocated;
  uint64_t m_prevDeallocated;
  size_t* m_cactive;
  mutable size_t m_cactiveLimit;
  static bool s_statsEnabled;
  static size_t s_cactiveLimitCeiling;
  bool m_enableStatsSync;
#endif
};

//////////////////////////////////////////////////////////////////////

}

#include "hphp/runtime/base/memory-manager-inl.h"

#endif
